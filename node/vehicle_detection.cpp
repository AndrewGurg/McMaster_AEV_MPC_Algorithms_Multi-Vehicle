#define _USE_MATH_DEFINES //M_PI

#include "ros/ros.h"

#include "std_msgs/Bool.h"
#include "std_msgs/String.h"
#include "std_msgs/Int32MultiArray.h"
#include "sensor_msgs/LaserScan.h" //receive msgs from lidar
#include "sensor_msgs/Imu.h" //receive msgs from Imu
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Pose2D.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h" //map localization via AMCL
#include "visualization_msgs/Marker.h" //plot marker line
#include "sensor_msgs/Image.h" // ros image
#include <vesc_msgs/VescStateStamped.h>
#include <std_msgs/Float64.h>

#include <tf/tf.h> //Quaternions
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>


#include "ackermann_msgs/AckermannDriveStamped.h" //Ackermann Steering

#include "nav_msgs/Odometry.h" //Odometer
#include <nav_msgs/OccupancyGrid.h> //Map
#include <f1tenth_simulator/YoloData.h> //Neural Network, vehicle detection msg

#include <string>
#include <vector>


//CV includes
#include <cv_bridge/cv_bridge.h>
#include <librealsense2/rs.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include  <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <realsense2_camera/Extrinsics.h>




//standard and external
#include <stdio.h>
#include <math.h> //cosf
#include <cmath> //M_PI, round
#include <sstream>
#include <algorithm>
#include <random> // Simulating measurements

#include <QuadProg++.hh>
#undef inverse // Remove the conflicting macro
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp> 
#include <nlopt.hpp>
#include <Eigen/Dense>

//C++ will auto typedef float3 data type
int nMPC=0; //Defined outside class to be used in predefined functions for nlopt MPC calculation
int kMPC=0;

int num_cons=9; // Number of inequality constraints used in optimization

struct float3
{
	float x;
	float y;
	float z;
};

struct plane
{
	float A;
	float B;
	float C;
	float D; 
};

struct tf_data
{
	double tf_x=0;
	double tf_y=0;
	double tf_theta=0;
	double tf_time=0;
};

struct vehicle_detection
{
	int init=0; //Whether the KF for this detetction has been initialized
	std::vector<int> bound_box={0,0,0,0}; //The CNN bounding box for the last cycle to try to match appropriately in the case of multi-detection
	//{miny, minx, maxy, maxx}
	std::vector<double> meas={0.0,0.0}; //The most recent measurement of x & y for the vehicle detected
	tf_data meas_tf; //The tf corresponding to the last measurement time, to align frames in KF

	int miss_fr=0; //Consecutive frames missed, if above a set threshold then stop tracking this vehicle
	int last_det=0; //Flag for identifying whether a measurement was made in the last cycle or not, affects how that time's KF works
	//Kalman Filter Parameters
	Eigen::VectorXd state = Eigen::VectorXd::Zero(5); //x, y, theta, vs, delta
	Eigen::MatrixXd cov_P = Eigen::MatrixXd::Zero(5,5); //Initial cov and proc noise are set upon initialization in yolo_callback

	//State, covariance as well as meas, proc noises which can evolve depending on conditions
	Eigen::MatrixXd proc_noise = cov_P;
	Eigen::MatrixXd meas_noise = Eigen::Vector2d(0.01, 0.01).asDiagonal();
	//State transition and observation matrices assumed equal for all detection structs so provided commonly outside struct

};



class GapBarrier 
{
	private:
		ros::NodeHandle nf;


		//Subscriptions
		ros::Subscriber lidar;
		ros::Subscriber image, info, confidence; 
		ros::Subscriber imu;
		ros::Subscriber mux;
		ros::Subscriber vesc_state_sub;
		ros::Subscriber servo_sub;
		ros::Subscriber ext_odom;
		// ros::Subscriber localize;
		ros::Subscriber amcl_sub;
		ros::Subscriber tf_sub;
		ros::Subscriber map_sub;
		ros::Subscriber yolo_sub;

		//More CV data members, used if use_camera is true
		ros::Subscriber depth_img;
		ros::Subscriber color_img;
		ros::Subscriber depth_info;
		ros::Subscriber color_info;
		ros::Subscriber cam_extrinsics;
		ros::Subscriber depth_img_confidence;
		sensor_msgs::LaserScan cv_ranges_msg;
		int cv_rows, cv_cols;
		xt::xarray<int> cv_sample_rows_raw;
		xt::xarray<int> cv_sample_cols_raw;
		
		


		
		//Publications
		ros::Publisher lidar_pub;
		ros::Publisher marker_pub;
		ros::Publisher mpc_marker_pub;
		ros::Publisher wall_marker_pub;
		ros::Publisher lobs;
		ros::Publisher robs;
		ros::Publisher bez_mark;
		ros::Publisher vehicle_detect;
		ros::Publisher vehicle_lidar;
		ros::Publisher driver_pub;
		ros::Publisher cv_ranges_pub;


		
		//topics
		std::string depth_image_topic, depth_info_topic, cv_ranges_topic, depth_index_topic, color_image_topic, color_info_topic, cam_extr_topic,
		depth_points_topic,lidarscan_topic, drive_topic, odom_topic, mux_topic, imu_topic, map_topic, yolo_data_topic, ext_drive_topic;

		// tf frames
		std::string map_frame, base_frame, scan_frame, odom_frame, tf_prefix, ext_prefix;

		// Kalman Filter Parameters
		std::string EKF_mode; 
		// Noise generation
		double dep_std_dev;
		double ang_std_dev;
		std::mt19937 noise_generator;
		std::normal_distribution<double> noise_dep;
		std::normal_distribution<double> noise_ang;

		//time
		double current_time = ros::Time::now().toSec();
		double prev_time = current_time;
		double time_ref = 0.0; 
		double heading_beam_angle;
		double sim_graph_time = 0.0;

		//lidar-preprocessing
		int scan_beams; double right_beam_angle, left_beam_angle;
		double right_beam_angle_MPC, left_beam_angle_MPC;
		int right_ind_MPC, left_ind_MPC;
		int ls_str, ls_end, ls_len_mod, ls_len_mod2; double ls_fov, angle_cen, ls_ang_inc;
		double max_lidar_range, safe_distance;
		double safe_distance_adapt;

		//obstacle point detection
		std::string drive_state; 
		double angle_bl, angle_al, angle_br, angle_ar;
		int n_pts_l, n_pts_r; double max_lidar_range_opt;

		//walls
		double tau;
		std::vector<double> wl0; std::vector<double> wr0;
		int optim_mode;


		//markers
		visualization_msgs::Marker marker;
		visualization_msgs::Marker mpc_marker;
		visualization_msgs::Marker wall_marker;
		visualization_msgs::Marker lobs_marker;
		visualization_msgs::Marker robs_marker;
		visualization_msgs::Marker bez;
		visualization_msgs::Marker vehicle_detect_path;
		visualization_msgs::Marker vehicle_lidar_dir;

		//steering & stop time
		double vel;
		double CenterOffset, wheelbase;
		double stop_distance, stop_distance_decay;
		double k_p, k_d;
		double max_steering_angle;
		double vehicle_velocity; double velocity_zero;
		
		double stopped_time;
		double stop_time1, stop_time2;

		double yaw0, dtheta; double turn_angle; 
		double turn_velocity;

		double max_servo_speed;
		double max_speed=0;
		double min_speed=0;
		double max_accel=0; //Max decel is set to equal

		//MPC parameters
		//int nMPC, kMPC;
		double angle_thresh;
		std::vector<double> deltas, thetas, x_vehicle, y_vehicle;
		double last_delta;
		int num1=0;
		int num2=0;
		int missing_pts=0;
		double velocity_MPC;
		double default_dt;
		int startcheck=0;
		int forcestop=0;

		double speed_to_erpm_gain, speed_to_erpm_offset;
		double steering_angle_to_servo_gain, steering_angle_to_servo_offset;
		std_msgs::Float64 last_servo_state;
		double vel_adapt=1;

		double testx, testy, testtheta;

		//odom and map transforms for map localization and tf of occupancy grid points
		double mapx=0, mapy=0, maptheta=0;
		double odomx=0, odomy=0, odomtheta=0, odomvel=0, odomsteer=0;
		double locx=0, locy=0, loctheta=0;
		double simx=0, simy=0, simtheta=0;
		double robtheta=0;
		std::vector<tf_data> past_tf;

		//MPC MAP localization parameters
		std::vector<std::vector<double>> map_pts;
		int map_saved=0;
		double map_thresh;
		int use_map=0; //Whether we use the pre-defined map as part of MPC

		int yolo_rows; //Rows & columns preset for depth camera
		int yolo_cols;
		std::vector<vehicle_detection> car_detects;

		Eigen::Matrix<double, 2, 5> meas_observability; //Measurement observability matrix, used in KF & same for all

		double lastx=0, lasty=0, lasttheta=0;

		int use_neural_net=0; //Whether one of the neural networks is being used for vehicle detection

		double veh_det_length=0;
		double veh_det_width=0;

		//Bezier MPC Parameters
		int bez_ctrl_pts=0;
		int bez_curv_pts=0;
		double bez_alpha=0;
		int bez_beta=0;
		double bez_min_dist=0;
		double bez_t_end=0;
		double obs_sep=0;
		double max_obs=0;
		double theta_band_smooth=0;
		double theta_band_diff=0;
		double vel_beta=0;
		double stop_dist_decay=0;
		double pot_field_factor_F_QBMPC=0;
		double velocity_factor_F_QBMPC=0;

		ros::Time timestamp_tf1; ros::Time timestamp_tf2;
		ros::Time timestamp_cam1; ros::Time timestamp_cam2;

		//imu
		double imu_roll, imu_pitch, imu_yaw;


		//mux
		int nav_mux_idx; int nav_active; 

		//odom
		double yaw;


		//camera and cv
		
		int use_camera;
		double min_cv_range;
        double max_cv_range;
        double cv_distance_to_lidar;
        double num_cv_sample_rows;
        double num_cv_sample_cols;

        double cv_ground_angle;
        double cv_lidar_range_max_diff;
        double camera_height;
		double camera_min,camera_max;
        double cv_real_to_theo_ground_range_ratio;
        double cv_real_to_theo_ground_range_ratio_near_horizon;
        double cv_ground_range_decay_row;
        double cv_pitch_angle_hardcoded;


		rs2_intrinsics intrinsics_depth;
		rs2_intrinsics intrinsics_color;
		rs2_extrinsics extrinsics;
		bool intrinsics_d_defined; bool intrinsics_c_defined;
		sensor_msgs::Image cv_image_data;
		bool cv_image_data_defined;

		std::vector<sensor_msgs::ImageConstPtr> depth_imgs;

		//ground plane parameters
		float cv_groundplane_max_height; 
		float cv_groundplane_max_distance; 

		

	public:
		
		GapBarrier(){

			nf = ros::NodeHandle("~");
			// topics	
			nf.getParam("depth_image_topic", depth_image_topic);
			nf.getParam("rgb_image_topic", color_image_topic);
			nf.getParam("depth_info_topic", depth_info_topic);
			nf.getParam("rgb_info_topic", color_info_topic);
			nf.getParam("cam_extrinsics_topic", cam_extr_topic);
			nf.getParam("cv_ranges_topic", cv_ranges_topic);
			nf.getParam("depth_index_topic", depth_index_topic);
			nf.getParam("depth_points_topic", depth_points_topic);
			nf.getParam("scan_topic", lidarscan_topic);
			nf.getParam("nav_drive_topic", drive_topic);
			nf.getParam("odom_topic", odom_topic);
			nf.getParam("mux_topic", mux_topic);
			nf.getParam("imu_topic", imu_topic);
			nf.getParam("map_topic", map_topic);
			nf.getParam("yolo_data_topic", yolo_data_topic);

			nf.getParam("speed_to_erpm_gain", speed_to_erpm_gain);
			nf.getParam("speed_to_erpm_offset", speed_to_erpm_offset);

			// Get EKF type and simulated noise
			nf.getParam("EKF_mode", EKF_mode);
			nf.getParam("dep_std_dev", dep_std_dev);
       		nf.getParam("ang_std_dev", ang_std_dev);

			noise_generator = std::mt19937(std::random_device{}());
 			noise_dep = std::normal_distribution<double>(0., dep_std_dev);
 			noise_ang = std::normal_distribution<double>(0., ang_std_dev);


			        // Get the transformation frame names
			nf.getParam("map_frame", map_frame);
			nf.getParam("base_frame", base_frame);
			nf.getParam("scan_frame", scan_frame);
			nf.getParam("odom_frame", odom_frame);
			nf.getParam("tf_prefix", tf_prefix);

			// Get an external vehicle's frame prefix
			if(tf_prefix != ""){
				int ext_car_num = (tf_prefix[tf_prefix.find('/')-1] - '0') + 1;	// Add one to this cars number
				ext_prefix = "racecar" + std::to_string(ext_car_num) + "/";
				
			}
			else{
				ext_prefix = "racecarX/";
			}
			std::cout << "External car prefix: " << ext_prefix + "base_link" << std::endl;

			// Subscribe to external drive topic to get actual velocity and steering angle commands
			ext_drive_topic = ext_prefix + drive_topic; 


			//lidar params
			nf.getParam("scan_beams", scan_beams);
			nf.getParam("right_beam_angle", right_beam_angle);
			nf.getParam("left_beam_angle", left_beam_angle);
			nf.getParam("scan_range", max_lidar_range);
			nf.getParam("safe_distance", safe_distance);
			safe_distance_adapt=safe_distance;

			//lidar init
			right_beam_angle_MPC = right_beam_angle-M_PI; //-M_PI/2;
			left_beam_angle_MPC = left_beam_angle-M_PI; //M_PI/2;
			ls_ang_inc = 2*M_PI/scan_beams;
			ls_str = int(round(scan_beams*right_beam_angle/(2*M_PI)));
			ls_end = int(round(scan_beams*left_beam_angle/(2*M_PI)));
			ls_len_mod = ls_end-ls_str+1;
			ls_fov = ls_len_mod*ls_ang_inc;
			angle_cen = ls_fov/2;
			ls_len_mod2 = 0;	


			//obstacle point detection
			drive_state = "normal";
			nf.getParam("angle_bl", angle_bl);
			nf.getParam("angle_al", angle_al);
			nf.getParam("angle_br", angle_br);
			nf.getParam("angle_ar", angle_ar);
			nf.getParam("n_pts_l", n_pts_l);
			nf.getParam("n_pts_r", n_pts_r);
			nf.getParam("max_lidar_range_opt", max_lidar_range_opt);
			nf.getParam("heading_beam_angle", heading_beam_angle);

			//walls
			nf.getParam("tau", tau); 
			wl0 = {0.0, -1.0}; wr0 = {0.0, 1.0};
			nf.getParam("optim_mode", optim_mode);


			//steering init
			nf.getParam("CenterOffset", CenterOffset);
			nf.getParam("wheelbase", wheelbase);
			nf.getParam("stop_distance", stop_distance);
			nf.getParam("stop_distance_decay", stop_distance_decay);
			nf.getParam("k_p", k_p);
			nf.getParam("k_d", k_d);
			nf.getParam("max_steering_angle", max_steering_angle);
			nf.getParam("vehicle_velocity", vehicle_velocity);
			nf.getParam("velocity_zero",velocity_zero);
			nf.getParam("turn_velocity", turn_velocity);
			nf.getParam("steering_angle_to_servo_gain", steering_angle_to_servo_gain);
    		nf.getParam("steering_angle_to_servo_offset", steering_angle_to_servo_offset);
			nf.getParam("max_steering_vel", max_servo_speed);
			nf.getParam("max_speed",max_speed);
			nf.getParam("min_speed",min_speed);
			nf.getParam("max_accel",max_accel); //Max decel set to equal

			vel = 0.0;

			//MPC parameters
            nf.getParam("nMPC",nMPC);
            nf.getParam("kMPC",kMPC);
			nf.getParam("angle_thresh", angle_thresh);
			nf.getParam("map_thresh", map_thresh);
			nf.getParam("use_map", use_map);

			//MPC init
			default_dt=0.077;
			deltas.resize(nMPC*kMPC,0);
			thetas.resize(nMPC*kMPC,0);
			x_vehicle.resize(nMPC*kMPC,0);
			for(int i=1; i<nMPC*kMPC; i++){
				x_vehicle[i] = x_vehicle[i-1]+vel_adapt*default_dt;
				
			}
			y_vehicle.resize(nMPC*kMPC,0);
			last_delta=0;
			velocity_MPC=vehicle_velocity;
			last_servo_state.data=steering_angle_to_servo_offset;
			
			nf.getParam("yolo_rows", yolo_rows);
			nf.getParam("yolo_cols", yolo_cols);
			nf.getParam("use_neural_net", use_neural_net);
			nf.getParam("veh_det_length", veh_det_length);
			nf.getParam("veh_det_width", veh_det_width);

			meas_observability = Eigen::Matrix<double, 2, 5>::Zero();
			meas_observability(0, 0) = 1;  // (1,1) in 1-based indexing
			meas_observability(1, 1) = 1;  // (2,2) in 1-based indexing

			//Bezier MPC Parameters
			nf.getParam("bez_ctrl_pts", bez_ctrl_pts);
			nf.getParam("bez_curv_pts", bez_curv_pts);
			nf.getParam("bez_alpha", bez_alpha);
			nf.getParam("bez_beta", bez_beta);
			nf.getParam("bez_min_dist", bez_min_dist);
			nf.getParam("bez_t_end", bez_t_end);
			nf.getParam("obs_sep", obs_sep);
			nf.getParam("max_obs", max_obs);

			// Fast Bezier MPC Parameters
			nf.getParam("theta_band_smooth", theta_band_smooth);
			nf.getParam("theta_band_diff", theta_band_diff);
			nf.getParam("vel_beta", vel_beta);
			nf.getParam("stop_dist_decay", stop_dist_decay);
			nf.getParam("pot_field_factor_F_QBMPC", pot_field_factor_F_QBMPC);
			nf.getParam("velocity_factor_F_QBMPC", velocity_factor_F_QBMPC);

			//timing
			nf.getParam("stop_time1", stop_time1);
			nf.getParam("stop_time2", stop_time2);
			stopped_time = 0.0;

			//camera
			nf.getParam("use_camera", use_camera);


			//imu init
			yaw0 = 0.0; dtheta = 0.0;

			//mux init
			nf.getParam("nav_mux_idx", nav_mux_idx);
			nav_active = 0;

			//cv
			ros::param::get("~min_cv_range", min_cv_range);
            ros::param::get("~max_cv_range", max_cv_range);
            ros::param::get("~cv_distance_to_lidar", cv_distance_to_lidar);
            ros::param::get("~num_cv_sample_rows", num_cv_sample_rows);
            ros::param::get("~num_cv_sample_cols",num_cv_sample_cols);

            ros::param::get("~cv_ground_angle", cv_ground_angle);
            ros::param::get("~cv_lidar_range_max_diff",cv_lidar_range_max_diff);
            ros::param::get("~camera_height",camera_height);
			ros::param::get("~camera_min",camera_min);
			ros::param::get("~camera_max", camera_max);
            ros::param::get("~cv_real_to_theo_ground_range_ratio",cv_real_to_theo_ground_range_ratio);
            ros::param::get("~cv_real_to_theo_ground_range_ratio_near_horizon",cv_real_to_theo_ground_range_ratio_near_horizon);
            ros::param::get("~cv_ground_range_decay_row",cv_ground_range_decay_row);
            ros::param::get("~cv_pitch_angle_hardcoded",cv_pitch_angle_hardcoded);

			ros::param::get("~cv_groundplane_max_height", cv_groundplane_max_height);
			ros::param::get("~cv_groundplane_max_distance", cv_groundplane_max_distance); 



			intrinsics_d_defined= false; intrinsics_c_defined= false;
        	cv_image_data_defined= false;

			//subscriptions
			lidar = nf.subscribe(lidarscan_topic,1, &GapBarrier::lidar_callback, this);
			imu = nf.subscribe(imu_topic,1, &GapBarrier::imu_callback, this);
			mux = nf.subscribe(mux_topic,1, &GapBarrier::mux_callback, this);
			vesc_state_sub= nf.subscribe("/sensors/core", 1, &GapBarrier::vesc_callback, this);
			servo_sub= nf.subscribe("/sensors/servo_position_command", 1,&GapBarrier::servo_callback, this);
			//odom = nf.subscribe(odom_topic,1, &GapBarrier::odom_callback, this);
			// localize = nf.subscribe("/pose_stamped",1, &GapBarrier::localize_callback, this);
			amcl_sub = nf.subscribe("/amcl_pose", 1, &GapBarrier::amcl_callback, this);
			tf_sub = nf.subscribe("/tf", 20, &GapBarrier::tf_callback, this);
			map_sub = nf.subscribe(map_topic, 1, &GapBarrier::map_callback, this);
			yolo_sub=nf.subscribe(yolo_data_topic, 1, &GapBarrier::yolo_callback, this);
			
			ext_odom = nf.subscribe("/"+ext_prefix+"odom",1, &GapBarrier::ext_odom_callback, this);


			//publications
			//lidar_pub = nf.advertise<std_msgs::Int32MultiArray>("chatter", 1000);
			marker_pub = nf.advertise<visualization_msgs::Marker>("wall_markers",2);
			mpc_marker_pub = nf.advertise<visualization_msgs::Marker>("mpc_markers",2);
			wall_marker_pub=nf.advertise<visualization_msgs::Marker>("walls",2);
			lobs=nf.advertise<visualization_msgs::Marker>("lobs",2);
			robs=nf.advertise<visualization_msgs::Marker>("robs",2);
			bez_mark=nf.advertise<visualization_msgs::Marker>("bez",2);
			vehicle_detect=nf.advertise<visualization_msgs::Marker>("vehicle_detect",2);
			vehicle_lidar=nf.advertise<visualization_msgs::Marker>("vehicle_lidar",2);

			driver_pub = nf.advertise<ackermann_msgs::AckermannDriveStamped>(drive_topic, 1);

			// Empty Files for Graphing Data
			FILE *file_states= fopen("/home/gjsk/catkin_ws/Sim_Data/states.txt", "w");
			fclose(file_states);
			FILE *file_var= fopen("/home/gjsk/catkin_ws/Sim_Data/variance.txt", "w");
			fclose(file_var);
			FILE *file_nees= fopen("/home/gjsk/catkin_ws/Sim_Data/NEES_NIS.txt", "w");
			fclose(file_nees);

			if(use_camera)
			{
				cv_ranges_msg= sensor_msgs::LaserScan(); //call constructor
				cv_ranges_msg.header.frame_id= scan_frame;
				cv_ranges_msg.angle_increment= this->ls_ang_inc; 
				cv_ranges_msg.time_increment = 0;
				cv_ranges_msg.range_min = 0;
				cv_ranges_msg.range_max = this->max_lidar_range;
				cv_ranges_msg.angle_min = 0;
				cv_ranges_msg.angle_max = 2*M_PI;

				cv_ranges_pub=nf.advertise<sensor_msgs::LaserScan>(cv_ranges_topic,1);
				
				depth_img=nf.subscribe(depth_image_topic,1, &GapBarrier::imageDepth_callback,this);
				depth_info=nf.subscribe(depth_info_topic,1, &GapBarrier::imageDepthInfo_callback,this);
				depth_img_confidence=nf.subscribe("/camera/confidence/image_rect_raw",1, &GapBarrier::confidenceCallback, this);
				color_img=nf.subscribe(color_image_topic,1, &GapBarrier::imageColor_callback,this);
				color_info=nf.subscribe(color_info_topic,1, &GapBarrier::imageColorInfo_callback,this);
				cam_extrinsics=nf.subscribe(cam_extr_topic,1, &GapBarrier::camExtrinsics_callback,this);
			}

		}



		/// ---------------------- GENERAL HELPER FUNCTIONS ----------------------


		int equiv_sign(double qt){
			if(qt < 0) return -1;
			else if (qt == 0 ) return 0;
			else return 1;
		}


		int arg_max(std::vector<float> ranges){

			int idx = 0;

			for(int i =1; i < int(ranges.size()); ++i){
				if(ranges[idx] < ranges[i]) idx = i;
			}

			return idx;


		}


		std::string getOdom() const { return odom_topic; }
		int getRightBeam() const { return right_beam_angle;}
		std::string getLidarTopic() const { return lidarscan_topic;}


		/// ---------------------- MAIN FUNCTIONS ----------------------

		void tf_callback(const tf2_msgs::TFMessage::ConstPtr& msg){ //Update the localization transforms
			int updated=0;
			 for (const geometry_msgs::TransformStamped& transform : msg->transforms)
			{
				if (transform.header.frame_id == odom_frame && transform.child_frame_id == base_frame)
				{
					odomx=transform.transform.translation.x;
					odomy=transform.transform.translation.y;
					// 		transform.transform.translation.z);
					double x=transform.transform.rotation.x;
					double y=transform.transform.rotation.y;
					double z=transform.transform.rotation.z;
					double w=transform.transform.rotation.w;
					odomtheta = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
					updated=1;

					timestamp_tf1 = transform.header.stamp;

					tf_data new_tf;
					new_tf.tf_x=odomx; new_tf.tf_y=odomy; new_tf.tf_theta=odomtheta; new_tf.tf_time=timestamp_tf1.toSec();
					past_tf.push_back(new_tf);

					if(past_tf.size()>10) past_tf.erase(past_tf.begin());

				}
				else if (transform.header.frame_id == map_frame && transform.child_frame_id == odom_frame)
				{
					mapx=transform.transform.translation.x;
					mapy=transform.transform.translation.y;
					// 		transform.transform.translation.z);
					double x=transform.transform.rotation.x;
					double y=transform.transform.rotation.y;
					double z=transform.transform.rotation.z;
					double w=transform.transform.rotation.w;
					maptheta = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
					updated=1;
				}

				else if (transform.header.frame_id == map_frame && transform.child_frame_id == (ext_prefix + "base_link")) //Simulation detection of other vehicle
				{
					//Just for the one vehicle detection case
					double robx=transform.transform.translation.x;
					double roby=transform.transform.translation.y;
					// 		transform.transform.translation.z);
					double x=transform.transform.rotation.x;
					double y=transform.transform.rotation.y;
					double z=transform.transform.rotation.z;
					double w=transform.transform.rotation.w;
					robtheta = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));

					// Adding noise to external vehicle position measurements
					robx += noise_dep(noise_generator);
					roby += noise_dep(noise_generator);
					robtheta += noise_ang(noise_generator);
					double detx=(robx-simx)*cos(simtheta)+(roby-simy)*sin(simtheta);
					double dety=-(robx-simx)*sin(simtheta)+(roby-simy)*cos(simtheta);

					tf_data new_tf;
					new_tf.tf_x=simx; new_tf.tf_y=simy; new_tf.tf_theta=simtheta; new_tf.tf_time=ros::Time::now().toSec();
					if(car_detects.size()<1){
						vehicle_detection new_det;
						new_det.bound_box={0,0,20,20}; //ymin, xmin, ymax, xmax PLACEHOLDERS
						new_det.meas={detx,dety};
						new_det.last_det=1;
						new_det.meas_tf=new_tf;
						new_det.cov_P(0,0)=0.01; new_det.cov_P(1,1)=0.01; new_det.cov_P(2,2)=std::pow(5 * M_PI / 180, 2);
						new_det.cov_P(3,3)=2; new_det.cov_P(4,4)=std::pow(5 * M_PI / 180, 2);
						new_det.proc_noise=new_det.cov_P;
		
						car_detects.push_back(new_det);
					}
					else{
						car_detects[0].bound_box={0,0,20,20}; //ymin, xmin, ymax, xmax PLACEHOLDERS
						car_detects[0].meas={detx,dety};
						car_detects[0].last_det=1; //Detected in this round
						car_detects[0].meas_tf=new_tf;
					}
				}
				else if(transform.header.frame_id == ext_prefix+"front_left_hinge"){ // Receive actual steering angle of external vehicle
					double x=transform.transform.rotation.x;
					double y=transform.transform.rotation.y;
					double z=transform.transform.rotation.z;
					double w=transform.transform.rotation.w;
					odomsteer = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
				}
				else if(transform.header.frame_id == map_frame && transform.child_frame_id == base_frame){ //This is for simulation only
					simx=transform.transform.translation.x;
					simy=transform.transform.translation.y;
					// 		transform.transform.translation.z);
					double x=transform.transform.rotation.x;
					double y=transform.transform.rotation.y;
					double z=transform.transform.rotation.z;
					double w=transform.transform.rotation.w;
					simtheta = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
				}
			}
			if(updated){
				locx=mapx+cos(maptheta)*odomx-sin(maptheta)*odomy;
				locy=mapy+sin(maptheta)*odomx+cos(maptheta)*odomy;
				loctheta=maptheta+odomtheta;
				while (loctheta > M_PI) loctheta -= 2 * M_PI;
    			while (loctheta < -M_PI) loctheta += 2 * M_PI;
				//THIS CONVERTS BASE_LINK POSITION INTO MAP FRAME
				//Can find the inverse from base_link to map only when required in MPC function so map frame can be transformed to base_link again
			}


		}

		void ext_odom_callback(const nav_msgs::Odometry::ConstPtr& odom_msg){
			// Read velocity information from odometer
			odomvel = odom_msg->twist.twist.linear.x;
		}

		void map_callback(const nav_msgs::OccupancyGrid & map_msg) {
			
			if(use_map && map_saved==0){ //Upon startup, save the map once and for full run
				for (int i=0;i<map_msg.info.width*map_msg.info.height;i++){
					if(map_msg.data[i]==100){
						int add_pt=1;
						for(int j=0;j<map_pts.size();j++){
							if(pow(map_pts[j][0]-map_msg.info.origin.position.x-i%map_msg.info.width*map_msg.info.resolution,2)+pow(map_pts[j][1]-map_msg.info.origin.position.y-i/map_msg.info.width*map_msg.info.resolution,2)<map_thresh){
								add_pt=0; //If the points are too close, don't add in order to reduce computation load
								break;
							}
						}
						if(add_pt){
							map_pts.push_back({map_msg.info.origin.position.x+i%map_msg.info.width*map_msg.info.resolution,map_msg.info.origin.position.y+i/map_msg.info.width*map_msg.info.resolution});
						}
					}
				}
				map_saved=1;
			}
			
		}

		void amcl_callback(const geometry_msgs::PoseWithCovarianceStamped & amcl_msg){
			// printf("cov:%lf, %lf, %lf\n",amcl_msg.pose.covariance[0],amcl_msg.pose.covariance[7],amcl_msg.pose.covariance[35]);
			// printf("pose:%lf, %lf, %lf\n",amcl_msg.pose.pose.position.x,amcl_msg.pose.pose.position.y,2*atan2(amcl_msg.pose.pose.orientation.z, amcl_msg.pose.pose.orientation.w));
			
		}

		void yolo_callback(const f1tenth_simulator::YoloData & yolo_msg){ //Process all other vehicle detections for the KF
			if(!cv_image_data_defined || !intrinsics_c_defined) return; //Check if initial data exists


			if(yolo_msg.classes.size()==0) return; //No detections

			if(depth_imgs.size()<1) return;
			int depth_ind=depth_imgs.size()-1; double mindiff=100;
			for (int i=0;i<depth_imgs.size();i++){ //Find matching depth image to the rgb, based on closest reported timestamp
				if(mindiff>std::abs(depth_imgs[i]->header.stamp.toSec()-yolo_msg.time)){
					mindiff=std::abs(depth_imgs[i]->header.stamp.toSec()-yolo_msg.time);
					depth_ind=i;
				}
			}

			tf_data my_tf;

			for (int mo=past_tf.size()-1; mo>0;mo--){ //Find the tf corresponding to the closest depth image, for finding right frame in KF
				if(past_tf[mo].tf_time>depth_imgs[depth_ind]->header.stamp.toSec() && past_tf[mo-1].tf_time<depth_imgs[depth_ind]->header.stamp.toSec()){
					if(std::abs(past_tf[mo].tf_time-depth_imgs[depth_ind]->header.stamp.toSec())<std::abs(past_tf[mo-1].tf_time-depth_imgs[depth_ind]->header.stamp.toSec())){
						my_tf=past_tf[mo];
					}
					else{
						my_tf=past_tf[mo-1];
					}
					break;
				}
			}		

			cv::Mat cv_image1=(cv_bridge::toCvCopy(depth_imgs[depth_ind],depth_imgs[depth_ind]->encoding))->image;

			std::vector<float> cv_point(3);
			//Multi-vehicle tracking, distinguishing between detections
			std::vector<int> temp_vec(car_detects.size(), -1); //yolo detections of already identified detections
			std::vector<int> no_det; //yolo detections of not pre-identified detections
			std::vector<std::vector<double>> yolo_xy; //x and y measurements of depth from yolo detection
			

			for (int i=0;i<yolo_msg.classes.size();i++){
				int num=-1; float dist=100000;
				const std::string& class_name = yolo_msg.classes[i];
				if(class_name=="car"){
					
					yolo_xy.push_back(depth_calc(cv_image1, yolo_msg, i));

					for (int j=0; j<car_detects.size();j++){ //Compare midpoints of bounding boxes, find minimum
						double x_d=pow((car_detects[j].bound_box[1]+car_detects[j].bound_box[3])/2-(yolo_msg.rectangles[4*i+1]+yolo_msg.rectangles[4*i+3])/2,2);
						double y_d=pow((car_detects[j].bound_box[0]+car_detects[j].bound_box[2])/2-(yolo_msg.rectangles[4*i]+yolo_msg.rectangles[4*i+2])/2,2);
						if(sqrt(x_d+y_d)<dist && sqrt(x_d+y_d)<141 && temp_vec[j]==-1){ //Find closest match, ensure we aren't matching two yolos to the same detection and threshold dist must be below
							dist=sqrt(x_d+y_d);
							num=j;
						}
					}
					if(num!=-1){
						temp_vec[num]=i; //ith detection is mapped
					}
					else no_det.push_back(i);

					double xorigin=cos(odomtheta)*(yolo_xy[0][0])-sin(odomtheta)*yolo_xy[0][1]+odomx;
					double yorigin=sin(odomtheta)*(yolo_xy[0][0])+cos(odomtheta)*yolo_xy[0][1]+odomy;

				}
			}

			//For all existing detections, provide the x and y of the depth measurement
			for (int q=0; q<car_detects.size();q++){
				int i=temp_vec[q];
				if(temp_vec[q]!=-1 &&yolo_xy[i][0]!=0 && yolo_xy[i][1]!=0){ //Detection found
					car_detects[q].bound_box={yolo_msg.rectangles[4*i],yolo_msg.rectangles[4*i+1],yolo_msg.rectangles[4*i+2],yolo_msg.rectangles[4*i+3]}; //ymin, xmin, ymax, xmax
					car_detects[q].meas={yolo_xy[i][0],yolo_xy[i][1]};
					car_detects[q].last_det=1; //Detected in this round
					car_detects[q].meas_tf=my_tf;
				}
			}

			//For all new detections, create structs appended to the vector (deletions should take place in lidar callback)
			for (int q=0; q<no_det.size();q++){
				int i=no_det[q];
				if(yolo_xy[i][0]!=0 && yolo_xy[i][1]!=0){
					vehicle_detection new_det;
					new_det.bound_box={yolo_msg.rectangles[4*i],yolo_msg.rectangles[4*i+1],yolo_msg.rectangles[4*i+2],yolo_msg.rectangles[4*i+3]}; //ymin, xmin, ymax, xmax
					new_det.meas={yolo_xy[i][0],yolo_xy[i][1]};
					new_det.last_det=1;
					new_det.meas_tf=my_tf;
					new_det.cov_P(0,0)=0.05; new_det.cov_P(1,1)=0.05; new_det.cov_P(2,2)=std::pow(45 * M_PI / 180, 2);
					new_det.cov_P(3,3)=2; new_det.cov_P(4,4)=std::pow(45 * M_PI / 180, 2);
					new_det.proc_noise=new_det.cov_P;
	
					car_detects.push_back(new_det);
				}	
			}
		}


		std::vector<double> depth_calc(cv::Mat cv_image1, const f1tenth_simulator::YoloData & yolo_msg, int det_ind){
			std::vector<float> cv_point(3);
			std::vector<float> cv_point1(3);
			std::vector<float> col_point(3);

			//If close to a border, select an average closer to the edge to ensure the car is detected, not background
			float x_start=0.45; float x_end=0.55; float y_start=0.45; float y_end=0.55;
			if(yolo_msg.rectangles[4*det_ind+1]<10 && yolo_msg.rectangles[4*det_ind+3]<630) x_start=0.25, x_end=0.35; //Left edge
			else if(yolo_msg.rectangles[4*det_ind+1]>10 && yolo_msg.rectangles[4*det_ind+3]>630) x_start=0.65, x_end=0.75; //Right edge

			if(yolo_msg.rectangles[4*det_ind]<10 && yolo_msg.rectangles[4*det_ind+2]<470) y_start=0.25, y_end=0.35; //Top edge
			else if(yolo_msg.rectangles[4*det_ind]>10 && yolo_msg.rectangles[4*det_ind+2]>470) y_start=0.65, y_end=0.75; //Bottom edge
			
			int x_true=yolo_msg.rectangles[4*det_ind+1]*(1-x_start)+yolo_msg.rectangles[4*det_ind+3]*x_start; int y_true=yolo_msg.rectangles[4*det_ind]*(1-y_start)+yolo_msg.rectangles[4*det_ind+2]*y_start;
			int x_cv=x_true/2; int y_cv=y_true/2;
			float depth_pixel[2] = {(float) x_cv, (float) y_cv};
			float color_pixel[2] = {(float) 0, (float) 0};
			while (color_pixel[0]<=x_true || color_pixel[1]<=y_true){
				if((cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000>0.01){
					rs2_deproject_pixel_to_point(cv_point1.data(), &intrinsics_depth, depth_pixel, (cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000);			
					rs2_transform_point_to_point(col_point.data(),&extrinsics,cv_point1.data());
					rs2_project_point_to_pixel(color_pixel,&intrinsics_color,col_point.data());
					if(color_pixel[0]<=x_true) depth_pixel[0]+=std::max(1,(int)((x_true-color_pixel[0])/2));
					if(color_pixel[1]<=y_true) depth_pixel[1]+=std::max(1,(int)((y_true-color_pixel[1])/2));
						
				}
				else{
					if(color_pixel[0]<=x_true) depth_pixel[0]++;
					if(color_pixel[1]<=y_true) depth_pixel[1]++;
				}
			}
			//Average depth calculation
			int count=0;
			double av_depth=0;
			double av_x=0; double av_y=0;
			float depth_pixel_start[2]; depth_pixel_start[0]=depth_pixel[0]; depth_pixel_start[1]=depth_pixel[1];

			while(color_pixel[0]<yolo_msg.rectangles[4*det_ind+1]*(1-x_end)+yolo_msg.rectangles[4*det_ind+3]*x_end){
				
				while(color_pixel[1]<yolo_msg.rectangles[4*det_ind]*(1-y_end)+yolo_msg.rectangles[4*det_ind+2]*y_end){
					if((cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000>0.01){
						av_depth+=(cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000;
						rs2_deproject_pixel_to_point(cv_point.data(), &intrinsics_depth, depth_pixel, (cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000);
						av_x=av_x+cv_point[2]+cv_distance_to_lidar; av_y=av_y-cv_point[0];
						count++;
					}
					depth_pixel[1]++;
					rs2_deproject_pixel_to_point(cv_point1.data(), &intrinsics_depth, depth_pixel, (cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000);			
					rs2_transform_point_to_point(col_point.data(),&extrinsics,cv_point1.data());
					rs2_project_point_to_pixel(color_pixel,&intrinsics_color,col_point.data());

				}
				depth_pixel[0]++; depth_pixel[1]=depth_pixel_start[1];
				rs2_deproject_pixel_to_point(cv_point1.data(), &intrinsics_depth, depth_pixel, (cv_image1.ptr<uint16_t>((int)depth_pixel[1])[(int)depth_pixel[0]])/(float)1000);			
				rs2_transform_point_to_point(col_point.data(),&extrinsics,cv_point1.data());
				rs2_project_point_to_pixel(color_pixel,&intrinsics_color,col_point.data());
			}

			av_depth=av_depth/count;
			av_x=av_x/count; av_y=av_y/count;
			std::vector<double> det_xy; det_xy.push_back(0); det_xy.push_back(0);
			if(count>0){
				det_xy[0]=av_x; det_xy[1]=av_y;
			}
			// else printf("No points found, can't update measurement\n");

			return det_xy;

		}


		void mux_callback(const std_msgs::Int32MultiArrayConstPtr& data){nav_active = data->data[nav_mux_idx]; }

		void servo_callback(const std_msgs::Float64 & servo){
			last_servo_state=servo;
		}

		void vesc_callback(const vesc_msgs::VescStateStamped & state){
        	vel_adapt = std::max(-( state.state.speed - speed_to_erpm_offset ) / speed_to_erpm_gain,0.1);
			last_delta = ( last_servo_state.data - steering_angle_to_servo_offset) / steering_angle_to_servo_gain;
		}

		void imu_callback(const sensor_msgs::Imu::ConstPtr& data){

				tf::Quaternion myQuaternion(
				data->orientation.x,
				data->orientation.y,
				data->orientation.z,
				data->orientation.w);
			
			tf::Matrix3x3 m(myQuaternion);
			m.getRPY(imu_roll, imu_pitch, imu_yaw);

		}


		
		void imageDepth_callback( const sensor_msgs::ImageConstPtr & img)
		{
			
			timestamp_cam1=img->header.stamp;
			
			if(intrinsics_d_defined)
			{
				if(depth_imgs.size()>2) depth_imgs.erase(depth_imgs.begin());
				depth_imgs.push_back(img);
				//Unsure how copy constructor behaves, therefore manually copyied all data members
				cv_image_data.header= img->header; 
				cv_image_data.height=img->height;
				cv_image_data.width=img->width;
				cv_image_data.encoding=img->encoding;
				cv_image_data.is_bigendian=img->is_bigendian;
				cv_image_data.step=img->step;
				cv_image_data.data=std::vector<uint8_t>(img->data.begin(), img->data.end());
				cv_image_data_defined=true;
			}
			else
			{
				cv_image_data_defined=false;
			}

		}

		void imageDepthInfo_callback(const sensor_msgs::CameraInfoConstPtr & cameraInfo)
		{
			//intrinsics is a struct of the form:
			/*
			int           width; 
			int           height
			float         ppx;   
			float         ppy;
			float         fx;
			float         fy;   
			rs2_distortion model;
			float coeffs[5];
			*/
			if(intrinsics_d_defined){ return; }

			//std::cout << "Defining Intrinsics" <<std::endl;

            intrinsics_depth.width = cameraInfo->width;
            intrinsics_depth.height = cameraInfo->height;
            intrinsics_depth.ppx = cameraInfo->K[2];
            intrinsics_depth.ppy = cameraInfo->K[5];
            intrinsics_depth.fx = cameraInfo->K[0];
            intrinsics_depth.fy = cameraInfo->K[4];
			
            if (cameraInfo->distortion_model == "plumb_bob") 
			{
				intrinsics_depth.model = RS2_DISTORTION_BROWN_CONRADY;   
			}
               
            else if (cameraInfo->distortion_model == "equidistant")
			{
				intrinsics_depth.model = RS2_DISTORTION_KANNALA_BRANDT4;
			}
            for(int i=0; i<5; i++)
			{
				intrinsics_depth.coeffs[i]=cameraInfo->D[i];
			}
			intrinsics_d_defined=true;

			cv_rows=intrinsics_depth.height;
			cv_cols=intrinsics_depth.width;

			//define pixels that will be sampled in each row and column, spaced evenly by linspace function

			cv_sample_rows_raw= xt::linspace<int>(0, cv_rows-1, num_cv_sample_rows);
			cv_sample_cols_raw= xt::linspace<int>(0, cv_cols-1, num_cv_sample_cols);


		}
		//Realsense D435 has no confidence data
		void confidenceCallback(const sensor_msgs::ImageConstPtr & data)
		{
			/*
			cv::Mat cv_image=(cv_bridge::toCvCopy(data,data->encoding))->image; 
			auto grades= cv::bitwise_and(cv_image >> 4, cv::Scalar(0x0f));
			*/



		}

		void imageColor_callback( const sensor_msgs::ImageConstPtr & img)
		{
			
		}


		void imageColorInfo_callback(const sensor_msgs::CameraInfoConstPtr & cameraInfo)
		{
			//intrinsics is a struct of the form:
			/*
			int           width; 
			int           height
			float         ppx;   
			float         ppy;
			float         fx;
			float         fy;   
			rs2_distortion model;
			float coeffs[5];
			*/
			if(intrinsics_c_defined){ return; }

			//std::cout << "Defining Intrinsics" <<std::endl;

            intrinsics_color.width = cameraInfo->width;
            intrinsics_color.height = cameraInfo->height;
            intrinsics_color.ppx = cameraInfo->K[2];
            intrinsics_color.ppy = cameraInfo->K[5];
            intrinsics_color.fx = cameraInfo->K[0];
            intrinsics_color.fy = cameraInfo->K[4];
			
            if (cameraInfo->distortion_model == "plumb_bob") 
			{
				intrinsics_color.model = RS2_DISTORTION_BROWN_CONRADY;   
			}
               
            else if (cameraInfo->distortion_model == "equidistant")
			{
				intrinsics_color.model = RS2_DISTORTION_KANNALA_BRANDT4;
			}
            for(int i=0; i<5; i++)
			{
				intrinsics_color.coeffs[i]=cameraInfo->D[i];
			}
			intrinsics_c_defined=true;
		}


		void camExtrinsics_callback(const realsense2_camera::Extrinsics::ConstPtr &msg)
		{
			cv::Mat extrinsics1 = cv::Mat::eye(4, 4, CV_64F);
			// Extract rotation matrix
			cv::Mat rotation = cv::Mat(3, 3, CV_64F, const_cast<double *>(msg->rotation.data()));
			rotation.convertTo(rotation, CV_64F);

			// Extract translation vector
			cv::Mat translation = cv::Mat(3, 1, CV_64F, const_cast<double *>(msg->translation.data()));
			translation.convertTo(translation, CV_64F);

			// Update global extrinsics matrix
			extrinsics1.at<double>(0, 0) = rotation.at<double>(0, 0); extrinsics1.at<double>(0, 1) = rotation.at<double>(0, 1); extrinsics1.at<double>(0, 2) = rotation.at<double>(0, 2);
			extrinsics1.at<double>(1, 0) = rotation.at<double>(1, 0); extrinsics1.at<double>(1, 1) = rotation.at<double>(1, 1); extrinsics1.at<double>(1, 2) = rotation.at<double>(1, 2);
			extrinsics1.at<double>(2, 0) = rotation.at<double>(2, 0); extrinsics1.at<double>(2, 1) = rotation.at<double>(2, 1); extrinsics1.at<double>(2, 2) = rotation.at<double>(2, 2);

			extrinsics1.at<double>(0, 3) = translation.at<double>(0, 0); extrinsics1.at<double>(1, 3) = translation.at<double>(1, 0); extrinsics1.at<double>(2, 3) = translation.at<double>(2, 0);

			//Matrix to rs2:extrinsics
			if (extrinsics1.rows == 4 && extrinsics1.cols == 4) {
				// Copy rotation (3x3)
				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						extrinsics.rotation[i * 3 + j] = extrinsics1.at<double>(i, j);
					}
				}

				// Copy translation (3)
				for (int i = 0; i < 3; ++i) {
					extrinsics.translation[i] = extrinsics1.at<double>(i, 3);
				}
			}

		}




		plane fit_groundplane(std::vector<float3> points)
		{

            float3 sum = { 0,0,0 };
            for (auto point : points)
			{
				sum.x+=point.x;
				sum.y+=point.y;
				sum.z+=point.z;
			}

            float3 centroid = {sum.x / float(points.size()), sum.y/ float(points.size()), sum.z/ float(points.size())};

            double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
            for (auto point : points) 
			{
                float3 temp = {point.x - centroid.x, point.y - centroid.y, point.z - centroid.z};
                xx += temp.x * temp.x;
                xy += temp.x * temp.y;
                xz += temp.x * temp.z;
                yy += temp.y * temp.y;
                yz += temp.y * temp.z;
                zz += temp.z * temp.z;
            }

            //double det_x = yy*zz - yz*yz;
            double det_y = xx*zz - xz*xz;
            //double det_z = xx*yy - xy*xy;


			//cramers rule solutions
           //double det_max = std::max({ det_x, det_y, det_z });
            //if (det_max <= 0) return{ 0, 0, 0, 0 };


			float3 dir{};

            /*if (det_max == det_x)
            {
                float a = static_cast<float>((xz*yz - xy*zz) / det_x);
                float b = static_cast<float>((xy*yz - xz*yy) / det_x);
                dir = { 1, a, b };
            }*/
            //else if (det_max == det_y)
            //{
            float a = static_cast<float>((yz*xz - xy*zz) / det_y);
            float b = static_cast<float>((xy*xz - yz*xx) / det_y);
            dir = { a, 1, b };
            //}
            /*else
            {
                float a = static_cast<float>((yz*xy - xz*yy) / det_z);
                float b = static_cast<float>((xz*xy - yz*xx) / det_z);
                dir = { a, b, 1 };
            }*/


			//normalize dir
			
			float mag= std::pow( std::pow(dir.x,2)+std::pow(dir.y,2)+std::pow(dir.z,2), 0.5 );
			//(x^2+y^2+z^2)^0.5

			dir.x=dir.x/mag, dir.y=dir.y/mag, dir.z=dir.z/mag;


			//return plane
			plane result;
			result.A = dir.x, result.B=dir.y, result.C= dir.z;
			result.D= -(dir.x*centroid.x + dir.y*centroid.y + dir.z*centroid.z);


			

			//std::cout << "Ground Plane. " << "A= " << result.A << ". B= " <<result.B << ". C= " <<result.C << ". D= " <<result.D <<std::endl;
			return result;
		}

		plane compute_groundplane(cv::Mat cv_image)
		{
			std::vector<float3> plane_points; //store all points close to the ground
			for(int i=0 ; i < (int)cv_sample_cols_raw.size() ; i++)
			{
				int col= cv_sample_cols_raw[i];

				for(int j=0; j < (int)cv_sample_rows_raw.size() ; j++)
				{
					int row=cv_sample_rows_raw[j];

					float depth= (cv_image.ptr<uint16_t>(row)[col])/(float)1000;
				
					if(depth > max_cv_range || depth < min_cv_range)
					{
						continue;
					}
					
					std::vector<float> cv_point(3); 
					float pixel[2] = {(float) col, (float) row};
					rs2_deproject_pixel_to_point(cv_point.data(), &intrinsics_depth, pixel, depth);

					//xyz points in 3D space, process and combine with lidar data
					float cv_coordx=cv_point[0];
					float cv_coordy=cv_point[1];
					float cv_coordz=cv_point[2];
					//track points to be used for plane fitting 
					float3 temp;
					temp.x=cv_coordx,temp.y=cv_coordy,temp.z=cv_coordz;
					//+y=down, -y=up

					//prefilter points 
					if(temp.y<cv_groundplane_max_height) { continue; } //too high
					else
					{
						//std::cout << "Adding Point to be part of ground plane fitting" << std::endl;
						plane_points.push_back(temp);
					}
				}
			}

			plane ground_plane=fit_groundplane(plane_points);
			return ground_plane;


		}			

		float distance_from_plane(plane groundplane, float3 point)
		{
			// project vector pointing from plane to point onto the planes normal vector

			//proj a B= proj of B onto a = (A dot B /(|A|^2)+*a
			//magnitude = |a dot b|/|a|. including sign on dot product will include direction relative to normal

			//1. Determine a point on the plane

			float3 plane_point={ 0 , 0 , 1.0/groundplane.C*-groundplane.D };

			//2. Compute a vector pointing from on the plane to the point in space

			float3 vector= {point.x-plane_point.x, point.y-plane_point.y, point.z-plane_point.z};

			//3. Project onto the normal vector of the plane, tracking magnitude and sign.

			float dot=groundplane.A*vector.x+groundplane.B*vector.y+groundplane.C*vector.z;
			float magnitude=std::pow(groundplane.A*groundplane.A+groundplane.B*groundplane.B+
									 groundplane.C*groundplane.C,0.5);
			
			float result= dot/magnitude;

			//std::cout <<"Result= " <<result<<std::endl;
			return result;
		}

		void augment_camera(std::vector<float> & lidar_ranges)
		{
			cv::Mat cv_image=(cv_bridge::toCvCopy(cv_image_data,cv_image_data.encoding))->image; //Encoding type is 16UC1 (depth in mm)

			plane ground= compute_groundplane(cv_image);


			

			//use to debug: Returning 1
			//bool assert=( (cv_rows==cv_image.rows) && (cv_cols==cv_image.cols) );

			//std::cout << "Augment Camera Assert = " << assert <<std::endl; 


			//1. Obtain pixel and depth
			
			for(int i=0 ; i < (int)cv_sample_cols_raw.size() ; i++)
			{
				int col= cv_sample_cols_raw[i];
				

				for(int j=0; j < (int)cv_sample_rows_raw.size() ; j++)
				{
					int row=cv_sample_rows_raw[j];

					
					
					float depth= (cv_image.ptr<uint16_t>(row)[col])/(float)1000;

					
					if(depth > max_cv_range or depth < min_cv_range)
					{
						continue;
					}
					//2 convert pixel to xyz coordinate in space using camera intrinsics, pixel coords, and depth info
					std::vector<float> cv_point(3); 
					float pixel[2] = {(float) col, (float) row};
					rs2_deproject_pixel_to_point(cv_point.data(), &intrinsics_depth, pixel, depth);

					//xyz points in 3D space, process and combine with lidar data
					float cv_coordx=cv_point[0];
					float cv_coordy=cv_point[1];
					float cv_coordz=cv_point[2];

					float3 point={cv_coordx,cv_coordy,cv_coordz};

					//ground point check

					//1. Compute distance from ground plane
					float distance= distance_from_plane(ground,point);
					//2. ignore if ground

					//distance is postive if along the same direction as plane normal (down), negative if oppsote plane normal(up)
					
					if ( distance> -cv_groundplane_max_distance || distance < -camera_max) //max distance from plane to which a point is considered ground
					{
						//postive=below plane, 
						continue; //ignore ground point
					}




					//imu_pitch=0;
					//imu_roll=0;
					


					/*
					float cv_coordy_s = -1*cv_coordx*std::sin(imu_pitch) + cv_coordy*std::cos(imu_pitch)*std::cos(imu_roll) 
					+ cv_coordz *std::cos(imu_pitch)*std::sin(imu_roll);

					
					if( cv_coordy_s > camera_min || cv_coordy_s < -camera_max)
					{
						continue;
					}
					*/


					//3. Overwrite Lidar Points with Camera Points taking into account dif frames of ref

					float lidar_coordx = (cv_coordz+cv_distance_to_lidar);
                	float lidar_coordy = -cv_coordx;
					float cv_range_temp = std::pow(std::pow(lidar_coordx,2) + std::pow(lidar_coordy,2),0.5);
					//(coordx^2+coordy^2)^0.5

					int beam_index= std::floor(scan_beams*std::atan2(lidar_coordy, lidar_coordx)/(2*M_PI));
					float lidar_range = lidar_ranges[beam_index];
					lidar_ranges[beam_index] = std::min(lidar_range, cv_range_temp);
				}
			}

			ros::Time current_time= ros::Time::now();
			cv_ranges_msg.header.stamp=current_time;
			cv_ranges_msg.ranges=lidar_ranges;

			cv_ranges_pub.publish(cv_ranges_msg);
			

		}





		std::pair <std::vector<std::vector<float>>, std::vector<float>>preprocess_lidar(std::vector<float> ranges){

			std::vector<std::vector<float>> data(ls_len_mod,std::vector<float>(2));
			std::vector<float> data2(100);

			//sets distance to zero for obstacles in safe distance, and max_lidar_range for those that are far.
			for(int i =0; i < ls_len_mod; ++i){
				if(ranges[ls_str+i] <= safe_distance_adapt) {data[i][0] = 0; data[i][1] = i*ls_ang_inc-angle_cen;}
				else if(ranges[ls_str+i] <= max_lidar_range) {data[i][0] = ranges[ls_str+i]; data[i][1] = i*ls_ang_inc-angle_cen;}
				else {data[i][0] = max_lidar_range; data[i][1] = i*ls_ang_inc-angle_cen;}
			}

			int k1 = 100; int k2 = 40;
			float s_range = 0; int index1, index2;
			
			//moving window
			for(int i =0; i < k1; ++i){
				s_range = 0;

				for(int j =0; j < k2; ++j){
					index1 = int(i*ranges.size()/k1+j);
					if(index1 >= int(ranges.size())) index1 -= ranges.size();
					
					index2 = int(i*ranges.size()/k1-j);
					if(index2 < 0) index2 += ranges.size();

					s_range += std::min(ranges[index1], (float) max_lidar_range) + std::min(ranges[index2], (float)max_lidar_range);

				}
				data2[i] = s_range;
			}

			return std::make_pair(data,data2);
			
		}

		std::vector<float> preprocess_lidar_MPC(std::vector<float> ranges, std::vector<double> lidar_angles){
			left_ind_MPC = 0; right_ind_MPC = 0;
			//sets distance to zero for obstacles in safe distance, and max_lidar_range for those that are far.
			int num_det=0;
			double safe_dist=safe_distance_adapt;
			std::vector<float> ranges1=ranges;
			while(num_det==0){
				num_det=0;
				left_ind_MPC = 0; right_ind_MPC = 0;
				ranges=ranges1;
				if(safe_dist<0.1){
					num_det=1;
					for(int i =0; i < ranges.size(); ++i){
						if(lidar_angles[i] <= right_beam_angle_MPC) right_ind_MPC +=1;
						if(lidar_angles[i] <= left_beam_angle_MPC) left_ind_MPC +=1;
					}
					left_ind_MPC +=1;
				}
				else{
					for(int i =0; i < ranges.size(); ++i){
						if(lidar_angles[i] <= right_beam_angle_MPC) right_ind_MPC +=1;
						if(lidar_angles[i] <= left_beam_angle_MPC) left_ind_MPC +=1;
						if(right_ind_MPC!=i+1 && left_ind_MPC==i+1){
							if(ranges[i] <= safe_dist) {ranges[i] = 0;}
							else if(ranges[i] > max_lidar_range) {ranges[i] = max_lidar_range; num_det++;}
							else{num_det++;}
						}
						
					}
				}
				safe_dist=safe_dist/2;
			}
			left_ind_MPC -=1;
			return ranges;
			
		}


		
		

		void visualize_detections(){
			//This should occur whether or not we are in autonomous mode

			//Publish the detected vehicle(s) trajectory(s)
			vehicle_detect_path.header.frame_id = base_frame;
			vehicle_detect_path.header.stamp = ros::Time::now();
			vehicle_detect_path.type = visualization_msgs::Marker::LINE_LIST;
			vehicle_detect_path.id = 0; 
			vehicle_detect_path.action = visualization_msgs::Marker::ADD;
			vehicle_detect_path.scale.x = 0.1;
			vehicle_detect_path.color.a = 1.0;
			vehicle_detect_path.color.r = 0.2; 
			vehicle_detect_path.color.g = 0.2;
			vehicle_detect_path.color.b = 0.9;
			vehicle_detect_path.pose.orientation.w = 1;
			
			vehicle_detect_path.lifetime = ros::Duration(0.1);
			geometry_msgs::Point p7;
			vehicle_detect_path.points.clear();

			for(int i=0; i<car_detects.size();i++){ //For each detection, plot trajectory over next 3 seconds	
				if(car_detects[i].init<2) continue;
				double x_det=car_detects[i].state[0]; double y_det=car_detects[i].state[1]; double theta_det=car_detects[i].state[2];
				double vel_det = car_detects[i].state[3]; double steer_det = car_detects[i].state[4];
				//double vel_det = odomvel; double steer_det = odomsteer;
				
				for (int traj=0;traj<40;traj++){
					p7.x = x_det;	p7.y = y_det;	p7.z = 0;
					vehicle_detect_path.points.push_back(p7);
					x_det=x_det+vel_det*cos(theta_det)/10; //0.1 second increments (coarse but just for visualization)
					y_det=y_det+vel_det*sin(theta_det)/10; //0.1 second increments
					theta_det=theta_det+vel_det/wheelbase*tan(steer_det)/10; //0.1 second increments
					p7.x = x_det;	p7.y = y_det;	p7.z = 0;
					vehicle_detect_path.points.push_back(p7);
				}
			}

			vehicle_detect.publish(vehicle_detect_path);
		}

		void visualize_vehicle_lidar(double det_angle){
			//Publish the detected vehicle(s) trajectory(s)
			vehicle_lidar_dir.header.frame_id = base_frame;
			vehicle_lidar_dir.header.stamp = ros::Time::now();
			vehicle_lidar_dir.type = visualization_msgs::Marker::LINE_LIST;
			vehicle_lidar_dir.id = 0; 
			vehicle_lidar_dir.action = visualization_msgs::Marker::ADD;
			vehicle_lidar_dir.scale.x = 0.1;
			vehicle_lidar_dir.color.a = 1.0;
			vehicle_lidar_dir.color.r = 0.9; 
			vehicle_lidar_dir.color.g = 0.2;
			vehicle_lidar_dir.color.b = 0.2;
			vehicle_lidar_dir.pose.orientation.w = 1;
			
			vehicle_lidar_dir.lifetime = ros::Duration(0.1);
			geometry_msgs::Point p7;
			vehicle_lidar_dir.points.clear();

			for(int i=0; i<car_detects.size();i++){ //For each detection, plot trajectory over next 3 seconds	
				if(car_detects[i].init<2) continue;
				// Center around ego vehicle 
				double x_det=0; double y_det=0;
				double dist_to_det = sqrt(pow(car_detects[i].state[0], 2) + pow(car_detects[i].state[1], 2));

				for (int traj=0;traj<40;traj++){
					p7.x = x_det;	p7.y = y_det;	p7.z = 0;
					vehicle_lidar_dir.points.push_back(p7);
					x_det=x_det+cos(det_angle)/10; //0.1 second increments (coarse but just for visualization)
					y_det=y_det+sin(det_angle)/10; //0.1 second increments
					// theta_det=theta_det+car_detects[i].state[3]/wheelbase*tan(car_detects[i].state[4])/10; //0.1 second increments
					p7.x = x_det;	p7.y = y_det;	p7.z = 0;
					vehicle_lidar_dir.points.push_back(p7);
				}
			}

			vehicle_lidar.publish(vehicle_lidar_dir);
		}

		void EKF_general(int q, double dt){
			// Run the extended kalman filter algorithm for unknown inputs of external vehicle (included in states)
			// Expected output: Modified state estimate and covariance estimate
			
			// initial covariance and residual covariance
			Eigen::MatrixXd cov_P = car_detects[q].cov_P;	
			Eigen::MatrixXd H_jac = Eigen::MatrixXd::Identity(2, 5); 
			Eigen::MatrixXd init_resid_cov = car_detects[q].meas_noise + H_jac * cov_P * H_jac.transpose();

			
			// State Prediction
			Eigen::VectorXd pred_state = Eigen::VectorXd::Zero(5);	
			Eigen::VectorXd x = car_detects[q].state;
			double adj_dt = std::max(default_dt, dt);
			pred_state(0) = x[0] + x[3]*cos(x[2])*adj_dt;
			pred_state(1) = x[1] + x[3]*sin(x[2])*adj_dt;
			pred_state(2) = x[2] + x[3]*tan(x[4])*adj_dt*(1/wheelbase);
			pred_state(3) = x[3];
			pred_state(4) = x[4];


			// Measurement Residual
			Eigen::VectorXd meas_resid = Eigen::VectorXd::Zero(2);
			meas_resid(0) = car_detects[q].meas[0] - pred_state(0);	
			meas_resid(1) = car_detects[q].meas[1] - pred_state(1);


			// State Prediction Covariance
			car_detects[q].proc_noise(0,0)=0.05; car_detects[q].proc_noise(1,1)=0.05; car_detects[q].proc_noise(2,2)=std::pow(7.5 * M_PI / 180, 2);
			car_detects[q].proc_noise(3,3)=0.125; car_detects[q].proc_noise(4,4)=std::pow(7.5 * M_PI / 180, 2);

			Eigen::MatrixXd F_jac = Eigen::MatrixXd::Identity(5,5);		// State Jacobian
			F_jac(0,2) = -adj_dt*x[3]*sin(x[2]);
			F_jac(1,2) =  adj_dt*x[3]*cos(x[2]);
			F_jac(0,3) =  adj_dt*cos(x[2]);
			F_jac(1,3) =  adj_dt*sin(x[2]);
			F_jac(2,4) =  adj_dt*x[3]*(1/wheelbase)*(1/(pow(cos(x[4]),2)));
			F_jac(2,3) =  adj_dt*tan(x[4])*(1/wheelbase);
			Eigen::MatrixXd state_pred_cov = Eigen::MatrixXd::Zero(5,5);	// State Prediction Covariance
			state_pred_cov = F_jac * car_detects[q].cov_P * (F_jac.transpose()) + car_detects[q].proc_noise;


			// Residual Covariance
			Eigen::MatrixXd resid_cov = Eigen::MatrixXd::Zero(2,2);
			//Eigen::MatrixXd H_jac = Eigen::MatrixXd::Identity(2, 5);
			resid_cov = car_detects[q].meas_noise + H_jac * state_pred_cov * (H_jac.transpose());

			// Filter Gain
			Eigen::MatrixXd kalman_gain = Eigen::MatrixXd::Zero(5,2);
			kalman_gain = state_pred_cov * (H_jac.transpose()) * resid_cov.inverse();

			// Updated state estimate and covariance
			car_detects[q].state = pred_state + kalman_gain * meas_resid;
			// Using Joseph form covariance update
			Eigen::MatrixXd I = Eigen::MatrixXd::Identity(5,5);
			car_detects[q].cov_P = (I - kalman_gain*H_jac)*state_pred_cov*((I - kalman_gain*H_jac).transpose()) + 
									kalman_gain*(car_detects[q].meas_noise)*(kalman_gain.transpose());	

			// Find NEES and NIS
			// First, need actual state to compare to predicted state
			Eigen::VectorXd real_state = Eigen::VectorXd::Zero(5);
			real_state(0) = car_detects[q].meas[0];
			real_state(1) = car_detects[q].meas[1];
			real_state(2) = robtheta;
			real_state(3) = odomvel;
			real_state(4) = odomsteer;

			// Find difference between real and predicted
			Eigen::VectorXd state_err = real_state - pred_state;
			
			// Find the NEES of this filter
			double NEES = (state_err.transpose()) * (cov_P.inverse()) * state_err;
			std::cout << "NEES: " << NEES << std::endl;
			if(NEES < 11.1 && NEES > 0){
				std::cout << "Passed!" << std::endl;
			} else{
				std::cout << "Failed..." << std::endl;
			}

			// Find the NIS of this filter
			double NIS = (meas_resid.transpose()) * (init_resid_cov.inverse()) * meas_resid;
			std::cout << "NIS: " << NIS << std::endl;
			if(NIS < 5.99 && NIS > 0){
				std::cout << "Passed!" << std::endl;
			} else{
				std::cout << "Failed..." << std::endl;
			}

			if(sim_graph_time < 50) { 
				// At the end of each time sample, collect simulation data for graphing
				// Position, Heading Estimate Vs. True
				FILE *file_states = fopen("/home/gjsk/catkin_ws/Sim_Data/states.txt", "a");
				fprintf(file_states,"%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", sim_graph_time, 
						car_detects[q].state[0], real_state(0), car_detects[q].state[1], real_state(1), car_detects[q].state[2], real_state(2), 
						car_detects[q].state[3], real_state(3), car_detects[q].state[4], real_state(4));
				fclose(file_states);
				// Velocity Estimate Vs. True

				// Steering Angle Estimate Vs. True

				// Predicted Vs. Updated Variance for each parameter
				FILE *file_var = fopen("/home/gjsk/catkin_ws/Sim_Data/variance.txt", "a");
				fprintf(file_states,"%lf, %lf, %lf, %lf, %lf, %lf\n", sim_graph_time, 
						car_detects[q].cov_P(0,0), car_detects[q].cov_P(1,1), car_detects[q].cov_P(2,2), car_detects[q].cov_P(3,3), car_detects[q].cov_P(4,4));
				fclose(file_var);
				// NEES 
				FILE *file_nees = fopen("/home/gjsk/catkin_ws/Sim_Data/NEES_NIS.txt", "a");
				fprintf(file_nees,"%lf, %lf, %lf\n",sim_graph_time,NEES,NIS);
				fclose(file_nees);
				sim_graph_time += dt;
			}

		}

		void EKF_known_input(int q, double dt){
			// Run the extended kalman filter algorithm for unknown inputs of external vehicle (included in states)
			// Expected output: Modified state estimate and covariance estimate
			
			// initial covariance and residual covariance
			Eigen::MatrixXd cov_P = Eigen::MatrixXd::Zero(3, 3);  
			cov_P(0,0) = car_detects[q].cov_P(0,0);	
			cov_P(0,1) = car_detects[q].cov_P(0,1);	
			cov_P(0,2) = car_detects[q].cov_P(0,2);	
			cov_P(1,0) = car_detects[q].cov_P(1,0);	
			cov_P(1,1) = car_detects[q].cov_P(1,1);	
			cov_P(1,2) = car_detects[q].cov_P(1,2);	
			cov_P(2,0) = car_detects[q].cov_P(2,0);	
			cov_P(2,1) = car_detects[q].cov_P(2,1);	
			cov_P(2,2) = car_detects[q].cov_P(2,2);	
			Eigen::MatrixXd H_jac = Eigen::MatrixXd::Identity(2, 3); 
			Eigen::MatrixXd init_resid_cov = car_detects[q].meas_noise + H_jac * cov_P * H_jac.transpose();

			// State Prediction
			Eigen::VectorXd pred_state = Eigen::VectorXd::Zero(3);	
			Eigen::VectorXd x = Eigen::VectorXd::Zero(3);
			x(0) = car_detects[q].state[0];
			x(1) = car_detects[q].state[1];
			x(2) = car_detects[q].state[2];
			Eigen::VectorXd in = Eigen::VectorXd::Zero(3);
			in(0) = odomvel;
			in(1) = odomsteer;
			double adj_dt = std::max(default_dt, dt);
			pred_state(0) = x[0] + in(0)*cos(x[2])*adj_dt;
			pred_state(1) = x[1] + in(0)*sin(x[2])*adj_dt;
			pred_state(2) = x[2] + in(0)*tan(in(1))*adj_dt*(1/wheelbase);

			// Measurement Residual
			Eigen::VectorXd meas_resid = Eigen::VectorXd::Zero(2);
			meas_resid(0) = car_detects[q].meas[0] - pred_state(0);	
			meas_resid(1) = car_detects[q].meas[1] - pred_state(1);

			// State Prediction Covariance
			car_detects[q].proc_noise(0,0)=0.05; car_detects[q].proc_noise(1,1)=0.05; car_detects[q].proc_noise(2,2)=std::pow(7.5 * M_PI / 180, 2);
			Eigen::MatrixXd proc_noise = Eigen::MatrixXd::Zero(3, 3);	
			proc_noise(0,0) = car_detects[q].proc_noise(0,0);			
			proc_noise(1,1) = car_detects[q].proc_noise(1,1);	
			proc_noise(2,2) = car_detects[q].proc_noise(2,2);	

			//Measurement noise should depend on the distance between the vehicles (maybe error of 2% of distance, increases when speeds increase)
			car_detects[q].meas_noise(0,0)=0.02*sqrt(std::pow(car_detects[q].meas[0],2)+std::pow(car_detects[q].meas[1],2));
			car_detects[q].meas_noise(1,1)=car_detects[q].meas_noise(0,0);
			
			//Also incorporate speed's effect on error
			//car_detects[q].meas_noise(0,0)*=std::max(vel_adapt*odomvel/0.5,1.0); car_detects[q].meas_noise(1,1)*=std::max(vel_adapt*odomvel/0.5,1.0);

			Eigen::MatrixXd F_jac = Eigen::MatrixXd::Identity(3,3);		// State Jacobian
			F_jac(0,2) = -adj_dt*in(0)*sin(x[2]);
			F_jac(1,2) =  adj_dt*in(0)*cos(x[2]);
			Eigen::MatrixXd state_pred_cov = Eigen::MatrixXd::Zero(3,3);	// State Prediction Covariance
			state_pred_cov = F_jac * cov_P * (F_jac.transpose()) + proc_noise;

			// Residual Covariance
			Eigen::MatrixXd resid_cov = Eigen::MatrixXd::Zero(2,2);
			//Eigen::MatrixXd H_jac = Eigen::MatrixXd::Identity(2, 5);
			resid_cov = car_detects[q].meas_noise + H_jac * state_pred_cov * (H_jac.transpose());

			// Filter Gain
			Eigen::MatrixXd kalman_gain = Eigen::MatrixXd::Zero(3,2);
			kalman_gain = state_pred_cov * (H_jac.transpose()) * resid_cov.inverse();

			// Updated state estimate and covariance
			Eigen::VectorXd updated_state = pred_state + kalman_gain * meas_resid;
			car_detects[q].state(0) = updated_state(0);
			car_detects[q].state(1) = updated_state(1);
			car_detects[q].state(2) = updated_state(2);

			// Using Joseph form covariance update
			Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3,3);
			Eigen::MatrixXd updated_cov_P = Eigen::MatrixXd::Zero(3,3);
			updated_cov_P = (I - kalman_gain*H_jac)*state_pred_cov*((I - kalman_gain*H_jac).transpose()) + 
									kalman_gain*(car_detects[q].meas_noise)*(kalman_gain.transpose());	
			car_detects[q].cov_P(0,0) = updated_cov_P(0,0);	
			car_detects[q].cov_P(0,1) = updated_cov_P(0,1);	
			car_detects[q].cov_P(0,2) = updated_cov_P(0,2);	
			car_detects[q].cov_P(1,0) = updated_cov_P(1,0);	
			car_detects[q].cov_P(1,1) = updated_cov_P(1,1);	
			car_detects[q].cov_P(1,2) = updated_cov_P(1,2);	
			car_detects[q].cov_P(2,0) = updated_cov_P(2,0);	
			car_detects[q].cov_P(2,1) = updated_cov_P(2,1);	
			car_detects[q].cov_P(2,2) = updated_cov_P(2,2);	

			// Find NEES and NIS
			// First, need actual state to compare to predicted state
			Eigen::VectorXd real_state = Eigen::VectorXd::Zero(3);
			real_state(0) = car_detects[q].meas[0];
			real_state(1) = car_detects[q].meas[1];
			real_state(2) = robtheta;

			// Find difference between real and predicted
			Eigen::VectorXd state_err = real_state - pred_state;
			
			// Find the NEES of this filter
			double NEES = (state_err.transpose()) * (cov_P.inverse()) * state_err;
			std::cout << "NEES: " << NEES << std::endl;
			if(NEES < 11.1 && NEES > 0){
				std::cout << "Passed!" << std::endl;
			} else{
				std::cout << "Failed..." << std::endl;
			}

			// Find the NIS of this filter
			double NIS = (meas_resid.transpose()) * (init_resid_cov.inverse()) * meas_resid;
			std::cout << "NIS: " << NIS << std::endl;
			if(NIS < 5.99 && NIS > 0){
				std::cout << "Passed!" << std::endl;
			} else{
				std::cout << "Failed..." << std::endl;
			}

			if(sim_graph_time < 50) { 
				// At the end of each time sample, collect simulation data for graphing
				// Position, Heading Estimate Vs. True
				FILE *file_states = fopen("/home/gjsk/catkin_ws/Sim_Data/states.txt", "a");
				fprintf(file_states,"%lf, %lf, %lf, %lf, %lf, %lf, %lf\n", sim_graph_time, 
						car_detects[q].state[0], real_state(0), car_detects[q].state[1], real_state(1), car_detects[q].state[2], real_state(2));
				fclose(file_states);
				// Velocity Estimate Vs. True

				// Steering Angle Estimate Vs. True

				// Predicted Vs. Updated Variance for each parameter
				FILE *file_var = fopen("/home/gjsk/catkin_ws/Sim_Data/variance.txt", "a");
				fprintf(file_states,"%lf, %lf, %lf, %lf\n", sim_graph_time, 
						car_detects[q].cov_P(0,0), car_detects[q].cov_P(1,1), car_detects[q].cov_P(2,2));
				fclose(file_var);
				// NEES 
				FILE *file_nees = fopen("/home/gjsk/catkin_ws/Sim_Data/NEES_NIS.txt", "a");
				fprintf(file_nees,"%lf, %lf, %lf\n",sim_graph_time,NEES,NIS);
				fclose(file_nees);
				sim_graph_time += dt;
			}

		}


		void lidar_callback(const sensor_msgs::LaserScanConstPtr &data){

			ls_ang_inc = static_cast<double>(data->angle_increment); 
			scan_beams = int(2*M_PI/data->angle_increment);
			ls_str = int(round(scan_beams*right_beam_angle/(2*M_PI)));
			ls_end = int(round(scan_beams*left_beam_angle/(2*M_PI)));

			ros::Time ttt = ros::Time::now();
			current_time = ttt.toSec();
			double dt = current_time - prev_time;
			if(dt>1){
				dt=default_dt;
			}
			prev_time = current_time;


			//Vehicle tracking even if we aren't currently driving
			
			for (int q=car_detects.size()-1; q>=0;q--){ //Iterate backwards to handle deletions
				if(car_detects[q].last_det==0){ //No detection over last cycle
					car_detects[q].miss_fr++;
					if(car_detects[q].init==1) car_detects[q].init=0; //Two point initialization requires consecutive detections
					if(car_detects[q].miss_fr>5) car_detects.erase(car_detects.begin()+q); //Not detected over past several frames, delete detection struct
				}
				else car_detects[q].miss_fr=0; //Found, reset consecutive missed frames to 0
				
			}

			for(int q=0; q<car_detects.size();q++){ //Run the KF for every current detections here
				
				if(car_detects[q].init==0){ //Initialize
					if(car_detects[q].last_det==1){ //Only if measurement received
						car_detects[q].state[0]=car_detects[q].meas[0];
						car_detects[q].state[1]=car_detects[q].meas[1];
						car_detects[q].init=1;
					}
					car_detects[q].last_det=0;
					continue;
				}
				if(car_detects[q].init==1){ //Finish initializing
					//First transform last x and y to this new frame (also the measurements)
					double tempx=0; double tempy=0; double curmeasx=0; double curmeasy=0;

					if(simx==0){
						tempx=cos(odomtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-odomx)+
							sin(odomtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-odomy);
						tempy=-sin(odomtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-odomx)+
							cos(odomtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-odomy);

						car_detects[q].state[0]=tempx; car_detects[q].state[1]=tempy;

						curmeasx=cos(odomtheta)*(car_detects[q].meas_tf.tf_x+car_detects[q].meas[0]*cos(car_detects[q].meas_tf.tf_theta)-car_detects[q].meas[1]*sin(car_detects[q].meas_tf.tf_theta)-odomx)+
							sin(odomtheta)*(car_detects[q].meas_tf.tf_y+car_detects[q].meas[0]*sin(car_detects[q].meas_tf.tf_theta)+car_detects[q].meas[1]*cos(car_detects[q].meas_tf.tf_theta)-odomy);
						curmeasy=-sin(odomtheta)*(car_detects[q].meas_tf.tf_x+car_detects[q].meas[0]*cos(car_detects[q].meas_tf.tf_theta)-car_detects[q].meas[1]*sin(car_detects[q].meas_tf.tf_theta)-odomx)+
							cos(odomtheta)*(car_detects[q].meas_tf.tf_y+car_detects[q].meas[0]*sin(car_detects[q].meas_tf.tf_theta)+car_detects[q].meas[1]*cos(car_detects[q].meas_tf.tf_theta)-odomy);
					}
					else{
						tempx=cos(simtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-simx)+
							sin(simtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-simy);
						tempy=-sin(simtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-simx)+
							cos(simtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-simy);

						car_detects[q].state[0]=tempx; car_detects[q].state[1]=tempy;
						curmeasx=car_detects[q].meas[0];
						curmeasy=car_detects[q].meas[1];
					}


					car_detects[q].state[4]=0; //steering angle, depends on process noise
					car_detects[q].state[3]=std::min(sqrt(pow(curmeasx-car_detects[q].state[0],2)+pow(curmeasy-car_detects[q].state[1],2))/dt,0.3); //Cap initial at 3 m/s so we don't get extreme values upon initialization
					car_detects[q].state[2]=atan2(curmeasy-car_detects[q].state[1],curmeasx-car_detects[q].state[0]);
					car_detects[q].state[1]=curmeasy;
					car_detects[q].state[0]=curmeasx;

					car_detects[q].init=2; //Done initializing
					car_detects[q].last_det=0;
					continue;
				}

				//Now for the KF update process in the 'already initialized scenario'

				//First transform last x, y & theta to this new frame (also the measurements)
				double tempx=0; double tempy=0; double curmeasx=0; double curmeasy=0;
				
				if(simx==0){
					tempx=cos(odomtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-odomx)+
						sin(odomtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-odomy);
					tempy=-sin(odomtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-odomx)+
						cos(odomtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-odomy);

					car_detects[q].state[0]=tempx; car_detects[q].state[1]=tempy; car_detects[q].state[2]=car_detects[q].state[2]-(odomtheta-lasttheta);
					while(car_detects[q].state[2]>M_PI) car_detects[q].state[2]-=2*M_PI;
					while(car_detects[q].state[2]<-M_PI) car_detects[q].state[2]+=2*M_PI;

					curmeasx=cos(odomtheta)*(car_detects[q].meas_tf.tf_x+car_detects[q].meas[0]*cos(car_detects[q].meas_tf.tf_theta)-car_detects[q].meas[1]*sin(car_detects[q].meas_tf.tf_theta)-odomx)+
						sin(odomtheta)*(car_detects[q].meas_tf.tf_y+car_detects[q].meas[0]*sin(car_detects[q].meas_tf.tf_theta)+car_detects[q].meas[1]*cos(car_detects[q].meas_tf.tf_theta)-odomy);
					curmeasy=-sin(odomtheta)*(car_detects[q].meas_tf.tf_x+car_detects[q].meas[0]*cos(car_detects[q].meas_tf.tf_theta)-car_detects[q].meas[1]*sin(car_detects[q].meas_tf.tf_theta)-odomx)+
						cos(odomtheta)*(car_detects[q].meas_tf.tf_y+car_detects[q].meas[0]*sin(car_detects[q].meas_tf.tf_theta)+car_detects[q].meas[1]*cos(car_detects[q].meas_tf.tf_theta)-odomy);

					car_detects[q].meas[0]=curmeasx; car_detects[q].meas[1]=curmeasy;
				}
				else{
					tempx=cos(simtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-simx)+
						sin(simtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-simy);
					tempy=-sin(simtheta)*(lastx+car_detects[q].state[0]*cos(lasttheta)-car_detects[q].state[1]*sin(lasttheta)-simx)+
						cos(simtheta)*(lasty+car_detects[q].state[0]*sin(lasttheta)+car_detects[q].state[1]*cos(lasttheta)-simy);

					car_detects[q].state[0]=tempx; car_detects[q].state[1]=tempy; car_detects[q].state[2]=car_detects[q].state[2]-(simtheta-lasttheta);
					while(car_detects[q].state[2]>M_PI) car_detects[q].state[2]-=2*M_PI;
					while(car_detects[q].state[2]<-M_PI) car_detects[q].state[2]+=2*M_PI;
					curmeasx=car_detects[q].meas[0];
					curmeasy=car_detects[q].meas[1];
				}

				// Diagnostics io stream
				std::cout << "******************************************" << std::endl;
				std::cout << "Measured Position: x = " << car_detects[q].meas[0] << "\t y = " << car_detects[q].meas[1] << std::endl;
				std::cout << "State Position: x = " << car_detects[q].state[0] << "\t y = " << car_detects[q].state[1] << std::endl;
				std::cout << "Actual Velocity: " << odomvel << std::endl;
				if(EKF_mode == "general"){std::cout << "Estimated Velocity: " << car_detects[q].state[3] << std::endl;}
				std::cout << "Ext. Vehicle Heading Angle: \t" << robtheta << std::endl;
				std::cout << "Predicted Heading Angle: \t" << car_detects[q].state[2] << std::endl;
				std::cout << "Ext. Vehicle Velocity: \t" << odomvel << std::endl;
				std::cout << "Predicted Velocity: \t" << car_detects[q].state[3] << std::endl;
				std::cout << "Ext. Vehicle Steering Angle: " << odomsteer << std::endl;
				std::cout << "Predicted Steering Angle: " << car_detects[q].state[4] << std::endl;
				// double angle_to_ext_state = atan2(car_detects[q].state[1], car_detects[q].state[0]);
				// std::cout << "Angle based on state: " << angle_to_ext_state << std::endl;
				// visualize_vehicle_lidar(angle_to_ext_state);


				// At this point the usable variables are all the ones in car_detects[q]
				// Measurement, state, covariance, process noise, measurement noise
				// We also have t in the form of dt

				if (EKF_mode == "general"){
					EKF_general(q, dt);
				}
				else if(EKF_mode == "known_in"){
					EKF_known_input(q,dt);
				}

				car_detects[q].last_det=0; // Reset detection, 


			}
			
			lastx=odomx; lasty=odomy; lasttheta=odomtheta; //Keep our vehicle frame from last cycle to transform frame to new this cycle
			if(simx!=0){lastx=simx; lasty=simy; lasttheta=simtheta;}
			timestamp_tf2=timestamp_tf1; timestamp_cam2=timestamp_cam1;
			visualize_detections(); //PLot the detections in rviz regardless of if we are in autonomous mode or not


		}

};

int main(int argc, char **argv){
		ros::init(argc, argv, "detection");
		GapBarrier gb;

		while(ros::ok()){
			
			ros::spinOnce();
		}

}


