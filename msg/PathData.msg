Header header
geometry_msgs/Point[] points
float32 time_horizon
int32 num_disc
bool fixed

# bez_pts: Positions of the bezier points relative to the vehicle
# time_horizon: Time horizon which the bezier curve is drawn through in seconds
# num_disc: Number of discretizations of the bezier curve
# fixed: Determines if the path can be changed further