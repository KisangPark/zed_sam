"""
pcl convert
    - use ros2 object of converting PointCloud2 into list of points
    - use python pcl library (optional)

sensor_msgs.point_cloud2
"""

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

def pointcloud2_to_pcl(msg):
    # receive PointCloud2 message, convert!

    points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    # need to check field name with pointfields

    pcl_points = [list(p) for p in points]

    return pcl_points
    # return list of points
    # to check specific point -> reshape?
    # print each output to check the output is correct