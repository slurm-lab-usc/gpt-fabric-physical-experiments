import pickle

import cv2
import numpy as np
import rospy
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image

from gpt_fabric_demo.camera_utils import get_depth_img, pixel_2_camera_frame
from gpt_fabric_demo.transform_utils import solve_rigid_transform, convert_camera_points

# (x, y)
pixel_points = np.array([[264, 236], [279, 236], [296, 237], [313, 237],
                         [329, 238], [346, 238], [363, 239], [380, 240],
                         [264, 219], [280, 219], [297, 221], [313, 220],
                         [330, 221], [348, 221], [364, 222], [380, 222]])
robot_points = np.array([[0.1, - 0.485, -0.025], [0.075, - 0.485, -0.025], [0.05, - 0.485, -0.025], [0.025, - 0.485, -0.025],
                         [0, - 0.485, -0.025], [-0.025, - 0.485, -0.025], [-0.05, - 0.485, -0.025], [-0.075, - 0.485, -0.025],
                         [0.1, - 0.510, -0.025], [0.075, - 0.510, -0.025], [0.05, - 0.510, -0.025], [0.025, - 0.510, -0.025],
                         [0, - 0.510, -0.025], [-0.025, - 0.510, -0.025], [-0.05, - 0.510, -0.025], [-0.075, - 0.510, -0.025]])

# test_points = np.array([[359, 399], [370, 394], [380, 390], [392, 385],
#                         [402, 380], [412, 375], [425, 371], [436, 367],
#                         [381, 382]])

if __name__ == '__main__':
    rospy.init_node('coodinate_transform')

    # Initialize camera model
    camera_model = PinholeCameraModel()

    # Assuming you have a CemeraInfo message with calibration data
    camera_info_msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
    camera_model.fromCameraInfo(camera_info_msg)

    depth_img = get_depth_img()

    # check depth image
    if_vis = False

    if if_vis:
        # Display the result
        cv2.imshow("Depth", depth_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if depth_img is None:
        exit(1)

    camera_points = []
    filtered_robot_points = []
    for point_id, pixel_point in enumerate(pixel_points):
        camera_point = pixel_2_camera_frame(pixel_point, depth_img, camera_model)
        if camera_point[2] == 0:
            continue
        filtered_robot_points.append(robot_points[point_id])
        camera_points.append(list(camera_point))
    camera_points = np.array(camera_points)
    filtered_robot_points = np.array(filtered_robot_points)

    RB_matrix = solve_rigid_transform(camera_points, filtered_robot_points)

    pickle.dump(RB_matrix, open('RB_matrix.pkl', 'wb'))

    # for test_point in test_points:
    #     test_camera_point = pixel_2_camera_frame(test_point, depth_img, camera_model)
    #     test_robot_point = convert_camera_points(test_camera_point, RB_matrix)
    #
    #     print(f"Robot point: {test_robot_point}")
