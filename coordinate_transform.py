import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rospy
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image

from gpt_fabric_demo.camera_utils import get_rgb_img_from_camera, get_depth_img

if __name__ == '__main__':
    rospy.init_node('coordinate_transform')

    # Initialize camera model
    camera_model = PinholeCameraModel()

    # Assuming you have a CemeraInfo message with calibration data
    camera_info_msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
    camera_model.fromCameraInfo(camera_info_msg)

    rgb_img = get_rgb_img_from_camera()
    depth_img = get_depth_img()

    if rgb_img is None or depth_img is None:
        exit(1)

    # Convert to grayscale
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    # Apply Harris corner detection
    dst = cv2.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)

    if_vis = False

    if if_vis:
        # Result is dilated for marking the corners
        # Threshold for an optimal value, it may vary depending on the image
        rgb_img[dst > 0.01 * dst.max()] = [0, 0, 255]

        # Display the result
        cv2.imshow("Harris Corners", rgb_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # To get the pixel coordinates of detected corners
    corners = np.where(dst > 0.01 * dst.max())
    lst_coordinates = list(zip(*corners[::-1]))  # zip the y and x coordinates, reverse to
    # print("Detected corners' coordinates: ", lst_coordinates)

    for coordinates in lst_coordinates:
        if 300 < coordinates[0] < 340 and 260 < coordinates[1] < 300:
            print(coordinates)
            rgb_img[coordinates[1], coordinates[0]] = [0, 0, 255]

    # Display the result
    # cv2.imshow("Harris Corners", rgb_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Specify a coordinate to pick
    x = 317
    y = 283

    import pdb
    pdb.set_trace()

    depth = depth_img[y, x]

    # Convert pixel to 3D point in camera frame
    point_in_camera = camera_model.projectPixelTo3dRay((x, y))

    # Scale the point by the depth to get the 3D coordinate in the camera frame
    x = point_in_camera[0] * depth
    y = point_in_camera[1] * depth
    z = depth

    print(f"3D coordinate in the camera frame: {x}, {y}, {z}.")
