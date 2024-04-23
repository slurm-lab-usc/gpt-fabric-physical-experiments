import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image


def get_rgb_img_from_camera(vis=False):
    # Get an image from the ros camera node
    rgb_img = rospy.wait_for_message("/camera/color/image_raw", Image)

    bridge = CvBridge()

    try:
        cv2_img = bridge.imgmsg_to_cv2(rgb_img, "bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        if vis:
            cv2.imshow("Image window", cv2_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return cv2_img

    return None


def get_depth_img(vis=False):
    # Get an image from the ros camera node
    depth_img = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)

    bridge = CvBridge()

    try:
        cv2_depth_img = bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
    except CvBridgeError as e:
        print(e)
    else:
        return cv2_depth_img

    return None


def pixel_2_camera_frame(pixel_point, depth_img, camera_model):
    depth = depth_img[pixel_point[1], pixel_point[0]]

    # Convert pixel to 3D point in camera frame
    point_in_camera = camera_model.projectPixelTo3dRay((pixel_point[0], pixel_point[1]))

    # Scale the point by the depth to get the 3D coordinate in the camera frame
    x = point_in_camera[0] * depth
    y = point_in_camera[1] * depth
    z = depth

    print(f"pixel coordinate: {pixel_point}, 3d coordinate: {(x, y, z)}")

    return x / 1000, y / 1000, z / 1000
