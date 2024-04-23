import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rospy
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image


def get_imgs_from_camera(vis=False):
    # Get an image from the ros camera node
    rgb_img = rospy.wait_for_message("/camera/color/image_raw", Image)
    depth_img = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)

    bridge = CvBridge()

    try:
        cv2_img = bridge.imgmsg_to_cv2(rgb_img, "bgr8")
        cv2_depth_img = bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
    except CvBridgeError as e:
        print(e)
    else:
        if vis:
            cv2.imshow("Image window", cv2_img)
            cv2.imshow("Depth image window", cv2_depth_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return cv2_img, cv2_depth_img

    return None, None


if __name__ == '__main__':
    rospy.init_node('find_test_coords')

    rgb_img, depth_img = get_imgs_from_camera()

    if rgb_img is None:
        exit(1)

    # Convert to grayscale
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    # Apply Harris corner detection
    dst = cv2.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)

    if_vis = True

    if if_vis:
        # Result is dilated for marking the corners
        # Threshold for an optimal value, it may vary depending on the image
        rgb_img[dst > 0.01 * dst.max()] = [0, 0, 255]

        # Display the result
        cv2.imshow("Harris Corners", rgb_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
