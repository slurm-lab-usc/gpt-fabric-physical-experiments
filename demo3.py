import os
import pickle
import sys
import datetime
import time

import adapy
import cv2
import numpy as np
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image
import rospy
from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
import pdb

from aikido.utils import translate_jaco_aikido2pybullet, translate_jaco_pybullet2aikido
from domains.tamp.human.obj_creation_utils import create_table, create_simple_bowl
from exp.gtamp.utils.pybullet_utils import initialize_pybullet_client
from gpt_fabric_demo.camera_utils import get_depth_img, pixel_2_camera_frame, get_rgb_img_from_camera
from gpt_fabric_demo.gpt_fabric.slurm_utils import get_initial_cloth_center, gpt_for_real_world
from gpt_fabric_demo.transform_utils import solve_rigid_transform, convert_camera_points
from pddlstream.language.generator import from_list_fn
from wecook_tools.jaco_tools.jaco_problems import create_jaco
from wecook_tools.jaco_tools.jaco_utils import set_arm_conf
from wecook_tools.pose_transformation import set_pose, get_pose
from wecook_tools.robot_tools.objects import Obj
from wecook_tools.robot_tools.robot_primitives import get_obj_agent_grasp_gen
from wecook_tools.robot_tools.robots import JACO
from wecook_tools.user_io import wait_for_user

if __name__ == '__main__':
    rospy.init_node("gpt_fabric_demo")
    roscpp_init('gpt_fabric_demo', [])
    rate = rospy.Rate(10)
    # IS_SIM = False
    #
    # if rospy.is_shutdown():
    #     exit(1)
    #
    # ada = adapy.Ada(IS_SIM)
    # if not IS_SIM:
    #     if not ada.start_trajectory_controllers():
    #         print("Could not start trajectory controller.")
    #         sys.exit(1)
    # rospy.sleep(1)
    #
    # ada.get_hand().execute_preshape([1., 1.])
    #
    # viewer = ada.start_viewer("dart_markers/gpt_fabric_demo", "map")
    # world = ada.get_world()
    #
    # pybullet_client_id = initialize_pybullet_client()
    #
    # jaco_pid = create_jaco()
    # jaco = JACO('jaco0', jaco_pid)
    # # aikido_init_joint_values = [-1.47713615, 2.92438603, 1.0026695, -2.08638991, 1.44293104, 1.32299172]
    # aikido_init_joint_values = [-2.49310893, 2.78821967, 1.00316212, -2.08609003, 1.44363025, 1.32265736]
    # set_arm_conf(jaco.pid, translate_jaco_aikido2pybullet(aikido_init_joint_values))
    #
    # traj = ada.plan_to_configuration(aikido_init_joint_values)
    # ada.execute_trajectory(traj)
    #
    # # create the workspace table
    # ws_table_top = 0.7165
    # ws_table = create_table((0.8, 0.45, 0.01), 'table0', pose=((-0.5, -0.6, -0.8), (0, 0, 2 ** 0.5 / 2, 2 ** 0.5 / 2)),
    #                         region_pose=((-0.1, -0.5, - 0.8), (0, 0, 0, 1)))
    #
    # # set up camera model and the related pipeline
    # # Initialize camera model
    # camera_model = PinholeCameraModel()
    #
    # # Assuming you have a CameraInfo message with calibration data
    # camera_info_msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
    # camera_model.fromCameraInfo(camera_info_msg)
    #
    # RB_matrix = pickle.load(open('RB_matrix.pkl', 'rb'))
    #
    # # create a virtual bowl that will be used for deciding robot configurations
    # virtual_bowl_pid = create_simple_bowl([0.01, 0.01, 0.01])
    # virtual_bowl = Obj(f'bowl{virtual_bowl_pid}', virtual_bowl_pid, 'FabricBowl', [],
    #                    properties=['Container', 'Movable'], status=['Graspable', 'Placeable'],
    #                    shape=[0.01, 0.01, 0.01])

    exp_name = f'{datetime.datetime.now()}'
    os.makedirs(exp_name)
    exp_dir = os.path.join(os.path.dirname(__file__), exp_name)

    depth_img = get_depth_img()
    pickle.dump(depth_img, open(os.path.join(exp_dir, 'depth00.png'), 'wb'))
    rgb_img = get_rgb_img_from_camera()
    cv2.imwrite(os.path.join(exp_dir, 'rgb00.png'), rgb_img)
    cloth_center = get_initial_cloth_center(os.path.join(exp_dir, 'rgb00.png'),
                                            os.path.join(exp_dir, 'cropped_gray00.png'),
                                            os.path.join(exp_dir, 'cropped_rgb00.png'))

    step = -1
    while True:
        step += 1

        depth_img = get_depth_img()
        pickle.dump(depth_img, open(os.path.join(exp_dir, f'depth{step}.png'), 'wb'))
        rgb_img = get_rgb_img_from_camera()
        cv2.imwrite(os.path.join(exp_dir, f'rgb{step}.png'), rgb_img)
        [pixel_x, pixel_y], [place_x, place_y] = gpt_for_real_world(
            os.path.join(exp_dir, f'rgb{step}.png'),
            os.path.join(exp_dir, f'cropped_gray{step}.png'),
            os.path.join(exp_dir, f'cropped_rgb{step}.png'),
            cloth_center,
            "DoubleTriangle",
            step)

        wait_for_user(f"Suggested pick point: [{pixel_x, pixel_y}], place point: [{place_x, place_y}]")
