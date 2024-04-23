import datetime
import os
import pickle
import sys

import adapy
import cv2
import numpy as np
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image
import rospy
from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
import pdb

from aikido.utils import translate_jaco_aikido2pybullet, translate_jaco_pybullet2aikido, create_aikido_obj
from domains.manipulation.fabric.gpt_fabric_problems import gpt_fabric_problem
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
    pybullet_client_id = initialize_pybullet_client()

    gpt_fabric_prob = gpt_fabric_problem()

    rospy.init_node("gpt_fabric_demo")
    roscpp_init('gpt_fabric_demo', [])
    rate = rospy.Rate(10)
    IS_SIM = False

    if rospy.is_shutdown():
        exit(1)

    ada = adapy.Ada(IS_SIM)
    if not IS_SIM:
        if not ada.start_trajectory_controllers():
            print("Could not start trajectory controller.")
            sys.exit(1)
    rospy.sleep(1)

    ada.get_hand().execute_preshape([1., 1.])

    viewer = ada.start_viewer("dart_markers/gpt_fabric_demo", "map")
    world = ada.get_world()

    # aikido_init_joint_values = [-1.47713615, 2.92438603, 1.0026695, -2.08638991, 1.44293104, 1.32299172]
    aikido_init_joint_values = [-2.49310893, 2.78821967, 1.00316212, -2.08609003, 1.44363025, 1.32265736]
    set_arm_conf(gpt_fabric_prob.robots[0].pid, translate_jaco_aikido2pybullet(aikido_init_joint_values))

    for obj in gpt_fabric_prob.objects:
        obj_pose = get_pose(obj.pid)
        aikido_obj = create_aikido_obj(world, obj.shape_key, obj_pose)

    collision = ada.get_world_collision_constraint()
    traj = ada.plan_to_configuration_rrtconnect(aikido_init_joint_values, collision)
    ada.execute_trajectory(traj)

    # create the workspace table
    ws_table_top = 0.7165
    ws_table = create_table((0.8, 0.45, 0.01), 'table0', pose=((-0.5, -0.6, -0.8), (0, 0, 2 ** 0.5 / 2, 2 ** 0.5 / 2)),
                            region_pose=((-0.1, -0.5, - 0.8), (0, 0, 0, 1)))

    # set up camera model and the related pipeline
    # Initialize camera model
    camera_model = PinholeCameraModel()

    # Assuming you have a CameraInfo message with calibration data
    camera_info_msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
    camera_model.fromCameraInfo(camera_info_msg)

    RB_matrix = pickle.load(open('RB_matrix.pkl', 'rb'))

    # create a virtual bowl that will be used for deciding robot configurations
    virtual_bowl_pid = create_simple_bowl([0.01, 0.01, 0.01])
    virtual_bowl = Obj(f'bowl{virtual_bowl_pid}', virtual_bowl_pid, 'FabricBowl', [],
                       properties=['Container', 'Movable'], status=['Graspable', 'Placeable'],
                       shape=[0.01, 0.01, 0.01])

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

    if_manual = False
    step = -1
    while True:
        step += 1

        if if_manual:
            inst = wait_for_user("Please provide an instruction in this format: (pixel_x, pixel_y, place_x, place_y)!")

            if inst == 'e':
                break

            pixel_x, pixel_y, place_x, place_y = inst.split('_')

            pixel_x = int(pixel_x)
            pixel_y = int(pixel_y)
            place_x = int(place_x)
            place_y = int(place_y)
        else:
            depth_img = get_depth_img()
            pickle.dump(depth_img, open(os.path.join(exp_dir, f'depth{step}.png'), 'wb'))
            rgb_img = get_rgb_img_from_camera()
            cv2.imwrite(os.path.join(exp_dir, f'rgb{step}.png'), rgb_img)
            [pixel_x, pixel_y], [place_x, place_y] = gpt_for_real_world(
                os.path.join(exp_dir, f'rgb{step}.png'),
                os.path.join(exp_dir, f'cropped_gray{step}.png'),
                os.path.join(exp_dir, f'cropped_rgb{step}.png'),
                cloth_center,
                "CornersEdgesInward",
                step)

            wait_for_user(f"Suggested pick point: [{pixel_x, pixel_y}], place point: [{place_x, place_y}]")

        depth_img = get_depth_img()

        if depth_img is None:
            exit(1)

        pick_camera_point = pixel_2_camera_frame([pixel_x, pixel_y], depth_img, camera_model)
        if pick_camera_point[0] == 0:
            print("Pick camera point is 0! Try a different pixel!")
            continue

        pick_robot_point = convert_camera_points(pick_camera_point, RB_matrix)
        pick_robot_point += np.array([[0.0], [- 0.005], [0]])

        place_camera_point = pixel_2_camera_frame([place_x, place_y], depth_img, camera_model)
        if place_camera_point[0] == 0:
            print("Place camera point is 0! Try a different pixel!")
            continue

        place_robot_point = convert_camera_points(place_camera_point, RB_matrix)
        place_robot_point += np.array([[0.0], [- 0.005], [0]])

        set_pose(virtual_bowl.pid, ((pick_robot_point[0][0], pick_robot_point[1][0], -0.01), (0, 0, 0, 1)))

        grasp_lst = next(from_list_fn(get_obj_agent_grasp_gen())(virtual_bowl, gpt_fabric_prob.robots[0]))

        for grasp, in grasp_lst:
            cmd = gpt_fabric_prob.robots[0].compute_cachable_single_agent_ik_solutions(virtual_bowl.pid,
                                                                                       get_pose(virtual_bowl.pid),
                                                                                       grasp,
                                                                                       tuple([]), tuple([]),
                                                                                       teleport=True, if_place=False)
            pybullet_goal_conf = cmd.commands[0].path[-1].arm_joint_values
            aikido_goal_conf = translate_jaco_pybullet2aikido(pybullet_goal_conf)

            traj = ada.plan_to_configuration_rrtconnect(aikido_goal_conf, collision)
            wait_for_user("Press to start moving to pick point!")
            ada.execute_trajectory(traj)

            # pdb.set_trace()

            # move down a bit
            # offset = [0., 0., -0.035]
            wait_for_user("Press to start moving down!")
            for i in range(5):
                offset = [0., 0., -0.01]
                hand_node = rospy.get_param("adaConf/hand_base")
                traj_off = ada.plan_to_offset(hand_node, offset)
                if traj_off is None:
                    continue
                ada.execute_trajectory(traj_off)

            # pdb.set_trace()

            wait_for_user("Press to start closing gripper!")
            # close gripper
            ada.get_hand().close()

            # pdb.set_trace()

            wait_for_user("Press to start moving up!")
            for i in range(5):
                offset = [0., 0., 0.01]
                hand_node = rospy.get_param("adaConf/hand_base")
                traj_off = ada.plan_to_offset(hand_node, offset)
                ada.execute_trajectory(traj_off)

            wait_for_user("Press to start moving to place point!")
            offset = np.array([place_robot_point[0][0] - pick_robot_point[0][0],
                               place_robot_point[1][0] - pick_robot_point[1][0], 0])
            num_steps = np.floor(np.linalg.norm(offset) / 0.05)
            for i in range(int(num_steps)):
                sub_offset = offset / np.linalg.norm(offset) * 0.05
                hand_node = rospy.get_param("adaConf/hand_base")
                traj_off = ada.plan_to_offset(hand_node, sub_offset)
                if traj_off is None:
                    continue
                ada.execute_trajectory(traj_off)

            wait_for_user("Press to start moving down!")
            for i in range(5):
                offset = [0., 0., -0.01]
                hand_node = rospy.get_param("adaConf/hand_base")
                traj_off = ada.plan_to_offset(hand_node, offset)
                if traj_off is None:
                    continue
                ada.execute_trajectory(traj_off)

            wait_for_user("Press to start openning the gripper!")
            # open gripper
            ada.get_hand().execute_preshape([1., 1.])

            wait_for_user("Press to start moving up")
            for i in range(5):
                offset = [0., 0., 0.01]
                hand_node = rospy.get_param("adaConf/hand_base")
                traj_off = ada.plan_to_offset(hand_node, offset)
                if traj_off is None:
                    continue
                ada.execute_trajectory(traj_off)

            ada.get_hand().execute_preshape([1., 1.])

            wait_for_user("Press to start moving to the initial position!")
            # send robot back
            traj = ada.plan_to_configuration_rrtconnect(aikido_init_joint_values, collision)
            ada.execute_trajectory(traj)
            ada.get_hand().execute_preshape([1., 1.])
