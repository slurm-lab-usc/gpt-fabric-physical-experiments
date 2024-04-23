#!/usr/bin/env python

import adapy
import rospy
import sys

import pdb
from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
import numpy as np

rospy.init_node("aikido_read_robot_joints")
roscpp_init('aikido_read_robot_joints', [])
rate = rospy.Rate(10)
IS_SIM = False

if not rospy.is_shutdown():
    ada = adapy.Ada(IS_SIM)
    if not IS_SIM:
        if not ada.start_trajectory_controllers():
            print("Could not start trajectory controller.")
            sys.exit(1)
    rospy.sleep(1)  # wait for ada to initialize

    viewer = ada.start_viewer("dart_markers/aikido_read_robot_joints", "map")
    world = ada.get_world()

    pdb.set_trace()

    positions = ada.get_arm_positions()
    print(positions)

    # to_pick_points = np.array([[-0.10046597, 0.29910648, -0.02204735], [-0.07510559, 0.29724373, -0.02951988],
    #                            [-0.05332997, 0.29800922, -0.02540835], [-0.02619546, 0.29719315, -0.0277306],
    #                            [-0.00298407, 0.29847582, -0.0285361], [0.01666895, 0.30637518, -0.00790251],
    #                            [0.04528815, 0.30258923, -0.01294152], [0.06975783, 0.30112595, -0.0154916]])
    #
    # for to_pick_point in to_pick_points:

