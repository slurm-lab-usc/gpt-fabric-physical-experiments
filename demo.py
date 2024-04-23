import sys

import adapy
import rospy
from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
import pdb

from aikido.utils import translate_jaco_aikido2pybullet, translate_jaco_pybullet2aikido
from domains.tamp.human.obj_creation_utils import create_table, create_simple_bowl
from exp.gtamp.utils.pybullet_utils import initialize_pybullet_client
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
    IS_SIM = False

    if rospy.is_shutdown():
        exit(1)

    ada = adapy.Ada(IS_SIM)
    if not IS_SIM:
        if not ada.start_trajectory_controllers():
            print("Could not start trajectory controller.")
            sys.exit(1)
    rospy.sleep(1)

    viewer = ada.start_viewer("dart_markers/gpt_fabric_demo", "map")
    world = ada.get_world()

    pybullet_client_id = initialize_pybullet_client()

    jaco_pid = create_jaco()
    jaco = JACO('jaco0', jaco_pid)
    aikido_init_joint_values = [-1.47713615, 2.92438603, 1.0026695, -2.08638991, 1.44293104, 1.32299172]
    set_arm_conf(jaco.pid, translate_jaco_aikido2pybullet(aikido_init_joint_values))

    # create the workspace table
    ws_table_top = 0.7165
    ws_table = create_table((0.8, 0.45, 0.01), 'table0', pose=((-0.5, -0.6, -0.8), (0, 0, 2 ** 0.5 / 2, 2 ** 0.5 / 2)),
                            region_pose=((-0.1, -0.5, - 0.8), (0, 0, 0, 1)))

    # create 8 bowls corresponding to 8 different test poses
    # test_robot_points = [[0.10098091, -0.29468136, -0.03732841], [0.07538013, -0.29369777, -0.04144392],
    #                      [0.05293798, -0.29339432, -0.04102764], [0.0280905, -0.2978246, -0.02850884],
    #                      [0.00433779, -0.29852489, -0.03103888], [-0.01582408, -0.30507754, -0.0174118],
    #                      [-0.04417534, -0.30266468, -0.01791775], [-0.06703825, -0.30395468, -0.0125108]]
    test_robot_points = [[0.04786985, -0.31453803, -0.02773036]]

    test_bowls = []
    for test_robot_point in test_robot_points:
        test_bowl_pid = create_simple_bowl([0.01, 0.01, 0.01], pose=test_robot_point)
        set_pose(test_bowl_pid, (test_robot_point, (0, 0, 0, 1)))
        test_bowls.append(Obj(f'bowl{test_bowl_pid}', test_bowl_pid, 'FabricBowl', [],
                              properties=['Container', 'Movable'], status=['Graspable', 'Placeable'],
                              shape=[0.01, 0.01, 0.01]))

    for test_bowl in test_bowls:
        grasp_lst = next(from_list_fn(get_obj_agent_grasp_gen())(test_bowl, jaco))

        for grasp, in grasp_lst:
            cmd = jaco.compute_cachable_single_agent_ik_solutions(test_bowl.pid, get_pose(test_bowl.pid), grasp,
                                                                  tuple([]), tuple([]), teleport=True, if_place=False)
            pybullet_goal_conf = cmd.commands[0].path[-1].arm_joint_values
            aikido_goal_conf = translate_jaco_pybullet2aikido(pybullet_goal_conf)

            # ada.get_hand().close()

            ada.get_hand().execute_preshape([0.7, 0.7])

            pdb.set_trace()

            traj = ada.plan_to_configuration(aikido_goal_conf)
            ada.execute_trajectory(traj)

            pdb.set_trace()

            # move down a bit
            offset = [0., 0., -0.03]
            hand_node = rospy.get_param("adaConf/hand_base")
            traj_off = ada.plan_to_offset(hand_node, offset)
            ada.execute_trajectory(traj_off)

            pdb.set_trace()

            # close gripper
            ada.get_hand().close()

            pdb.set_trace()

            offset = [0., 0., 0.06]
            hand_node = rospy.get_param("adaConf/hand_base")
            traj_off = ada.plan_to_offset(hand_node, offset)
            ada.execute_trajectory(traj_off)

    wait_for_user()
