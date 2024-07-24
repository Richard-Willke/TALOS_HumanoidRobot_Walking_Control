#!/usr/bin/env python3
"""
A class that manage the gait with a finite state machine
"""

import numpy as np
import pinocchio as pin
import pybullet as pb

from foot_plan.footstep_planner import Side, FootStep, FootStepPlanner
from simulator.pybullet_wrapper import PybulletWrapper

class GaitFSM():
    """
    A class that manage the gait with a finite state machine

    This class automatically manage the (receding horizon) of the footstep planning queue
        - Could manage the footstep queue with correct display in Pybullet
        - Could output the next swinging foot target
          When outputting the next swinging foot target, 
           - it will automatically pop the current footstep
           - it will automatically plan the next footstep if conditions are met
           - it will output a series of footstep for MPC planning
        - Could interrupt the footstep planning queue
    """
    def __init__(self, conf, support_foot: FootStep, step_length: float, pb_wrapper: PybulletWrapper):
        """
        Initialize the gait finite state machine

        Args:
            conf: the configuration of the talos robot 
            support_foot: the very initial support foot pose for initialization
            pb_wrapper: the pybullet wrapper for visualization
        """
        self.footstepsList = []                  # the raw footsteps for planning
        self.plot_id_list = []                   # the plot id list for visualization
        self.stop_mode = False                   # the stop mode flag
        self.pb_wrapper = pb_wrapper             # the pybullet wrapper for visualization
        self.no_steps_per_horizon = conf.no_steps_per_horizon # the number of steps per horizon
        self.conf = conf                         # the configuration of the talos robot

        self.footstep_planner = FootStepPlanner(step_length, conf.step_size_y, conf.lfxp, conf.lfxn, conf.lfyp, conf.lfyn)
        self.footstepsList = self.footstep_planner.plan_steps(support_foot.get_pose(), support_foot.side, self.no_steps_per_horizon, 0, walking_start=True, walking_stop=False) 
        self.__display_footsteps()

    def __enter__(self):
        """Enter the context manager (with block)"""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager (with block)"""
        self.undisplay_footsteps()

    def __display_single_footstep(self, footstep: FootStep):
        """Display a single footstep in Pybullet
        Args:
            footstep: the footstep to be displayed
        """
        plot_ids = [0, 0, 0, 0, 0]
        
        xyzquat = pin.SE3ToXYZQUAT(footstep.get_pose())
        x = xyzquat[0:3]
        q = xyzquat[3:7]
        rect_size = [footstep.footprint_array[0]+footstep.footprint_array[1], footstep.footprint_array[2]+footstep.footprint_array[3]]
        
        plot_ids[0:4] = self.pb_wrapper.addGlobalDebugRectancle(footstep.get_center_pose().translation, q, rect_size[0], rect_size[1])
        plot_ids[4]   = self.pb_wrapper.addGlobalDebugLine(x, x + footstep.footprint_array[1]*footstep.pose.rotation[:, 0], lineWidth=2)

        self.plot_id_list.append(plot_ids)

    def __display_footsteps(self):
        """Display all the footsteps in Pybullet"""
        for footstep in self.footstepsList:
            self.__display_single_footstep(footstep)
    
    def __display_last_footstep(self):
        """Display the last footstep in Pybullet"""
        footstep = self.footstepsList[-1]
        self.__display_single_footstep(footstep)

    def undisplay_footsteps(self):
        """Undisplay all the footsteps in Pybullet"""
        for plot_ids in self.plot_id_list:
            for plot_id in plot_ids:
                self.pb_wrapper.removeDebugItem(plot_id)

    def __undisplay_first_footstep(self):
        """Undisplay the first footstep in Pybullet"""
        plot_ids = self.plot_id_list[0]
        for plot_id in plot_ids:
            self.pb_wrapper.removeDebugItem(plot_id)
        self.plot_id_list.pop(0)

    def init_horizon(self):
        """Initialize the footstep horizon for the first time
          ^
        [[1]]         
        [[0]]   [2]
        |--ZMP--|-----
        This is always the first horizon for the footstep planning, which does not require any remove.
        After [1] is stepped on and become the support foot, the next horizon will be same as next_horizon().

        Call this function: (with default self.no_steps_per_horizon = 2)
        1 will be the swing foot target
        [0,1,2] will be the footstep horizon for MPC.

        Args:
            None

        Returns:
            swing_foot_target (pin.SE3): the next swinging foot target
            foot_horizon (list(FootStep)): the footstep horizon for MPC, including 3 footstep:
              current support foot, current swing foot target, and the planned next footstep. 
        """
        swing_foot_target = self.footstepsList[1].get_pose()
        foot_horizon = list(self.footstepsList)
        return swing_foot_target, foot_horizon

    def next_horizon(self, step_length: float, curvature: float, step_width: float, angle: float = 0.0):
        """Plan the next horizon
               [[1]]            [3]
        [[0]]->         [2]
               |---ZMP horz-----|----------

        Call this function: (with default self.no_steps_per_horizon = 2)
        2 will be the swing foot target
        0 will be popped
        3 will be planned for the next new step
        [1,2,3] will be the footstep horizon for MPC

        - output the next swinging foot target
        - pop the last support foot (which is now just lifted)
        - plan for the next new step (if it is not the stop mode)
        - output a series of footstep for MPC planning

        If the robot is not in the stop mode, [3] will be planned according to the step length and curvature.
        If the robot is in the stop mode, nothing will be planned further ([3] is already set parallel to [2]).

        Args:
            step_length: the step length
            curvature: the step curvature
            angle: the step angle (when step_length and curvature are both 0, only turn the robot feet)

        Returns:
            swing_foot_target (pin.SE3): the next swinging foot target
            foot_horizon (list(FootStep)): the footstep horizon for MPC, including 3 footstep:
              current support foot, current swing foot target, and the planned next footstep.
            is_terminal (bool): whether the current support foot is the terminal footstep, which [3] is parallel to [2]
              and nothing will be planned further.
        """
        swing_foot_target = self.footstepsList[2].get_pose()
        
        self.footstepsList.pop(0)
        self.__undisplay_first_footstep()

        if self.stop_mode:
            foot_horizon = list(self.footstepsList)
            is_terminal = True
        else:
            last_footstep = self.footstepsList[-1]
            self.footstep_planner.update_step_size(step_length, step_width)
            if angle == 0.0:
                new_footstep = self.footstep_planner.plan_steps(last_footstep.get_pose(), last_footstep.side, 2, curvature)[1]
            else:
                new_footstep = self.footstep_planner.plan_steps_rotate(last_footstep.get_pose(), last_footstep.side, 2, angle)[1]

            self.footstepsList.append(new_footstep)
            self.__display_last_footstep()

            foot_horizon = list(self.footstepsList)
            is_terminal = False

        return swing_foot_target, foot_horizon, is_terminal

    def to_stop_mode(self):
        """Set the robot to the stop mode. 
        
        In this case, a footstep will be planned parallel to the current support foot, and nothing will be planned further.

        Args:
            None

        Returns:
            None
        """
        self.stop_mode = True

        last_footstep = self.footstepsList[-1]
        new_footstep = self.footstep_planner.plan_steps(last_footstep.get_pose(), last_footstep.side, 0, 0, walking_stop=True)[1]
        self.footstepsList.append(new_footstep)
        self.__display_last_footstep()

def debugRemoveAndAdd(id_debug_list, simulator, swing_foot_target, foot_horizon):
    # for id_debug in id_debug_list:
    #     pb.removeBody(id_debug)
    # id_debug_list = []
    # id_debug = simulator.addSphereMarker(swing_foot_target.translation, [0, 0, 0, 1], color=[0, 0, 1, 1])
    # id_debug_list.append(id_debug)
    # for footstep in foot_horizon:
    #     id_debug = simulator.addSphereMarker(footstep.get_center_pose().translation, [0, 0, 0, 1], color=[0, 1, 0, 1])
    #     id_debug_list.append(id_debug)
    return id_debug_list

if __name__=='__main__':
    """ 
    Visualization test of gait_fsm in pybullet.
    """
    import time
    from bullet_sims import talos_conf as conf
    simulator = PybulletWrapper()
    start_pose = pin.SE3.Identity()
    start_step = FootStep(start_pose, np.array([conf.lfxp, conf.lfxn, conf.lfyp, conf.lfyn]), Side.RIGHT)
    
    with GaitFSM(conf, start_step, 0.2, simulator) as gait_fsm:
        
        swing_foot_target, foot_horizon = gait_fsm.init_horizon()
        print("Init horizon")
        id_debug_list = []
        id_debug_list = debugRemoveAndAdd(id_debug_list, simulator, swing_foot_target, foot_horizon)

        time.sleep(1)
        
        for i in range(3):
            swing_foot_target, foot_horizon, is_terminal = gait_fsm.next_horizon(0.2, 0, 0.095)
            assert not is_terminal
            id_debug_list = debugRemoveAndAdd(id_debug_list, simulator, swing_foot_target, foot_horizon)
            time.sleep(0.5)

        for i in range(3):
            swing_foot_target, foot_horizon, is_terminal = gait_fsm.next_horizon(0.2, 0.5, 0.095)
            assert not is_terminal
            id_debug_list = debugRemoveAndAdd(id_debug_list, simulator, swing_foot_target, foot_horizon)
            time.sleep(0.5)

        for i in range(3):
            swing_foot_target, foot_horizon, is_terminal = gait_fsm.next_horizon(0.2, -0.5, 0.095)
            assert not is_terminal
            id_debug_list = debugRemoveAndAdd(id_debug_list, simulator, swing_foot_target, foot_horizon)
            time.sleep(0.5)

        for i in range(5):
            swing_foot_target, foot_horizon, is_terminal = gait_fsm.next_horizon(0, 0, 0.095, angle = 0.25)
            assert not is_terminal
            id_debug_list = debugRemoveAndAdd(id_debug_list, simulator, swing_foot_target, foot_horizon)
            time.sleep(0.5)

        gait_fsm.to_stop_mode()

        swing_foot_target, foot_horizon, is_terminal = gait_fsm.next_horizon(0.2, 0, 0.095)
        assert is_terminal
        id_debug_list = debugRemoveAndAdd(id_debug_list, simulator, swing_foot_target, foot_horizon)
        time.sleep(0.5)
        for id_debug in id_debug_list:
            pb.removeBody(id_debug)

    time.sleep(0.5)