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
    def __init__(self, conf, support_foot: FootStep, pb_wrapper: PybulletWrapper):
        """
        Initialize the gait finite state machine

        Args:
            conf: the configuration of the talos robot 
            support_foot: the very initial support foot pose for initialization
            pb_wrapper: the pybullet wrapper for visualization
        """
        self.footstepsList = []     # the raw footsteps for planning
        self.stop_mode = False      # the stop mode flag
        self.pb_wrapper = pb_wrapper

    def next_horizon(self, step_length: float, curvature: float):
        """Plan the next horizon
               [[1]]            [3]
        [[0]]->         [2]
               |---ZMP horz-----|

        Call this function: 
        2 will be the next support foot
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

        Returns:
            swing_foot_target (pin.SE3): the next swinging foot target
            foot_horizon (list(FootStep)): the footstep horizon for MPC, including 3 footstep:
              current support foot, current swing foot target, and the planned next footstep.
            is_terminal (bool): whether the current support foot is the terminal footstep, which [3] is parallel to [2]
              and nothing will be planned further.
        """
        pass

    def to_stop_mode(self):
        """Set the robot to the stop mode. 
        
        In this case, a footstep will be planned parallel to the current support foot, and nothing will be planned further.

        Args:
            None

        Returns:
            None
        """
        self.stop_mode = True
        pass

