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
      - Could output the current and next swinging footstep in use
    For the queue input
      - Could interrupt the footstep planning queue
      - When the Step queue is not long enough, the reading will still automatically append to the result.
    """
    def __init__(self, n_queue: int, n_forward: int, T_support_w: pin.SE3, pb_wrapper: PybulletWrapper):
        """
        Initialize the gait finite state machine

        Args:
            n_queue (int): number of footsteps minimum in the queue
            n_forward (int): number of footsteps to plan in advance
            T_support_w (pin.SE3): the support foot pose for initialization
            pb_wrapper (PybulletWrapper): the pybullet wrapper for visualization
        """
        self.footstepsRaw = []     # the raw footsteps for planning
        self.footstepsFull = []    # the full footsteps for planning
        self.pb_wrapper = pb_wrapper
        self.updateStateParams(n_queue, n_forward)

    def updateStateParams(self, n_queue: int, n_forward: int):
        """
        Update the state parameters for the gait

        Args:
            n_queue (int): number of footsteps minimum in the queue
            n_forward (int): number of footsteps to plan in advance
        """
        self.n_queue = n_queue
        self.n_forward = n_forward

    def getCurrentFootStepData(self):
        """
        Get the current footstep (which is the first in the queue)

        Returns:
            CurrentSwingingStep: the current swinging step
            NextSwingingStep: the next swinging step
        """
        pass
    
    def popCurrentFootStepData(self):
        """
        Pop the current footstep (which is the first in the queue)
        """
        pass

    def planFootSteps(self, velocity_command: float, curvature: float, is_initial: bool, is_terminal: bool):
        """
        Attempt to plan the current footstep

        Args:
            velocity_command (float): the velocity command for the current footstep. If zero, it refuses to plan.
            curvature (float): the curvature of the current footstep. Left is positive, right is negative.
            is_initial (bool): Whether we assume current footstep is the initial footstep (have parallel step by start).
            is_terminal (bool): Whether the current footstep is the terminal footstep (have parallel step by end).
        """
        pass

    def interruptFootSteps(self):
        """
        Interrupt all the future footstep planning and clear the queue for a immediate stop
        (Will replan last three steps)
        """
        pass

