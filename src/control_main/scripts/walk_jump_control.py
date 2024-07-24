#!/usr/bin/env python3
import rospy
import numpy as np
import pinocchio as pin
import pybullet as pb

from simulator.robot import Robot
from simulator.pybullet_wrapper import PybulletWrapper

from bullet_sims.talos import Talos

from foot_plan.footstep_planner import Side, FootStep
from foot_plan.foot_trajectory import SwingFootTrajectory

from zmp_plan.lip_mpc import LIPMPC, LIPInterpolator
from bullet_sims import talos_conf as conf
from enum import Enum

class State(Enum):
    """The state of the robot"""
    INIT = 0
    STAND = 1
    PRE_WALK = 2
    WALK = 3
    POST_WALK = 4

class Main_Controller:
    "The main class that controls the robot"
    def __init__(self, simulator: PybulletWrapper, robot: Talos):
        # setup the simulator and robot
        self.simulator = simulator
        self.robot = robot

        # initialize some states and variables
        self.state = State.INIT
        self.t = 0.0
        self.go_command = 0.0
        self.turn_command = 0.0

        T_support_w = self.robot.supportFootPose()  
        T_support_w.rotation = np.eye(3)
        start_step = FootStep(T_support_w, np.array([conf.lfxp, conf.lfxn, conf.lfyp, conf.lfyn]), Side.RIGHT)
        x0 = np.array([start_step.get_center_pose().translation[0], 0, start_step.get_center_pose().translation[1], 0])

        # initialize some extra classes
        self.foot_trajectory = SwingFootTrajectory(pin.SE3.Identity(), pin.SE3.Identity(), conf.step_dur)
        self.mpc = LIPMPC(conf)
        self.interpolator = LIPInterpolator(x0, conf)

        # initialize the robot state
        # c, c_dot, c_ddot = self.interpolator.comState()
        # robot.stack.setComRefState(c, c_dot, c_ddot)

    def joy_callback(self, msg):
        pass

    def command_callback(self, msg):
        pass

    def detect_state_transition(self):
        if self.state == State.INIT:
            if self.t > 2.0:
                self.state = State.STAND
        elif self.state == State.STAND:
            pass
        elif self.state == State.PRE_WALK:
            pass
        elif self.state == State.WALK:
            pass
        elif self.state == State.POST_WALK:
            pass

    def main_execute(self):
        """The main function that controls the robot
           Call this function to start the ROS node and execute the control
        """
        rospy.init_node('talos_walk_jump')
        rospy.loginfo("Starting the talos_walk_jump node")

        # setup some constants needed
        dt = self.simulator.stepTime()   # simulator dt

        # setup some variables needed
        t_publish = 0.0

        while not rospy.is_shutdown():
            # read the status
            self.t = self.simulator.simTime()     # simulator time
            
            # control the robot and transit the state
            if self.state == State.INIT:
                pass
            elif self.state == State.STAND:
                pass
            elif self.state == State.PRE_WALK:
                pass
            elif self.state == State.WALK:
                pass
            elif self.state == State.POST_WALK:
                pass

            self.detect_state_transition()

            # run the simulation
            self.simulator.step()
            self.robot.update()

            # extra visualization publish to ros
            if t - t_publish > 1./30.:
                t_publish = t
                self.robot.publish()

            # record debug info
            # Add in need


if __name__=='__main__':
    ############################################################################
    # setup
    ############################################################################
    # setup the simulator
    simulator = PybulletWrapper()
        
    # setup the robot
    robot = Talos(simulator)
    
    ############################################################################
    # execute
    ############################################################################
    # setup the Main_Controller
    main_controller = Main_Controller(simulator, robot)
    main_controller.main_execute()