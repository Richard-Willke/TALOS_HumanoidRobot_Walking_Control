#!/usr/bin/env python3
import rospy
import numpy as np
import pinocchio as pin
import pybullet as pb

from simulator.robot import Robot
from simulator.pybullet_wrapper import PybulletWrapper

from bullet_sims.talos import Talos
from bullet_sims import talos_conf as conf

from foot_plan.footstep_planner import Side, FootStep
from foot_plan.foot_trajectory import SwingFootTrajectory
from zmp_plan.lip_mpc import LIPMPC, LIPInterpolator

from state_machines.gait_fsm import GaitFSM
from state_machines.mpc_fsm import MPC_FSM
from debugger.debugger import Debugger

from enum import Enum

DEBUG_INFO = True
debugger = Debugger()

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

        self.joy_stick_vertical = 0.0
        self.joy_stick_horizontal = 0.0
        self.step_length_command = 0.15
        self.curvature_command = 0.2

        self.start_prewalk_clock = 0.0
        self.start_step_clock = 0.0
        self.N_walk = 0
        self.to_post_walk = False

        # initialize some extra classes
        self.foot_trajectory = SwingFootTrajectory(pin.SE3.Identity(), pin.SE3.Identity(), conf.step_dur)
        self.mpc = LIPMPC(conf)
        self.interpolator = None
        self.gait_fsm = None
        self.mpc_fsm = None
        self.com_ankle_desire = np.array([0, 0, 0])
        self.com_ankle_desire_dot = np.array([0, 0, 0])

        # initialize the robot reference
        c = np.array([0, 0, 0.85])
        zero_array = np.zeros(3)
        robot.stack.setComRefState(c, zero_array, zero_array)

    def joy_callback(self, msg):
        pass

    def detect_state_transition(self):
        if self.state == State.INIT:
            if self.t > 1.5:
                self.state = State.STAND
                rospy.loginfo("[INIT] --> [STAND]")

        elif self.state == State.STAND:
            if self.step_length_command > 0.0:
                T_support_w = self.robot.supportFootPose()  
                T_support_w.rotation = pin.exp3(np.array([0,0,pin.log3(T_support_w.rotation)[2]]))
                start_step = FootStep(T_support_w, np.array([conf.lfxp, conf.lfxn, conf.lfyp, conf.lfyn]), self.robot.support_foot)

                self.gait_fsm = GaitFSM(conf, start_step, self.step_length_command, self.simulator)

                x0 = np.array([start_step.get_center_pose().translation[0], 0, start_step.get_center_pose().translation[1], 0])
                self.interpolator = LIPInterpolator(x0, conf)

                c, c_dot, c_ddot = self.interpolator.comState() 
                robot.stack.setComRefState(c, c_dot, c_ddot)
                
                self.start_prewalk_clock = self.t
                self.state = State.PRE_WALK
                rospy.loginfo("[STAND] --> [PRE_WALK]")

        elif self.state == State.PRE_WALK:
            if self.t - self.start_prewalk_clock > 2.0:
                robot.unContactSwingFoot()
                self.N_walk = 0
                self.to_post_walk = False
                self.com_ankle_desire = self.robot.stack.comState().value()
                self.com_ankle_desire_dot = self.robot.stack.comState().derivative()
                
                self.state = State.WALK
                rospy.loginfo("[PRE_WALK] --> [WALK]")

        elif self.state == State.WALK:
            if self.to_post_walk:
                self.state = State.POST_WALK
                rospy.loginfo("[WALK] --> [POST_WALK]")

        elif self.state == State.POST_WALK:
            pass


    def main_execute(self):
        """The main function that controls the robot
           Call this function to start the ROS node and execute the control
        """
        rospy.init_node('talos_walk_jump')
        rospy.loginfo("Starting the talos_walk_jump node")

        # setup some constants needed
        dt = self.simulator.stepTime()                # simulator dt
        steps_per_horizon = conf.no_steps_per_horizon # default 2
        sim_per_mpc = conf.no_sim_per_mpc             # default 100

        # setup some variables needed
        t_publish = 0.0
        is_last = False

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
                if self.N_walk == 0:
                    swing_foot_target, foot_horizon = self.gait_fsm.init_horizon()
                    swing_pose_current = self.robot.swingFootPose()
                    self.foot_trajectory.reset(swing_pose_current, swing_foot_target)
                    self.start_step_clock = self.t

                    self.mpc_fsm = MPC_FSM(conf, foot_horizon, steps_per_horizon + 2, self.mpc)
                
                if self.N_walk % sim_per_mpc == 0:
                    if is_last:
                        self.robot.contactSwingFoot()
                        self.robot.setSwingFoot(self.robot.support_foot)
                        swing_foot_target, foot_horizon, is_terminal = self.gait_fsm.next_horizon(self.step_length_command, self.curvature_command)
                        swing_pose_current = self.robot.swingFootPose()
                        self.foot_trajectory.reset(swing_pose_current, swing_foot_target)
                        self.start_step_clock = self.t

                        if not is_terminal:
                            self.mpc_fsm = MPC_FSM(conf, foot_horizon, steps_per_horizon + 2, self.mpc)
                        else:
                            self.mpc_fsm = MPC_FSM(conf, foot_horizon, steps_per_horizon + 1, self.mpc)
                            self.to_post_walk = True

                        is_last = False

                    x0 = self.interpolator.x
                    u_k, is_last = self.mpc_fsm.next_horizon(x0)
                
                T_swing, V_swing, A_swing = self.foot_trajectory.evaluate(self.t - self.start_step_clock)
                robot.updateSwingFootRef(T_swing, V_swing, A_swing)

                self.interpolator.integrate(u_k)
                c, c_dot, c_ddot = self.interpolator.comState()
                robot.stack.setComRefState(c, c_dot, c_ddot)

                omega = self.robot.compute_omega()
                Kp = omega - 1.5
                Kx = omega + 1.0
                zmp_real = self.robot.compute_ZMP()[0]
                zmp_ref = np.hstack([u_k,0])
                self.com_ankle_desire_dot = c_dot - Kx * (self.com_ankle_desire - c) + Kp * (zmp_real - zmp_ref)
                self.com_ankle_desire += self.com_ankle_desire_dot * dt
                robot.stack.setComRefState(self.com_ankle_desire, self.com_ankle_desire_dot)
                
                robot.stack.setTorsoRefState(T_swing)
                
                self.N_walk += 1

            elif self.state == State.POST_WALK:
                pass
                self.N_walk += 1

            self.detect_state_transition()

            # run the simulation
            self.simulator.step()
            self.robot.update()

            # extra visualization publish to ros
            if self.t - t_publish > 1./30.:
                t_publish = self.t
                self.robot.publish()

            # record debug info
            try:
                u_k
            except NameError:
                u_k = None 
            if DEBUG_INFO:
                if u_k is not None:
                    u_k_debug = np.hstack([u_k, 0])
                else:
                    u_k_debug = np.array([0, 0, 0])
                if self.com_ankle_desire is not None:
                    com_ankle_desire_debug = np.array(self.com_ankle_desire)
                else:
                    com_ankle_desire_debug = np.array([0, 0, 0])

                if self.com_ankle_desire_dot is not None:
                    com_ankle_desire_dot_debug = np.array(self.com_ankle_desire_dot)
                else:
                    com_ankle_desire_dot_debug = np.array([0, 0, 0])
                debugger.record(self.robot, self.interpolator, dt, u_k_debug, com_ankle_desire_debug, com_ankle_desire_dot_debug)


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