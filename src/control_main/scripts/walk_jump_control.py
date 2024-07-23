#!/usr/bin/env python3
import rospy
import numpy as np
import pinocchio as pin
import pybullet as pb

from simulator.robot import Robot
from simulator.pybullet_wrapper import PybulletWrapper

from bullet_sims.talos import Talos

from foot_plan.foot_trajectory import SwingFootTrajectory
from foot_plan.footstep_planner import Side, FootStepPlanner
from zmp_plan.lip_mpc import LIPMPC, generate_zmp_reference, LIPInterpolator
from bullet_sims import talos_conf as conf

DEBUG_LOGGING = False

class Main_Controller:
    pass

def main(): 
    ############################################################################
    # setup
    ############################################################################
    
    # setup the simulator
    simulator = PybulletWrapper()
        
    # setup the robot
    robot = Talos(simulator)
    
    # inital footsteps
    T_swing_w = robot.swingFootPose()       
    T_support_w = robot.supportFootPose()   
                               
    # setup the plan
    planner = FootStepPlanner()

    # setup the LIPMPC and Interpolator to predict the CoM state
    mpc = LIPMPC(conf)
    x0 = np.array([T_support_w.translation[0], 0, T_support_w.translation[1], 0])
    interpolator = LIPInterpolator(x0, conf)
    c, c_dot, c_ddot = interpolator.comState()
    robot.stack.setComRefState(c, c_dot, c_ddot)