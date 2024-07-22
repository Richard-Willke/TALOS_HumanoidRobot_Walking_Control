#!/usr/bin/env python3
"""
talos walking simulation
"""

import rospy
import numpy as np
import pinocchio as pin
import pybullet as pb


# simulator
# import simulator
from simulator.robot import Robot
from simulator.pybullet_wrapper import PybulletWrapper

# robot configs
# Import talos robot
from bullet_sims.talos import Talos

# modules
# Import all previously coded modules
from foot_plan.foot_trajectory import SwingFootTrajectory
from foot_plan.footstep_planner import Side, FootStepPlanner
from zmp_plan.lip_mpc import LIPMPC, generate_zmp_reference, LIPInterpolator
from bullet_sims import talos_conf as conf
        
################################################################################
# main
################################################################################  
    
def main(): 
    
    ############################################################################
    # setup
    ############################################################################
    
    simulator = PybulletWrapper()
    robot = Talos(simulator)
    
    t_step_elapsed = 0.0                                # elapsed time within current step (use to evaluate spline)
    t_publish = 0.0                                     # last publish time (last time we published something)
    


    ########################################################################
    # update the simulation
    ########################################################################

    while not rospy.is_shutdown():
        t = simulator.simTime()
        simulator.step()
        robot.update()


        # publish to ros
        if t - t_publish > 1./30.:
            t_publish = t
            # publish
            # T_b_w = robot.robot.baseCoMPosition()
            robot.publish()
            
  



if __name__ == '__main__': 
    rospy.init_node('talos_walking')
    main()
