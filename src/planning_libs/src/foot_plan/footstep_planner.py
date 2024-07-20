#!/usr/bin/env python3
import numpy as np
import pinocchio as pin
from enum import Enum
import pybullet as pb
from scipy.spatial.transform import Rotation as R

# Files
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
from walking_control import talos_conf as conf

class Side(Enum):
    """Side
    Describes which foot to use
    """
    LEFT=0
    RIGHT=1

def other_foot_id(id):
    """
    Args:
        id (Side): either Side.RIGHT or Side.LEFT

    Return: 
        the side of the other foot, given an id
    """
    if id == Side.LEFT:
        return Side.RIGHT
    else:
        return Side.LEFT
        
class FootStep:
    """FootStep
    Holds all information describing a single footstep
    """
    def __init__(self, pose, footprint, side=Side.LEFT, offset_translation = np.zeros(3)):
        """init FootStep

        Args:
            pose (pin.SE3): the pose of the footstep
            footprint (np.array): 3 by n matrix of foot vertices
            side (_type_, optional): Foot identifier. Defaults to Side.LEFT.
        """
        self.pose = pose
        self.footprint = footprint
        self.side = side
        self.offset = offset_translation
        
    def poseInWorld(self):
        """
        Returns:
            pose (pin.SE3)
        """
        return self.pose
        
    def plot(self, simulation):
        """
        Args:
            simulation (PybulletWrapper()): the pose of the footstep
        
        Returns:
            plot of lines in pybullet simulation environment
        """

        # plot in pybullet: recangular footprint in wolrd coordinate frame 
        simulation.addGlobalDebugRectancle(self.pose.translation + self.offset, 
                                           Q = [pin.Quaternion(self.pose.rotation).x,pin.Quaternion(self.pose.rotation).y,pin.Quaternion(self.pose.rotation).z,pin.Quaternion(self.pose.rotation).w])
        
        # plot in pybullet: display the side of the step
        if self.side == Side.LEFT:
            simulation.addUserDebugText(0, 0, 'LEFT', self.pose.translation + self.offset)
        else:
            simulation.addUserDebugText(0, 0, 'RIGHT', self.pose.translation + self.offset)
        
        # plot in pybullet: plot step target position with a spherical marker 
        simulation.addSphereMarker(self.pose.translation + self.offset, [0,0,1], radius=0.02)
        
        return None

class FootStepPlanner:
    """FootStepPlanner
    Creates footstep plans (list of right and left steps)
    """
    
    def __init__(self, step_size_x = 0.20, step_size_y = 0.096, foot_scaling = 1., lfxp = 0.12, lfxn = 0.08, lfyp = 0.065, lfyn = 0.065):
        self.steps = []

        ################################################################################
        # walking
        ################################################################################

        # step dimensions
        self.step_size_x = step_size_x               # step size in x direction
        self.step_size_y = step_size_y               # step size in y direction

        # the displacement between steps in x and y direction
        self.dx = self.step_size_x
        self.dy = 2*self.step_size_y

        ################################################################################
        # foot print
        ################################################################################
        
        # the footprint of the robot
        self.foot_scaling = foot_scaling                  # scaling factor of foot dimensions 
        self.lfxp = self.foot_scaling*lfxp                # foot length in positive x direction
        self.lfxn = self.foot_scaling*lfxn                # foot length in negative x direction
        self.lfyp = self.foot_scaling*lfyp                # foot length in positive y direction
        self.lfyn = self.foot_scaling*lfyn                # foot length in negative y direction
        
        self.footprint = np.array([[self.lfxp, self.lfxp, self.lfxn, self.lfxn],
                                   [self.lfyp, self.lfyn, self.lfyn, self.lfyp],
                                   [0, 0, 0, 0]])
        
    def planLine(self, T_0_w, side, no_steps, turn_angle, walking_start = False, walking_stop = False):
        """plan a sequence of steps in a strait line

        Args:
            T_0_w (pin.SE3): The inital starting position of the plan. holds the angular and cartesian transformation of the start of the sequence from first foot local frame to global frame
            side (Side): The intial foot for starting the plan
            no_steps (int): The number of steps to take
            turn_angle (float): The angular rotation around the local z-axis of each footstep to the  following in radians
            walking_start (bool): flag, that tells the planner whether the robot is currently standing still 
            walking_stop (bool): flag, that tells the planner whether the robot needs to stop after the executed amount of steps 
            
        Returns:
            list: sequence of steps
        """

        steps=[]

        #Plan a sequence of steps with T_0_w being the first step pose.
        pose = pin.SE3(T_0_w.rotation, np.array([0, 0, 0]))

        if walking_start == False:
            side = other_foot_id(side)

        if walking_start:
            # First Step: T_0_W
            pose = pin.SE3(T_0_w.rotation, np.array([0, 0, 0]))
            steps.append(FootStep(pose, self.footprint, side, T_0_w.translation))

            # Second Step: Parallel to the first step
            side = other_foot_id(side)
            step_trans = T_0_w.rotation @ np.array([0, self.dy, 0])
            pose = pin.SE3(T_0_w.rotation, step_trans) 
            steps.append(FootStep(pose, self.footprint, side, T_0_w.translation))

        # Step Sequence
        T_step_w = pin.SE3(T_0_w.rotation, np.array([0, 0, 0]))
        T_foot_w = pin.SE3(np.eye(3), np.array([0, 0, 0]))
        for i in range(2, no_steps):
            side = other_foot_id(side)
            T_foot_w *= pin.SE3(np.array([[np.cos(turn_angle), -np.sin(turn_angle), 0],
                                          [np.sin(turn_angle),  np.cos(turn_angle), 0],
                                          [0,                                    0, 1]]), 
                                np.array([0, 0, 0]))
            step_translation =  T_foot_w @ np.array([self.dx, self.dy*(-1)**(i-1), 0, 1])
            pose = pin.SE3(T_step_w.rotation @ T_foot_w.rotation, pose.translation + T_step_w.rotation @ step_translation[0:3])
            steps.append(FootStep(pose, self.footprint, side, T_0_w.translation))

        if walking_stop:
            # Last Step: Parallel to the last step
            side = other_foot_id(side)
            step_translation = T_foot_w @ np.array([0, self.dy*(-1)**(i), 0, 1])
            pose = pin.SE3(pose.rotation, pose.translation + T_step_w.rotation @ step_translation[0:3])  
            steps.append(FootStep(pose, self.footprint, side, T_0_w.translation))
                                    
        self.steps = steps
        return steps

    
    def plot(self, simulation):
        """
        Args:
            simulation (PybulletWrapper())
            
        Returns:
            plot of lines of planned footsteps in pybullet simulation
        """

        for step in self.steps:
            step.plot(simulation)

            
if __name__=='__main__':
    """ 
    Test footstep planner in pybullet.
    Plot three lines with 20 footsteps each, a specified turn angle.
    The first line starts at a specified position and orientation with both feet flat on the ground. 
    The second line is appended onto the first with its own specified angle.
    The third line stops the footstep planning by ending with both final footsteps next to one another.
    """
    
    # Generate a plan and plot it in pybullet.
    # Check that the plan looks as expected
    simulator = PybulletWrapper()
    foot_planner = FootStepPlanner(conf)
    turn_angle = 0.2
    no_steps = 20

    #first trajectory
    turn_angle_glob = -0.5
    T_0_w = pin.SE3(np.array([[np.cos(turn_angle_glob), -np.sin(turn_angle_glob), 0],
                              [np.sin(turn_angle_glob),  np.cos(turn_angle_glob), 0],
                              [0,                                              0, 1]]), 
                    np.array([1, -0.5, 0]))
    steps  = foot_planner.planLine(T_0_w, Side.RIGHT, no_steps,  turn_angle, walking_start = True, walking_stop = False)
    foot_planner.plot(simulator)

    # second trajectory 
    foot_planner2 = FootStepPlanner(conf)
    foot_planner.steps[-1].pose.translation += foot_planner.steps[-1].offset
    steps2 = foot_planner.planLine(foot_planner.steps[-1].pose, Side.RIGHT, no_steps, -turn_angle, walking_start = False, walking_stop = False)
    foot_planner.plot(simulator)

    # third trajectory
    foot_planner.steps[-1].pose.translation += foot_planner.steps[-1].offset
    steps3 = foot_planner.planLine(foot_planner.steps[-1].pose, Side.RIGHT, no_steps, 0.5 * turn_angle, walking_start = False, walking_stop = True)
    foot_planner.plot(simulator)

    while True:
        pb.stepSimulation()