#!/usr/bin/env python3
import numpy as np
import pinocchio as pin
from enum import Enum
import pybullet as pb

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
    if id == Side.LEFT:
        return Side.RIGHT
    else:
        return Side.LEFT
        
class FootStep:
    """FootStep
    Holds all information describing a single footstep
    """
    def __init__(self, pose, footprint, side=Side.LEFT):
        """inti FootStep

        Args:
            pose (pin.SE3): the pose of the footstep
            footprint (np.array): 3 by n matrix of foot vertices
            side (_type_, optional): Foot identifier. Defaults to Side.LEFT.
        """
        self.pose = pose
        self.footprint = footprint
        self.side = side
        
    def poseInWorld(self):
        return self.pose
        
    def plot(self, simulation):
        
        #>>>>TODO: plot in pybullet footprint, addGlobalDebugRectancle(...) 
        simulation.addGlobalDebugRectancle(self.pose.translation)
        
        #>>>>TODO: display the side of the step, addUserDebugText(...)
        # simulation.addUserDebugText(0, 0, self.side.name, self.pose.translation) 
        if self.side == Side.LEFT:
            simulation.addUserDebugText(0, 0, 'LEFT', self.pose.translation)
        else:
            simulation.addUserDebugText(0, 0, 'RIGHT', self.pose.translation)
        
        #>>>>TODO: plot step target position addSphereMarker(...)
        simulation.addSphereMarker(self.pose.translation, [0,0,1], radius=0.02)
        
        return None

class FootStepPlanner:
    """FootStepPlanner
    Creates footstep plans (list of right and left steps)
    """
    
    def __init__(self, conf):
        self.conf = conf
        self.steps = []
        
    def planLine(self, T_0_w, side, no_steps):
        """plan a sequence of steps in a strait line

        Args:
            T_0_w (pin.SE3): The inital starting position of the plan
            side (Side): The intial foot for starting the plan
            no_steps (_type_): The number of steps to take

        Returns:
            list: sequence of steps
        """
        
        # the displacement between steps in x and y direction
        dx = self.conf.step_size_x
        dy = 2*self.conf.step_size_y
        
        # the footprint of the robot
        lfxp, lfxn = self.conf.lfxp, self.conf.lfxn
        lfyp, lfyn = self.conf.lfyp, self.conf.lfyn
        
        #>>>>TODO: Plan a sequence of steps with T_0_w being the first step pose.
        #>>>>Note: Plan the second step parallel to the first step (robot starts standing on both feet)
        #>>>>Note: Plan the final step parallel to the last-1 step (robot stops standing on both feet)
        steps=[]

        # First Step: T_0_W
        footprint = np.array([[lfxp, lfxp, lfxn, lfxn],
                              [lfyp, lfyn, lfyn, lfyp],
                              [0, 0, 0, 0]])
        pose = pin.SE3(T_0_w.rotation, T_0_w.translation + np.array([0, 0, 0]))
        steps.append(FootStep(pose, footprint, side))

        # Second Step: Parallel to the first step
        side = other_foot_id(side)
        pose = pin.SE3(T_0_w.rotation, T_0_w.translation + np.array([0, dy, 0]))
        steps.append(FootStep(pose, footprint, side))

        # Step Sequence
        for i in range(2, no_steps):
            side = other_foot_id(side)
            pose = pin.SE3(T_0_w.rotation, pose.translation + np.array([dx, dy*(-1)**(i-1), 0])) # type: ignore
            steps.append(FootStep(pose, footprint, side))

        # Last Step: Parallel to the last step
        side = other_foot_id(side)
        pose = pin.SE3(pose.rotation, pose.translation + np.array([0, dy*(-1)**(i), 0]))  # type: ignore
        steps.append(FootStep(pose, footprint, side))
                                
        self.steps = steps
        return steps

    
    def plot(self, simulation):
        for step in self.steps:
            step.plot(simulation)

            
if __name__=='__main__':
    """ Test footstep planner
    """
    
    #>>>>TODO: Generate a plan and plot it in pybullet.
    simulator = PybulletWrapper()
    foot_planner = FootStepPlanner(conf)
    T_0_w = pin.SE3(np.eye(3), np.array([0, 0, 0]))
    steps = foot_planner.planLine(T_0_w, Side.RIGHT, 10)
    foot_planner.plot(simulator)

    #>>>>TODO: Check that the plan looks as expected


    while True:
        pb.stepSimulation()