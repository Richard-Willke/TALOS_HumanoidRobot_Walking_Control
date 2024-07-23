#!/usr/bin/env python3
import numpy as np
import pinocchio as pin
import pybullet as pb
from enum import Enum

# Files
from simulator.pybullet_wrapper import PybulletWrapper
from bullet_sims import talos_conf as conf

class Side(Enum):
    """Describes which side of the foot to use"""
    LEFT=0
    RIGHT=1

def other_foot_id(id: Side):
    """Change the side of the foot
    
    Args:
        id (Side): either Side.RIGHT or Side.LEFT

    Return: 
        the side of the other foot, either Side.RIGHT or Side.LEFT
    """
    if id == Side.LEFT:
        return Side.RIGHT
    else:
        return Side.LEFT
        
class FootStep:
    """Holds all information describing one single footstep."""
    def __init__(self, pose: pin.SE3, footprint_array: np.array, side: Side):
        """constructor the FootStep

        Args:
            pose: the pose of the footstep
            footprint: 4 by 1 marking the footprint size of the footstep [lfxp, lfxn, lfyp, lfyn] (see talos_conf)
                       lfxp, lfxn could be different, but we handle lfyp, lfyn always equal to foot in the planner
            side: Foot identifier marking the side of the footstep.

        Returns:
            FootStep object
        """
        # check if the z axis of the pose is pointing upwards
        if not np.allclose(pose.rotation[:, 2], np.array([0, 0, 1]), atol=1e-4):
            print("[ Warning ]: FootStep().__init__: The z-axis of the pose is not pointing upwards.\n", pose)
        self.pose = pose
        self.footprint_array = footprint_array
        self.side = side
        
    def get_pose(self):
        """ Return the pose of the footstep in the world frame.
        Args:
            None

        Returns:
            pose (pin.SE3): the pose of the footstep in the world frame
        """
        return pin.SE3(self.pose)
    
    def get_center_pose(self):
        """ Return the center pose of the footstep. This is used better for ZMP references or plotting, but is
        not necessary the coordinate of the footstep.
        
        Args:
            None

        Returns:
            zmp (pin.SE3): the Zero Moment Point of the footstep
        """
        center_pose = self.get_pose()
        rect_offset = np.array([(self.footprint_array[0] - self.footprint_array[1])/2, (self.footprint_array[2] - self.footprint_array[3])/2, 0])
        center_pose.translation =  center_pose.translation - self.pose.rotation @ rect_offset
        return center_pose
        
    def plot(self, simulation: PybulletWrapper, id: int):
        """Plot the footstep in the pybullet simulation. (For initial debugging, deprecated)
        
        Args:
            simulation: The pyBullet simulation environment for plotting
            id: the id of the footstep

        Returns:
            None
        """

        # get the position and orientation of the footstep
        xyzquat = pin.SE3ToXYZQUAT(self.pose)
        x = xyzquat[0:3]
        q = xyzquat[3:7]

        # plot in pybullet: recangular footprint in wolrd coordinate frame
        rect_size = [self.footprint_array[0]+self.footprint_array[1], self.footprint_array[2]+self.footprint_array[3]]
        simulation.addGlobalDebugRectancle(self.get_center_pose().translation, q, rect_size[0], rect_size[1])

        # plot in pybuttet: mark the positive x direction of the footstep
        simulation.addGlobalDebugLine(x, x + self.footprint_array[1]*self.pose.rotation[:, 0], lineWidth=2)
        
        # plot in pybullet: display the side of the step
        if self.side == Side.LEFT:
            simulation.addUserDebugText(0, 0, 'LEFT '+str(id), x)
        else:
            simulation.addUserDebugText(0, 0, 'RIGHT '+str(id), x)
        
        # plot in pybullet: plot step target position with a spherical marker 
        simulation.addSphereMarker(x, [0, 0, 0, 1])

class FootStepPlanner:
    """Creates footstep plans (list of right and left steps) for the robot to follow."""
    
    def __init__(self, step_size_x: float, step_size_y: float, lfxp: float, lfxn: float, lfyp: float, lfyn: float):
        """constructor of the FootStepPlanner

        [Notice]: here the x direction is the LOCAL forward direction of the robot.
        
        Args:
            step_size_x: step size in x direction
            step_size_y: step size in y direction (deviated from center line of the robot)
            lfxp: foot length in positive x direction
            lfxn: foot length in negative x direction
            lfyp: foot length in positive y direction
            lfyn: foot length in negative y direction
        
        Returns:
            FootStepPlanner object
        """
        self.steps = []
        self.update_step_size(step_size_x, step_size_y)
        self.footprint_array = np.array([lfxp, lfxn, lfyp, lfyn])

        self.deprecated_plot_warning = False

    def update_step_size(self, step_size_x: float, step_size_y: float):
        """Update the step size of the robot
        
        Args:
            step_size_x: step size in x direction
            step_size_y: step size in y direction (deviated from center line of the robot)

        Returns:
            None
        """
        self.step_size_x = step_size_x
        self.step_size_y = step_size_y

    def center_to_foot(self, T_center_w: pin.SE3, side: Side):
        """Calculate the pose of the foot from the center of the robot.
           During the walking along the curve, the center of the robot is the on the curve, while the foot is deviated from the center line.
           This is to deviate the foot from the center line.

                             ^x+ 
                   -------       --------
                   |     |       |      |
                   |     |       |      |
             <y+   |  0  |   x   |   1  |    
                   |     |       |      |
                   |     |       |      |
                   -------       --------
                                            w
           w: world frame
           x: T_center_w
           0: left foot T_foot_w
           1: right foot T_foot_w

        Args:
            T_center_w: the pose of the center of the robot in the world frame
            side: the side of the foot

        Returns:
            T_foot_w (pin.SE3): the pose of the foot in the world frame
        """
        T_foot_w = pin.SE3.Identity()
        T_foot_w.rotation = T_center_w.rotation
        if side == Side.LEFT:
            T_foot_w.translation = T_center_w.translation + T_center_w.rotation @ np.array([0, self.step_size_y, 0])
        else:
            T_foot_w.translation = T_center_w.translation + T_center_w.rotation @ np.array([0, -self.step_size_y, 0])
        return T_foot_w
    
    def rotate_with_center(self, T_center_w: pin.SE3, curvature: float):
        """Rotate the center of the robot with a given angle with given curvature and step size
                                      ^x+
                           c>0        c=0        c<0
                              1        2        3 
                                ' ,    |    , '   
                                    ,  |  ,       
                                     , | ,        
                                      ,|,         
                <y+    c--------------,x,---------------c
                          w
            w: world frame
            x: center of the robot
            c: rotation center of the circle (from curvature)
            1, 2, 3: the desired footstep position

        Args:
            T_center_w: the pose of the center of the robot in the world frame
            curvature: the curvature to be followed in the line, could be positive(left) or negative(right), 0 for straight line

        Returns:
            T_center_w (pin.SE3): the new pose of the center of the robot in the world frame
        """
        T_center_w_new = pin.SE3.Identity()     # new robot center (1,2,3)
        T_rotation_w = pin.SE3.Identity()       # from world to curve rotational center        (w->c)
        T_center_rotation = pin.SE3.Identity()  # from curve rotational center to robot center (c->x)

        if curvature == 0:
            T_center_w_new.translation = T_center_w.translation + T_center_w.rotation @ np.array([self.step_size_x, 0, 0])
            T_center_w_new.rotation = T_center_w.rotation
        else:
            T_rotation_w.translation = T_center_w.translation + T_center_w.rotation @ np.array([0, 1/curvature, 0])
            T_rotation_w.rotation = T_center_w.rotation
            T_center_rotation.translation = -np.array([0, 1/curvature, 0])

            angle = self.step_size_x * curvature
            T_rotation = pin.SE3(pin.rpy.rpyToMatrix(0, 0, angle), np.array([0, 0, 0]))

            T_center_w_new = T_rotation_w * T_rotation * T_center_rotation
        
        return T_center_w_new
    
    def __append_steps(self, T_center_w: pin.SE3, side: Side):
        """Append a step to the footstep plan
        
        Args:
            T_center_w: the pose of the center of the robot in the world frame
            side: the side of the foot

        Returns:
            None
        """
        T_foot_w = self.center_to_foot(T_center_w, side)
        self.steps.append(FootStep(T_foot_w, self.footprint_array, side))
        
    def plan_steps(self, T_first_w: pin.SE3, side: Side, no_steps: int, curvature: float, walking_start: bool = False, walking_stop: bool = False):
        """plan a sequence of steps in a line following a specified curvature.

        Args:
            T_first_w: the pose of the first step in the world frame
            side: the side of the first step
            no_steps: number of steps to be planned in the line (including the first step)
            curvature: the curvature to be followed in the line
            walking_start: if the robot is walking from a standstill to have a extra parallel step at the beginning
            walking_stop: if the robot is stopping to the standstill to have a extra parallel step at the end
        
        Returns:
            steps( list(FootStep) ): a list of FootStep
        """
        T_center_w = pin.SE3.Identity()
        if side == Side.RIGHT:                         # Notice: this is the opposite of the center_to_foot function
            T_center_w.translation = T_first_w.translation + T_first_w.rotation @ np.array([0, self.step_size_y, 0])
        else:
            T_center_w.translation = T_first_w.translation + T_first_w.rotation @ np.array([0, -self.step_size_y, 0])
        T_center_w.rotation = T_first_w.rotation

        self.steps = []
        self.steps.append(FootStep(T_first_w, self.footprint_array, side))
        side = other_foot_id(side)

        if walking_start:
            self.__append_steps(T_center_w, side)
            side = other_foot_id(side)

        for i in range(1, no_steps):
            T_center_w = self.rotate_with_center(T_center_w, curvature)
            self.__append_steps(T_center_w, side)
            side = other_foot_id(side)

        if walking_stop:
            self.__append_steps(T_center_w, side)
            side = other_foot_id(side)
                                    
        return self.steps

    def plot(self, simulation):
        """plot all the planned footsteps in the pybullet simulation
        Args:
            simulation (PybulletWrapper())
            
        Returns:
            plot of lines of planned footsteps in pybullet simulation (For initial debugging, deprecated)
        """
        if not self.deprecated_plot_warning:
            print("[ Warning ]: The plot function is deprecated. Please don't use it in production code.")
            self.deprecated_plot_warning = True
        for i in range(len(self.steps)):
            self.steps[i].plot(simulation, i)
            
if __name__=='__main__':
    """ 
    Visualization test of footstep planner in pybullet.
    """
    
    # Generate a plan and plot it in pybullet.
    # Check that the plan looks as expected
    simulator = PybulletWrapper()
    foot_planner = FootStepPlanner()
    start_pose = pin.SE3.Identity()

    # independent curve test trajectory
    steps1 = foot_planner.plan_steps(start_pose, Side.RIGHT, 5, 0, walking_start = True, walking_stop = True)
    foot_planner.plot(simulator)

    start_pose.translation += np.array([0, 2, 0])
    steps2 = foot_planner.plan_steps(start_pose, Side.RIGHT, 10, 0.5, walking_start = True, walking_stop = False)
    foot_planner.plot(simulator)

    start_pose.translation += np.array([0, 2, 0])
    steps3 = foot_planner.plan_steps(start_pose, Side.RIGHT, 10, -0.5, walking_start = False, walking_stop = True)
    foot_planner.plot(simulator)

    start_pose.translation += np.array([0, 2, 0])
    steps3 = foot_planner.plan_steps(start_pose, Side.RIGHT, 40, 1, walking_start = False, walking_stop = False)
    foot_planner.plot(simulator)

    # a continuous curve test trajectory
    start_pose = pin.SE3.Identity()
    start_pose.translation += np.array([2, 0, 0])
    steps_c1 = foot_planner.plan_steps(start_pose, Side.RIGHT, 5, 0, walking_start = True, walking_stop = False)
    foot_planner.plot(simulator)

    last_step = steps_c1[-1]
    step_c2 = foot_planner.plan_steps(last_step.get_pose(), last_step.side, 5, 1, walking_start = False, walking_stop = False)
    foot_planner.plot(simulator)

    last_step = step_c2[-1]
    step_c3 = foot_planner.plan_steps(last_step.get_pose(), last_step.side, 5, -1, walking_start = False, walking_stop = False)
    foot_planner.plot(simulator)

    last_step = step_c3[-1]
    step_c4 = foot_planner.plan_steps(last_step.get_pose(), last_step.side, 8, 2, walking_start = False, walking_stop = True)
    foot_planner.plot(simulator)

    while True:
        pb.stepSimulation()