#!/usr/bin/env python3
"""
A class that manage the MPC with a finite state machine
"""

import numpy as np
import pinocchio as pin
import pybullet as pb

#from foot_plan.footstep_planner import Side, FootStep, FootStepPlanner
#from simulator.pybullet_wrapper import PybulletWrapper
from zmp_plan.lip_mpc import LIPMPC

class MPC_FSM():
    """A class that manage the MPC with a finite state machine
    """
    def __init__(self, conf, foot_plan: list , terminal_index: int, mpc: LIPMPC):
        """Initialize the MPC FSM

        In case of non terminal footstep (default with conf.no_steps_per_horizon = 2):
                    [0]      [2]
                         [1]

            [0]       [1]          [2]
            T1 ... Tn Tn+1 ... T2n T2n+1 ... T3n
            |-----------------------|                1st call
                   |------------------------|        no_mpc_samples_per_step-th call
        
        In case of terminal footstep:
                    [0]  [2]
                         [1]

                                   terminal      hard-terminal
            [0]       [1]          [2]           [2]
            T1 ... Tn Tn+1 ... T2n T2n+1 ... T3n T3n+1 ... T4n
            |----------------------|                1st call
                               |--------------------------|        2 * no_mpc_samples_per_step-th call

        Args:
            conf: the configuration file
            foot_plan: the foot plan as a list of FootStep
            terminal_index: the index of the terminal footstep, 
                If larger than the length of foot_plan (sugggest to large than 1), then no terminal footstep is set.
                If some step is set as terminal, then a stopping condition (robot standing still) is considered.
            mpc: the MPC object (to prevent multiple initialization)
        
        Returns:
            MPC_FSM: the MPC FSM object
        """
        self.conf = conf
        self.foot_plan = list(foot_plan)
        
        self.no_mpc_samples_per_step = conf.no_mpc_samples_per_step                    # default is 8
        self.no_steps_per_horizon = conf.no_steps_per_horizon                          # default is 2
        self.length_horizon = self.no_mpc_samples_per_step * self.no_steps_per_horizon # default is 16                                               

        if self.no_steps_per_horizon > len(foot_plan) + 1:
            print("[ Warning ] MPC_FSM.__init__(): no_steps_per_horizon is larger than the length of foot_plan + 1.")
            print("    no_steps_per_horizon = ", self.no_steps_per_horizon)
            print("    len(foot_plan) = ", len(foot_plan))
            print(" No freedom to slide the horizon")

        if terminal_index > len(foot_plan):
            if not terminal_index == len(foot_plan) + 1:
                print("[ Warning ] MPC_FSM.__init__(): terminal_index is larger than the length of foot_plan + 1")
                print("    terminal_index = ", terminal_index)
                print("    len(foot_plan) = ", len(foot_plan))
                print(" When no terminal constraints, suggest terminal_index = foot_plan + 1.")
            self.terminal_horizon = self.length_horizon + self.no_mpc_samples_per_step

        else:
            terminal_ZMP = FootStep(foot_plan[-1].get_pose(), foot_plan[-1].footprint_array, foot_plan[-1].side)
            terminal_ZMP.translation = (foot_plan[-1].get_pose().translation + foot_plan[-1].get_pose().translation)/2
            self.foot_plan.append(terminal_ZMP)
            self.terminal_horizon = self.length_horizon + self.no_mpc_samples_per_step - 1
        
        self.ZMP_ref = []
        self.current_horizon = 0
        for step in self.foot_plan:
                for _ in range(conf.no_mpc_samples_per_step):
                    self.ZMP_ref.append(step.get_center_pose())

        self.mpc = mpc

    def next_horizon(self, x0: np.array):
        """Read the next horizon from the foot plan, and move horizon once

        __init__() -> next_horizon() -> next_horizon() -> ... -> next_horizon()

        T0 T1 T2 T3 ..... Tn
        |----------|           next_horizon()
           |----------|        next_horizon()
                ....
               |---------|     next_horizon() is_last=True

        Args:
            x0: Initial MPC state from iterpolator. Shape (4,). Represents [x, dx, y, dy].

        Returns:
            u_k: the calculated control input of MPC, which is the position of ZMP
            is_last: when the last horizon is reached, return True
        """
        if self.current_horizon + self.length_horizon >= len(self.ZMP_ref):
            raise ValueError("MPC_FSM.next_horizon(): The horizon is already at the end of the foot plan.")
        elif self.current_horizon + self.length_horizon == len(self.ZMP_ref) - 1:
            is_last = True
        else:
            is_last = False
        
        ZMP_ref_k = self.ZMP_ref[self.current_horizon:self.current_horizon + self.length_horizon]
        terminal_idx = self.terminal_horizon - self.current_horizon
        u_k = self.mpc.buildSolveOCP(x0, ZMP_ref_k, terminal_idx)

        self.current_horizon += 1

        return u_k, is_last
        

if __name__=='__main__':
    """ 
    Visualization test of gait_fsm in pybullet.
    """
    from foot_plan.footstep_planner import Side, FootStep, FootStepPlanner
    from bullet_sims import talos_conf as conf
    from zmp_plan.lip_mpc import LIPInterpolator
    import matplotlib.pyplot as plt

    start_pose = pin.SE3.Identity()
    start_step = FootStep(start_pose, np.array([conf.lfxp, conf.lfxn, conf.lfyp, conf.lfyn]), Side.RIGHT)

    footstep_planner = FootStepPlanner(conf.step_size_x, conf.step_size_y, conf.lfxp, conf.lfxn, conf.lfyp, conf.lfyn)
    mpc = LIPMPC(conf)
    x0 = np.array([start_step.get_center_pose().translation[0], 0, start_step.get_center_pose().translation[1], 0])
    interpolator = LIPInterpolator(x0, conf)
    
    ######### plan a footstep 1
    footstepsList1 = footstep_planner.plan_steps(start_step.get_pose(), start_step.side, 2, 0, walking_start=True, walking_stop=False)
    mpc_fsm = MPC_FSM(conf, footstepsList1, 4, mpc)

    u_k_all = []
    zmp_ref_all = []
    for T in mpc_fsm.ZMP_ref:
        zmp_ref_all.append(T.translation)
    com_all = []

    is_last = False
    while not is_last:
        x0 = interpolator.x
        u_k, is_last = mpc_fsm.next_horizon(x0)
        u_k_all.append(u_k)
        for _ in range(100):
            interpolator.integrate(u_k)
            com, _ , _ = interpolator.comState()
            com_all.append(com)

    # plot the result
    com_all = np.array(com_all)
    zmp_ref_all = np.array(zmp_ref_all)
    u_k_all = np.array(u_k_all)
    plt.plot(com_all[:, 0], com_all[:, 1], label='CoM')
    plt.scatter(zmp_ref_all[:, 0], zmp_ref_all[:, 1], label='ZMP ref')
    plt.scatter(u_k_all[:, 0], u_k_all[:, 1], label='ZMP')

    ########### plan a footstep 2 (continuation)
    start_step = footstepsList1[1]
    footstepsList2 = footstep_planner.plan_steps(start_step.get_pose(), start_step.side, 3, 0, walking_start=False, walking_stop=False)
    mpc_fsm = MPC_FSM(conf, footstepsList2, 4, mpc)

    u_k_all = []
    zmp_ref_all = []
    for T in mpc_fsm.ZMP_ref:
        zmp_ref_all.append(T.translation)
    com_all = []

    is_last = False
    while not is_last:
        x0 = interpolator.x
        u_k, is_last = mpc_fsm.next_horizon(x0)
        u_k_all.append(u_k)
        for _ in range(100):
            interpolator.integrate(u_k)
            com, _ , _ = interpolator.comState()
            com_all.append(com)

    # plot the result
    com_all = np.array(com_all)
    zmp_ref_all = np.array(zmp_ref_all)
    u_k_all = np.array(u_k_all)
    plt.plot(com_all[:, 0], com_all[:, 1], label='CoM 2')
    plt.scatter(zmp_ref_all[:, 0], zmp_ref_all[:, 1], label='ZMP ref 2')
    plt.scatter(u_k_all[:, 0], u_k_all[:, 1], label='ZMP 2')

    ########### plan a footstep 3 (continuation)
    start_step = footstepsList2[1]
    footstepsList3 = footstep_planner.plan_steps(start_step.get_pose(), start_step.side, 3, 0, walking_start=False, walking_stop=False)
    mpc_fsm = MPC_FSM(conf, footstepsList3, 4, mpc)

    u_k_all = []
    zmp_ref_all = []
    for T in mpc_fsm.ZMP_ref:
        zmp_ref_all.append(T.translation)
    com_all = []

    is_last = False
    while not is_last:
        x0 = interpolator.x
        u_k, is_last = mpc_fsm.next_horizon(x0)
        u_k_all.append(u_k)
        for _ in range(100):
            interpolator.integrate(u_k)
            com, _ , _ = interpolator.comState()
            com_all.append(com)

    # plot the result
    com_all = np.array(com_all)
    zmp_ref_all = np.array(zmp_ref_all)
    u_k_all = np.array(u_k_all)
    plt.plot(com_all[:, 0], com_all[:, 1], label='CoM 3')
    plt.scatter(zmp_ref_all[:, 0], zmp_ref_all[:, 1], label='ZMP ref 3')
    plt.scatter(u_k_all[:, 0], u_k_all[:, 1], label='ZMP 3')
    
    # plt.legend(loc='upper center') 
    plt.show(block=False)

    print("Test passed")      

    