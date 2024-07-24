#!/usr/bin/env python3
"""
A debugger that records the result of the simulation
"""

import numpy as np
import pybullet as pb

class Debugger():
    def __init__(self):
        self.time = []

        self.com_pin = []
        self.com_tsid = []
        self.com_ref = []
        self.com_ankle = []

        self.com_pin_vel = []
        self.com_tsid_vel = []
        self.com_ref_vel = []
        self.com_ankle_vel = []
        
        self.com_pin_acc = []
        self.com_tsid_acc = []
        self.com_ref_acc = []

        self.ZMP_plan = []
        self.ZMP_est = []
        self.DCM_est = []

        self.lf_pos = []
        self.lf_vel = []
        self.lf_acc = []

        self.rf_pos = []
        self.rf_vel = []
        self.rf_acc = []
        
        self.lf_normal = []
        self.rf_normal = []

        self.angularMoment = []
        self.refMoment = []
    
    def record(self, robot, interpolator, dt, u_k, com_ankle_desire, com_ankle_desire_dot):
        if interpolator is not None:
            c, c_dot, c_ddot = interpolator.comState()
        else:
            c, c_dot, c_ddot = np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0])
        self.com_ref.append(c)
        self.com_ref_vel.append(c_dot)
        self.com_ref_acc.append(c_ddot)
        self.com_ankle.append(com_ankle_desire)
        self.com_ankle_vel.append(com_ankle_desire_dot)
        self.com_tsid.append(robot.stack.comState().value())
        self.com_tsid_vel.append(robot.stack.comState().derivative())
        try:
            acc = (self.com_tsid_vel[-1] - self.com_tsid_vel[-2])/dt
            self.com_tsid_acc.append(acc)
        except:
            self.com_tsid_acc.append([0,0,0])        
        self.com_pin.append(robot.robot.baseCoMPosition())
        self.com_pin_vel.append(robot.robot.baseCoMVelocity())        
        try:
            acc = (self.com_pin_vel[-1] - self.com_pin_vel[-2])/dt
            self.com_pin_acc.append(acc)
        except:
            self.com_pin_acc.append([0,0,0])

        if len(self.time) >= 1:
            self.time.append(self.time[-1] + dt)
        else:
            self.time.append(dt)

        self.ZMP_plan.append(u_k)
        self.ZMP_est.append(robot.zmp)
        self.DCM_est.append(robot.dcm)

        self.lf_pos.append(robot.stack.get_LF_3d_pos_vel_acc(robot.dq_sol)[0])
        self.lf_vel.append(robot.stack.get_LF_3d_pos_vel_acc(robot.dq_sol)[1])
        self.lf_acc.append(robot.stack.get_LF_3d_pos_vel_acc(robot.dq_sol)[2])

        self.rf_pos.append(robot.stack.get_RF_3d_pos_vel_acc(robot.dq_sol)[0])
        self.rf_vel.append(robot.stack.get_RF_3d_pos_vel_acc(robot.dq_sol)[1])
        self.rf_acc.append(robot.stack.get_RF_3d_pos_vel_acc(robot.dq_sol)[2])

        wrench_left = pb.getJointState(robot.robot.id(), robot.robot.jointNameIndexMap()['leg_left_6_joint'])[2]
        wrench_right = pb.getJointState(robot.robot.id(), robot.robot.jointNameIndexMap()['leg_right_6_joint'])[2]

        self.lf_normal.append(wrench_left[0:3])
        self.rf_normal.append(wrench_right[0:3])

        self.angularMoment.append(robot.stack.amTask.momentum)
        self.refMoment.append(robot.stack.amTask.momentum_ref)

    def plot(self):
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-dark')
        
        self.com_pin = np.array(self.com_pin).T
        self.com_tsid = np.array(self.com_tsid).T
        self.com_ref = np.array(self.com_ref).T
        self.com_ankle = np.array(self.com_ankle).T
        
        self.com_pin_vel = np.array(self.com_pin_vel).T
        self.com_tsid_vel = np.array(self.com_tsid_vel).T
        self.com_ref_vel = np.array(self.com_ref_vel).T
        self.com_ankle_vel = np.array(self.com_ankle_vel).T

        self.com_pin_acc = np.array(self.com_pin_acc).T
        self.com_tsid_acc = np.array(self.com_tsid_acc).T
        self.com_ref_acc = np.array(self.com_ref_acc).T

        #print(com_pin.shape, com_pb.shape, com_ref.shape)

        # plot everything

        #################################################
        #                 CoM Position                  #
        #################################################
        figure_pos, ax_pos = plt.subplots(3, 1)
        
        ax_pos[0].plot(self.time, np.array(self.com_pin)[0][:], label='COM PIN 0')
        ax_pos[0].plot(self.time, np.array(self.com_tsid)[0][:], label='COM tsid 0')
        ax_pos[0].plot(self.time, np.array(self.com_ref)[0][:], linestyle=':', label='COM REF 0')
        ax_pos[0].plot(self.time, np.array(self.com_ankle)[0][:], linestyle='--', label='COM Ankle 0')

        ax_pos[1].plot(self.time, np.array(self.com_pin)[1][:], label='COM PIN 1')
        ax_pos[1].plot(self.time, np.array(self.com_tsid)[1][:], label='COM tsid 1')
        ax_pos[1].plot(self.time, np.array(self.com_ref)[1][:], linestyle=':', label='COM REF 1')
        ax_pos[1].plot(self.time, np.array(self.com_ankle)[1][:], linestyle='--', label='COM Ankle 1')

        ax_pos[2].plot(self.time, np.array(self.com_pin)[2][:], label='COM PIN 2')
        ax_pos[2].plot(self.time, np.array(self.com_tsid)[2][:], label='COM tsid 2')
        ax_pos[2].plot(self.time, np.array(self.com_ref)[2][:], linestyle=':', label='COM REF 2')
        ax_pos[2].plot(self.time, np.array(self.com_ankle)[2][:], linestyle='--', label='COM Ankle 2')

        ax_pos[0].set_ylabel('CoM Position x [m]')
        ax_pos[1].set_ylabel('CoM Position y [m]')
        ax_pos[2].set_ylabel('CoM Position z [m]')

        for ax in ax_pos:
            ax.set_xlabel('Time [s]')
            ax.legend(loc='upper right')
            ax.grid(True)


        #################################################
        #                 CoM Velocity                  #
        #################################################
        figure_vel, ax_vel = plt.subplots(3, 1)
        ax_vel[0].plot(self.time, np.array(self.com_pin_vel)[0][:], label='COM PIN 0')
        ax_vel[0].plot(self.time, np.array(self.com_tsid_vel)[0][:], label='COM tsid 0')
        ax_vel[0].plot(self.time, np.array(self.com_ref_vel)[0][:], linestyle=':', label='COM REF 0')
        ax_vel[0].plot(self.time, np.array(self.com_ankle_vel)[0][:], linestyle='--', label='COM Ankle 0')

        ax_vel[1].plot(self.time, np.array(self.com_pin_vel)[1][:], label='COM PIN 1')
        ax_vel[1].plot(self.time, np.array(self.com_tsid_vel)[1][:], label='COM tsid 1')
        ax_vel[1].plot(self.time, np.array(self.com_ref_vel)[1][:], linestyle=':', label='COM REF 1')
        ax_vel[1].plot(self.time, np.array(self.com_ankle_vel)[1][:], linestyle='--', label='COM Ankle 1')
        
        ax_vel[2].plot(self.time, np.array(self.com_pin_vel)[2][:], label='COM PIN 2')
        ax_vel[2].plot(self.time, np.array(self.com_tsid_vel)[2][:], label='COM tsid 2')
        ax_vel[2].plot(self.time, np.array(self.com_ref_vel)[2][:], linestyle=':', label='COM REF 2')
        ax_vel[2].plot(self.time, np.array(self.com_ankle_vel)[2][:], linestyle='--', label='COM Ankle 2')

        ax_vel[0].set_ylabel('CoM Velocity x [m/s]')
        ax_vel[1].set_ylabel('CoM Velocity y [m/s]')
        ax_vel[2].set_ylabel('CoM Velocity z [m/s]')

        for ax in ax_vel:
            ax.set_xlabel('Time [s]')
            ax.legend(loc='upper right')
            ax.grid(True)


        #################################################
        #               CoM Acceleration                #
        #################################################
        figure_acc, ax_acc = plt.subplots(3, 1)

        ax_acc[0].plot(self.time, np.array(self.com_pin_acc)[0][:], label='COM PIN 0')
        ax_acc[0].plot(self.time, np.array(self.com_tsid_acc)[0][:], label='COM tsid 0')
        ax_acc[0].plot(self.time, np.array(self.com_ref_acc)[0][:], linestyle=':', label='COM REF 0')

        ax_acc[1].plot(self.time, np.array(self.com_pin_acc)[1][:], label='COM PIN 1')
        ax_acc[1].plot(self.time, np.array(self.com_tsid_acc)[1][:], label='COM tsid 1')
        ax_acc[1].plot(self.time, np.array(self.com_ref_acc)[1][:], linestyle=':', label='COM REF 1')
        
        ax_acc[2].plot(self.time, np.array(self.com_pin_acc)[2][:], label='COM PIN 2')
        ax_acc[2].plot(self.time, np.array(self.com_tsid_acc)[2][:], label='COM tsid 2')
        ax_acc[2].plot(self.time, np.array(self.com_ref_acc)[2][:], linestyle=':', label='COM REF 2')

        ax_acc[0].set_ylabel('CoM Acceleration x [m/s]')
        ax_acc[1].set_ylabel('CoM Acceleration y [m/s]')
        ax_acc[2].set_ylabel('CoM Acceleration z [m/s]')

        for ax in ax_acc:
            ax.set_xlabel('Time [s]')
            ax.legend(loc='upper right')
            ax.grid(True)





        #################################################
        #                    ZMP, DCM                   #
        #################################################
        figure_ZMP, ax_ZMP = plt.subplots(2, 1)
        ax_ZMP[0].plot(self.time, np.array(self.ZMP_plan)[:,0], linestyle=':', label='ZMP Ref 0')
        ax_ZMP[0].plot(self.time, np.array(self.ZMP_est)[:,0] , label='ZMP PB 0')

        ax_ZMP[1].plot(self.time, np.array(self.ZMP_plan)[:,1], linestyle=':', label='ZMP Ref 1')
        ax_ZMP[1].plot(self.time, np.array(self.ZMP_est)[:,1], label='ZMP PB 1')
        
        ax_ZMP[0].set_ylabel('ZMP x [m]')
        ax_ZMP[1].set_ylabel('ZMP y [m]')
        for ax in ax_ZMP:
            ax.set_xlabel('Time [s]')
            ax.legend(loc='upper right')
            ax.grid(True)


        figure_DCM, ax_DCM = plt.subplots(2, 1)
        ax_DCM[0].plot(self.time, np.array(self.DCM_est)[:,0], label='DCM_est X')
        ax_DCM[1].plot(self.time, np.array(self.DCM_est)[:,1], label='DCM_est Y')

        ax_DCM[0].set_ylabel('DCM x [m]')
        ax_DCM[1].set_ylabel('DCM y [m]')
        for ax in ax_DCM:
            ax.set_xlabel('Time [s]')
            ax.legend(loc='upper right')
            ax.grid(True)


        #################################################
        #                 Normal Forces                 #
        #################################################
        figure_force_normal, ax_force_normal = plt.subplots(3, 1)
        ax_force_normal[0].plot(self.time, np.array(self.lf_normal)[:,0], label='lf normal X')
        ax_force_normal[1].plot(self.time, np.array(self.lf_normal)[:,1], label='lf normal Y')
        ax_force_normal[2].plot(self.time, np.array(self.lf_normal)[:,2], label='lf normal Z')

        ax_force_normal[0].plot(self.time, np.array(self.rf_normal)[:,0], label='rf normal X')
        ax_force_normal[1].plot(self.time, np.array(self.rf_normal)[:,1], label='rf normal Y')
        ax_force_normal[2].plot(self.time, np.array(self.rf_normal)[:,2], label='rf normal Z')

        ax_force_normal[0].set_ylabel('normal forces feet x [m]')
        ax_force_normal[1].set_ylabel('normal forces feet y [m]')
        ax_force_normal[2].set_ylabel('normal forces feet z [m]')
        for ax in ax_force_normal:
            ax.set_xlabel('Time [s]')
            ax.legend(loc='upper right')
            ax.grid(True)


        #################################################
        #               Foot Pos, Vel, Acc              #
        #################################################
        figure_feet_pos, ax_foot_pos = plt.subplots(3, 1)
        ax_foot_pos[0].plot(self.time, np.array(self.lf_pos)[:,0], label='Left Foot Pos X')
        ax_foot_pos[1].plot(self.time, np.array(self.lf_pos)[:,1], label='Left Foot Pos Y')
        ax_foot_pos[2].plot(self.time, np.array(self.lf_pos)[:,2], label='Left Foot Pos Z')
        ax_foot_pos[0].plot(self.time, np.array(self.rf_pos)[:,0], label='Right Foot Pos X')
        ax_foot_pos[1].plot(self.time, np.array(self.rf_pos)[:,1], label='Right Foot Pos Y')
        ax_foot_pos[2].plot(self.time, np.array(self.rf_pos)[:,2], label='Right Foot Pos Z')
        for ax in ax_foot_pos:
            ax.set_xlabel('Time [s]')
            ax.legend(loc='upper right')
            ax.grid(True)

        figure_feet_vel, ax_foot_vel = plt.subplots(3, 1)
        ax_foot_vel[0].plot(self.time, np.array(self.lf_vel)[:,0], label='Left Foot Vel X')
        ax_foot_vel[1].plot(self.time, np.array(self.lf_vel)[:,1], label='Left Foot Vel Y')
        ax_foot_vel[2].plot(self.time, np.array(self.lf_vel)[:,2], label='Left Foot Vel Z')
        ax_foot_vel[0].plot(self.time, np.array(self.rf_vel)[:,0], label='Right Foot Vel X')
        ax_foot_vel[1].plot(self.time, np.array(self.rf_vel)[:,1], label='Right Foot Vel Y')
        ax_foot_vel[2].plot(self.time, np.array(self.rf_vel)[:,2], label='Right Foot Vel Z')
        for ax in ax_foot_vel:
            ax.set_xlabel('Time [s]')
            ax.legend(loc='upper right')
            ax.grid(True)

        figure_feet_acc, ax_foot_acc = plt.subplots(3, 1)
        ax_foot_acc[0].plot(self.time, np.array(self.lf_acc)[:,0], label='Left Foot Acc X')
        ax_foot_acc[1].plot(self.time, np.array(self.lf_acc)[:,1], label='Left Foot Acc Y')
        ax_foot_acc[2].plot(self.time, np.array(self.lf_acc)[:,2], label='Left Foot Acc Z')
        ax_foot_acc[0].plot(self.time, np.array(self.rf_acc)[:,0], label='Right Foot Acc X')
        ax_foot_acc[1].plot(self.time, np.array(self.rf_acc)[:,1], label='Right Foot Acc Y')
        ax_foot_acc[2].plot(self.time, np.array(self.rf_acc)[:,2], label='Right Foot Acc Z')
        for ax in ax_foot_acc:
            ax.set_xlabel('Time [s]')
            ax.legend(loc='upper right')
            ax.grid(True)

        

        #################################################
        #                 Angular Moment                #
        #################################################
        figure_angular, ax_angular = plt.subplots(3, 1)

        ax_angular[0].plot(self.time, np.array(self.angularMoment)[:,0], label='Angular_Moment X')
        ax_angular[1].plot(self.time, np.array(self.angularMoment)[:,1], label='Angular_Moment Y')
        ax_angular[2].plot(self.time, np.array(self.angularMoment)[:,2], label='Angular_Moment Z')

        ax_angular[0].plot(self.time, np.array(self.refMoment)[:,0], label='Ref Angular_Moment X')
        ax_angular[1].plot(self.time, np.array(self.refMoment)[:,1], label='Ref Angular_Moment Y')
        ax_angular[2].plot(self.time, np.array(self.refMoment)[:,2], label='Ref Angular_Moment Z')

        ax_angular[0].set_ylabel('momentum x')
        ax_angular[1].set_ylabel('momentum y')
        ax_angular[2].set_ylabel('momentum z')

        for ax in ax_angular:
            ax.set_xlabel('Time [s]')
            ax.legend(loc='upper right')
            ax.grid(True)



        #################################################
        #                   Plot Show                   #
        #################################################
        plt.tight_layout()
        plt.show(block=False)