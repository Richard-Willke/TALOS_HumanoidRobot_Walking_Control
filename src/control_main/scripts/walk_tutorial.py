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
    
    # setup the simulator
    # simulation
    simulator = PybulletWrapper()
        
    # setup the robot
    robot = Talos(simulator)
    
    # inital footsteps
    T_swing_w = robot.swingFootPose()       # set intial swing foot pose to left foot
    T_support_w = robot.supportFootPose()   # set intial swing foot pose to right foot 
                               
    
    # setup the plan
    no_steps = 20
    planner = FootStepPlanner(conf)         # Create the planner
    plan = planner.planLine(T_support_w, Side.RIGHT, no_steps)      # Create the plan
    # Append the two last steps once more to the plan so our mpc horizon will never run out 
    # plan.append(plan[-2:])
    plan.extend(plan[-2:])
    foot_trajectory = SwingFootTrajectory(T_swing_w, plan[2].pose, conf.step_dur)


    # generate reference
    ZMP_ref = generate_zmp_reference(plan, conf.no_mpc_samples_per_step)      # Generate the mpc reference
    # plot the plan (make sure this workes first)
    planner.plot(simulator)
    
    # setup the lip models
    mpc = LIPMPC(conf)      # setup mpc
    
    # Assume the com is over the first support foot
    # Build the intial mpc state vector
    # x0 = [p_x, v_x, p_y, v_y]
    x0 = np.array([T_support_w.translation[0], 0, T_support_w.translation[1], 0])

    # Create the interpolator and set the inital state
    interpolator = LIPInterpolator(x0, conf)
    
    # set the com task over the support foot
    c, c_dot, c_ddot = interpolator.comState()
    # Set the COM reference to be over supporting foot 
    robot.stack.setComRefState(c, c_dot, c_ddot)

    
    ############################################################################
    # logging
    ############################################################################

    pre_dur = 3.0   # Time to wait befor walking should start
    N_pre = int(pre_dur / conf.dt)                      # number of sim steps before walking starts 
    N_sim = int(no_steps * conf.step_dur / conf.dt)     # total number of sim steps during walking
    N_mpc = int(N_sim / conf.no_sim_per_mpc)            # total number of mpc steps during walking
    
    # Create vectors to log all the data of the simulation
    # - COM_POS, COM_VEL, COM_ACC (from the planned reference, pinocchio and pybullet)
    # - Angular momentum (from pinocchio)
    # - Left and right foot POS, VEL, ACC (form planned reference, pinocchio) 
    # - ZMP (from planned reference, from estimator )
    # - DCM (from estimtor)
    # - Normal forces in right and left foot (from pybullet ft sensors, from pinocchio)
    TIME = np.nan*np.empty(N_sim)
    
    time = []

    com_pin = []
    com_pb = []
    com_ref = []

    com_pin_vel = []
    com_pb_vel = []
    com_ref_vel = []
    
    com_pin_acc = []
    com_pb_acc = []
    com_ref_acc = []

    ZMP_plan = []
    ZMP_est = []
    DCM_est = []

    lf_pos = []
    lf_vel = []
    lf_acc = []

    rf_pos = []
    rf_vel = []
    rf_acc = []
    
    lf_normal = []
    rf_normal = []

    angularMoment = []
    refMoment = []
    
    ############################################################################
    # logging
    ############################################################################
    
    k = 0                                               # current MPC index                          
    plan_idx = 1                                        # current index of the step within foot step plan
    t_step_elapsed = 0.0                                # elapsed time within current step (use to evaluate spline)
    t_publish = 0.0                                     # last publish time (last time we published something)
    
    for i in range(-N_pre, N_sim):
        t = simulator.simTime()     # simulator time
        dt = simulator.stepTime()   # simulator dt
        
        ########################################################################
        # update the mpc very no_sim_per_mpc steps
        ########################################################################
            
        # when to update mpc
        if i >= 0 and (i % conf.no_sim_per_mpc == 0):        
            # Implement MPC update
            x_k = interpolator.x
            terminal_idx = k + conf.no_mpc_samples_per_horizon*2
            ZMP_ref_k = ZMP_ref[k:k+terminal_idx]
            u_k = mpc.buildSolveOCP(x_k, ZMP_ref_k, terminal_idx)

            k += 1        

        ########################################################################
        # update the foot spline 
        ########################################################################


        # when to update spline
        if i >= 0 and (i % conf.no_sim_per_step == 0):
            # Start next step
            target_step = plan[plan_idx]
            robot.setSwingFoot(target_step.side)

            T_swing_pose_current = robot.swingFootPose()
            T_swing_pose_next = target_step.pose
            foot_trajectory.reset(T_swing_pose_current, T_swing_pose_next)

            t_step_elapsed = 0.0
            plan_idx += 1

            
        ########################################################################
        # in every iteration when walking
        ########################################################################
        
        if i >= 0:
            T_swing, V_swing, A_swing = foot_trajectory.evaluate(t_step_elapsed)
            robot.updateSwingFootRef(T_swing, V_swing, A_swing)
            interpolator.integrate(u_k)

            c, c_dot, c_ddot = interpolator.comState()
            robot.stack.setComRefState(c, c_dot, c_ddot)

            t_step_elapsed += dt

        ########################################################################
        # update the simulation
        ########################################################################

        # update the simulator and the robot
        simulator.step()
        robot.update()

        # publish to ros
        if t - t_publish > 1./30.:
            t_publish = t
            # publish
            # T_b_w = robot.robot.baseCoMPosition()
            robot.publish()
            
        # store for visualizations
        if i >= 0:
            TIME[i] = t
            # log information
            time.append(t)

            com_pin.append(robot.stack.comState().value())
            com_pb.append(robot.robot.baseCoMPosition())
            com_ref.append(robot.stack.comReference().value().copy())

            com_pin_vel.append(robot.stack.comState().derivative())
            com_pb_vel.append(robot.robot.baseCoMVelocity())
            com_ref_vel.append(robot.stack.comReference().derivative())

            com_pin_acc.append(robot.stack.comState().second_derivative())
            try:
                acc = (com_ref_vel[-1] - robot.robot.baseCoMVelocity())/dt
                com_pb_acc.append(acc)
            except:
                com_pb_acc([0,0,0])
            com_ref_acc.append(robot.stack.comReference().second_derivative())

            ZMP_plan.append(ZMP_ref[k])
            ZMP_est.append(robot.zmp)
            DCM_est.append(robot.dcm)

            lf_pos.append(robot.stack.get_LF_3d_pos_vel_acc(robot.dq_sol)[0])
            lf_vel.append(robot.stack.get_LF_3d_pos_vel_acc(robot.dq_sol)[1])
            lf_acc.append(robot.stack.get_LF_3d_pos_vel_acc(robot.dq_sol)[2])

            rf_pos.append(robot.stack.get_RF_3d_pos_vel_acc(robot.dq_sol)[0])
            rf_vel.append(robot.stack.get_RF_3d_pos_vel_acc(robot.dq_sol)[1])
            rf_acc.append(robot.stack.get_RF_3d_pos_vel_acc(robot.dq_sol)[2])

            wrench_left = pb.getJointState(robot.robot.id(), robot.robot.jointNameIndexMap()['leg_left_6_joint'])[2]
            wrench_right = pb.getJointState(robot.robot.id(), robot.robot.jointNameIndexMap()['leg_right_6_joint'])[2]

            lf_normal.append(wrench_left[0:3])
            rf_normal.append(wrench_right[0:3])

            angularMoment.append(robot.stack.amTask.momentum)
            refMoment.append(robot.stack.amTask.momentum_ref)

    ########################################################################
    # enough with the simulation, lets plot
    ########################################################################
    
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-dark')
    
    com_pin = np.array(com_pin).T
    com_pb = np.array(com_pb).T
    com_ref = np.array(com_ref).T
    
    com_pin_vel = np.array(com_pin_vel).T
    com_pb_vel = np.array(com_pb_vel).T
    com_ref_vel = np.array(com_ref_vel).T

    com_pin_acc = np.array(com_pin_acc).T
    com_pb_acc = np.array(com_pb_acc).T
    com_ref_acc = np.array(com_ref_acc).T

    #print(com_pin.shape, com_pb.shape, com_ref.shape)

    # plot everything

    #################################################
    #                 CoM Position                  #
    #################################################
    figure_pos, ax_pos = plt.subplots(3, 1)
    
    ax_pos[0].plot(time, np.array(com_pin)[0][:], label='COM PIN 0')
    ax_pos[0].plot(time, np.array(com_pb)[0][:], label='COM PB 0')
    ax_pos[0].plot(time, np.array(com_ref)[0][:], linestyle=':', label='COM REF 0')

    ax_pos[1].plot(time, np.array(com_pin)[1][:], label='COM PIN 1')
    ax_pos[1].plot(time, np.array(com_pb)[1][:], label='COM PB 1')
    ax_pos[1].plot(time, np.array(com_ref)[1][:], linestyle=':', label='COM REF 1')

    ax_pos[2].plot(time, np.array(com_pin)[2][:], label='COM PIN 2')
    ax_pos[2].plot(time, np.array(com_pb)[2][:], label='COM PB 2')
    ax_pos[2].plot(time, np.array(com_ref)[2][:], linestyle=':', label='COM REF 2')

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
    ax_vel[0].plot(time, np.array(com_pin_vel)[0][:], label='COM PIN 0')
    ax_vel[0].plot(time, np.array(com_pb_vel)[0][:], label='COM PB 0')
    ax_vel[0].plot(time, np.array(com_ref_vel)[0][:], linestyle=':', label='COM REF 0')

    ax_vel[1].plot(time, np.array(com_pin_vel)[1][:], label='COM PIN 1')
    ax_vel[1].plot(time, np.array(com_pb_vel)[1][:], label='COM PB 1')
    ax_vel[1].plot(time, np.array(com_ref_vel)[1][:], linestyle=':', label='COM REF 1')
    
    ax_vel[2].plot(time, np.array(com_pin_vel)[2][:], label='COM PIN 2')
    ax_vel[2].plot(time, np.array(com_pb_vel)[2][:], label='COM PB 2')
    ax_vel[2].plot(time, np.array(com_ref_vel)[2][:], linestyle=':', label='COM REF 2')

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

    ax_acc[0].plot(time, np.array(com_pin_acc)[0][:], label='COM PIN 0')
    ax_acc[0].plot(time, np.array(com_ref_acc)[0][:], linestyle=':', label='COM REF 0')

    ax_acc[1].plot(time, np.array(com_pin_acc)[1][:], label='COM PIN 1')
    ax_acc[1].plot(time, np.array(com_ref_acc)[1][:], linestyle=':', label='COM REF 1')
    
    ax_acc[2].plot(time, np.array(com_pin_acc)[2][:], label='COM PIN 2')
    ax_acc[2].plot(time, np.array(com_ref_acc)[2][:], linestyle=':', label='COM REF 2')

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
    ax_ZMP[0].plot(time, np.array(ZMP_plan)[:,0], linestyle=':', label='ZMP Ref 0')
    ax_ZMP[0].plot(time, - np.array(ZMP_est)[:,0] , label='ZMP PB 0')

    ax_ZMP[1].plot(time, np.array(ZMP_plan)[:,1], linestyle=':', label='ZMP Ref 1')
    ax_ZMP[1].plot(time, np.array(ZMP_est)[:,1], label='ZMP PB 1')
    
    ax_ZMP[0].set_ylabel('ZMP x [m]')
    ax_ZMP[1].set_ylabel('ZMP y [m]')
    for ax in ax_ZMP:
        ax.set_xlabel('Time [s]')
        ax.legend(loc='upper right')
        ax.grid(True)


    figure_DCM, ax_DCM = plt.subplots(2, 1)
    ax_DCM[0].plot(time, np.array(DCM_est)[:,0], label='DCM_est X')
    ax_DCM[1].plot(time, np.array(DCM_est)[:,1], label='DCM_est Y')

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
    ax_force_normal[0].plot(time, np.array(lf_normal)[:,0], label='lf normal X')
    ax_force_normal[1].plot(time, np.array(lf_normal)[:,1], label='lf normal Y')
    ax_force_normal[2].plot(time, np.array(lf_normal)[:,2], label='lf normal Z')

    ax_force_normal[0].plot(time, np.array(rf_normal)[:,0], label='rf normal X')
    ax_force_normal[1].plot(time, np.array(rf_normal)[:,1], label='rf normal Y')
    ax_force_normal[2].plot(time, np.array(rf_normal)[:,2], label='rf normal Z')

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
    ax_foot_pos[0].plot(time, np.array(lf_pos)[:,0], label='Left Foot Pos X')
    ax_foot_pos[1].plot(time, np.array(lf_pos)[:,1], label='Left Foot Pos Y')
    ax_foot_pos[2].plot(time, np.array(lf_pos)[:,2], label='Left Foot Pos Z')
    ax_foot_pos[0].plot(time, np.array(rf_pos)[:,0], label='Right Foot Pos X')
    ax_foot_pos[1].plot(time, np.array(rf_pos)[:,1], label='Right Foot Pos Y')
    ax_foot_pos[2].plot(time, np.array(rf_pos)[:,2], label='Right Foot Pos Z')
    for ax in ax_foot_pos:
        ax.set_xlabel('Time [s]')
        ax.legend(loc='upper right')
        ax.grid(True)

    figure_feet_vel, ax_foot_vel = plt.subplots(3, 1)
    ax_foot_vel[0].plot(time, np.array(lf_vel)[:,0], label='Left Foot Vel X')
    ax_foot_vel[1].plot(time, np.array(lf_vel)[:,1], label='Left Foot Vel Y')
    ax_foot_vel[2].plot(time, np.array(lf_vel)[:,2], label='Left Foot Vel Z')
    ax_foot_vel[0].plot(time, np.array(rf_vel)[:,0], label='Right Foot Vel X')
    ax_foot_vel[1].plot(time, np.array(rf_vel)[:,1], label='Right Foot Vel Y')
    ax_foot_vel[2].plot(time, np.array(rf_vel)[:,2], label='Right Foot Vel Z')
    for ax in ax_foot_vel:
        ax.set_xlabel('Time [s]')
        ax.legend(loc='upper right')
        ax.grid(True)

    figure_feet_acc, ax_foot_acc = plt.subplots(3, 1)
    ax_foot_acc[0].plot(time, np.array(lf_acc)[:,0], label='Left Foot Acc X')
    ax_foot_acc[1].plot(time, np.array(lf_acc)[:,1], label='Left Foot Acc Y')
    ax_foot_acc[2].plot(time, np.array(lf_acc)[:,2], label='Left Foot Acc Z')
    ax_foot_acc[0].plot(time, np.array(rf_acc)[:,0], label='Right Foot Acc X')
    ax_foot_acc[1].plot(time, np.array(rf_acc)[:,1], label='Right Foot Acc Y')
    ax_foot_acc[2].plot(time, np.array(rf_acc)[:,2], label='Right Foot Acc Z')
    for ax in ax_foot_acc:
        ax.set_xlabel('Time [s]')
        ax.legend(loc='upper right')
        ax.grid(True)

    

    #################################################
    #                 Angular Moment                #
    #################################################
    figure_angular, ax_angular = plt.subplots(3, 1)

    ax_angular[0].plot(time, np.array(angularMoment)[:,0], label='Angular_Moment X')
    ax_angular[1].plot(time, np.array(angularMoment)[:,1], label='Angular_Moment Y')
    ax_angular[2].plot(time, np.array(angularMoment)[:,2], label='Angular_Moment Z')

    ax_angular[0].plot(time, np.array(refMoment)[:,0], label='Ref Angular_Moment X')
    ax_angular[1].plot(time, np.array(refMoment)[:,1], label='Ref Angular_Moment Y')
    ax_angular[2].plot(time, np.array(refMoment)[:,2], label='Ref Angular_Moment Z')

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
    plt.show(figure_pos)
    plt.show(figure_vel)
    plt.show(figure_acc)
    plt.show(figure_ZMP)
    plt.show(figure_DCM)
    plt.show(figure_force_normal)
    plt.show(figure_feet_pos)
    plt.show(figure_feet_vel)
    plt.show(figure_feet_acc)
    plt.show(figure_angular)



if __name__ == '__main__': 
    rospy.init_node('talos_walking')
    main()
