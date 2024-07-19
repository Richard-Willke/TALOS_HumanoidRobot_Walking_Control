#!/usr/bin/env python3
import unittest
import numpy as np
from scipy.linalg import expm
from scipy.signal import cont2discrete
from pydrake.all import MathematicalProgram, Solve
from scipy.linalg import block_diag


from zmp_plan.lip_mpc import continious_LIP_dynamics, discrete_LIP_dynamics, LIPInterpolator, LIPMPC, generate_zmp_reference
from bullet_sims import talos_conf as conf
from foot_plan.foot_trajectory import SwingFootTrajectory
from foot_plan.footstep_planner import Side, FootStepPlanner
from bullet_sims.talos import Talos
from simulator.robot import Robot
from simulator.pybullet_wrapper import PybulletWrapper




class TestLIPMPC(unittest.TestCase):
    
    def setUp(self):
        self.conf = conf
        self.conf.dt = 0.1
        self.conf.g = 9.81
        self.conf.h = 1.0
        self.conf.alpha = 1.0
        self.conf.gamma = 1.0
        self.conf.no_mpc_samples_per_horizon = 10
        self.conf.footprint = np.array([0.05, 0.05])
        self.x_initial = np.array([0.0, 0.0, -0.09, 0.0])
        
        simulator = PybulletWrapper()
        robot = Talos(simulator)
        T_support_w = robot.supportFootPose() 
        no_steps = 20
        planner = FootStepPlanner(conf) 
        plan = planner.planLine(T_support_w, Side.RIGHT, no_steps)
        plan.extend(plan[-2:])

        k = 0
        self.terminal_idx = k + conf.no_mpc_samples_per_horizon * 2
        ZMP_ref = generate_zmp_reference(plan, conf.no_mpc_samples_per_step)
        # self.ZMP_ref_k = np.array(ZMP_ref[k:k + self.terminal_idx])


    def test_continious_LIP_dynamics(self):
        A, B = continious_LIP_dynamics(self.conf.g, self.conf.h)

        A_single = np.array([[0, 1], [self.conf.g / self.conf.h, 0]])
        A_expected = block_diag(A_single, A_single)
        
        B_expected = np.array([[0, 0], [-self.conf.g / self.conf.h, 0], [0, 0], [0, -self.conf.g / self.conf.h]])
        
        self.assertTrue(np.allclose(A, A_expected))
        self.assertTrue(np.allclose(B, B_expected))
        

    def test_discrete_LIP_dynamics(self):
        Ad, Bd = discrete_LIP_dynamics(self.conf.g, self.conf.h, self.conf.dt)
        
        # A_continuous = np.array([[0, 1], [self.conf.g / self.conf.h, 0]])
        # B_continuous = np.array([[0], [1]])
        # system_continuous = (A_continuous, B_continuous, np.zeros((2, 1)), np.zeros((1, 1)))
        # dt = self.conf.dt
        # system_discrete = cont2discrete(system_continuous, dt)
        # Ad_expected, Bd_expected = system_discrete[0], system_discrete[1]

        # FIXME
        omega = np.sqrt(self.conf.g / self.conf.h)
        A_continuous = np.array([[0, 1], [self.conf.g / self.conf.h, 0]])
        Ad_expected = expm(A_continuous * self.conf.dt)
        Bd_expected = np.array([[1 - np.cosh(omega * self.conf.dt)], [-omega * np.sinh(omega * self.conf.dt)]])

        self.assertTrue(np.allclose(Ad, Ad_expected))
        self.assertTrue(np.allclose(Bd, Bd_expected))


    def test_LIPInterpolator(self):
        interpolator = LIPInterpolator(self.x_initial, self.conf)
        u = np.array([0.1, 0.1])
        
        # Principle
        A, B = continious_LIP_dynamics(self.conf.g, self.conf.h)
        x_expected = self.x_initial + (A @ self.x_initial + B @ u) * self.conf.dt
        x_next = interpolator.integrate(u)
        self.assertTrue(np.allclose(x_next, x_expected), msg="x_next values do not match the expected values.")

        c_expected = np.array([x_expected[0], x_expected[2], self.conf.h])
        c_dot_expected = np.array([x_expected[1], x_expected[3], 0])
        c_ddot_expected = self.conf.g / self.conf.h * np.array([x_expected[0] - u[0], x_expected[2] - u[1], 0])
        
        # Use Funciton
        c, c_dot, c_ddot = interpolator.comState()
        self.assertTrue(np.allclose(c, c_expected), msg="c values do not match the expected values.")
        self.assertTrue(np.allclose(c_dot, c_dot_expected), msg="c_dot values do not match the expected values.")
        self.assertTrue(np.allclose(c_ddot, c_ddot_expected), msg="c_ddot values do not match the expected values.")



    def test_LIPMPC(self):
        mpc = LIPMPC(self.conf)
        
        foot_steps = [
            type('FootStep', (object,), {'pose': type('Pose', (object,), {'translation': np.array([i * 0.2, (-1)**i * 0.1])})}) 
            for i in range(10)
        ]
        ZMP_ref_k = np.array(generate_zmp_reference(foot_steps, self.conf.no_mpc_samples_per_horizon))
        
        interpolator = LIPInterpolator(self.x_initial, self.conf)
        x_k = interpolator.x
        u = mpc.buildSolveOCP(x_k, ZMP_ref_k, self.terminal_idx)
        
        self.assertEqual(u.shape, (2,))
        
        u_lower_bound = ZMP_ref_k[0] - self.conf.footprint
        u_upper_bound = ZMP_ref_k[0] + self.conf.footprint
        
        self.assertTrue(np.all(u >= u_lower_bound) and np.all(u <= u_upper_bound), msg="Control output u is out of the expected range.")
        
        self.assertIsNotNone(mpc.X_k, msg="MPC optimization did not produce a solution for state trajectory X_k.")
        self.assertIsNotNone(mpc.U_k, msg="MPC optimization did not produce a solution for control trajectory U_k.")
        self.assertIsNotNone(mpc.ZMP_ref_k, msg="MPC optimization did not produce a ZMP reference trajectory.")
        
        self.assertEqual(mpc.X_k.shape, (self.conf.no_mpc_samples_per_horizon, 4), msg="State trajectory X_k has incorrect shape.")
        self.assertEqual(mpc.U_k.shape, (self.conf.no_mpc_samples_per_horizon, 2), msg="Control trajectory U_k has incorrect shape.")
        
        self.assertTrue(np.all(mpc.X_k >= -10) and np.all(mpc.X_k <= 10), msg="State trajectory X_k is out of the expected range [-10, 10].")
        self.assertTrue(np.all(mpc.U_k >= -1) and np.all(mpc.U_k <= 1), msg="Control trajectory U_k is out of the expected range [-1, 1].")
        
        # self.assertEqual(mpc.ZMP_ref_k.shape, (self.conf.no_mpc_samples_per_horizon, 2), msg="ZMP reference trajectory has incorrect shape.")
        for i in range(self.conf.no_mpc_samples_per_horizon):
            self.assertTrue(np.allclose(mpc.ZMP_ref_k[i], ZMP_ref_k[i], atol=1e-2), msg=f"ZMP reference at step {i} does not match the expected values.")



    def test_generate_zmp_reference(self):
        class FootStep:
            def __init__(self, x, y):
                self.pose = self.Pose(x, y)
            
            class Pose:
                def __init__(self, x, y):
                    self.translation = [x, y]
        
        foot_steps = [FootStep(0.1 * i, 0.1 * i) for i in range(10)]
        zmp_ref = generate_zmp_reference(foot_steps, self.conf.no_mpc_samples_per_horizon)
        self.assertEqual(len(zmp_ref), 10 * self.conf.no_mpc_samples_per_horizon)






if __name__ == '__main__':
    import rostest
    rostest.rosrun('zmp_plan', 'test_lip_mpc', TestLIPMPC)