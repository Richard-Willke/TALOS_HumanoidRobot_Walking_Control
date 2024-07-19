#!/usr/bin/env python3
import unittest
import numpy as np
from pydrake.all import MathematicalProgram, Solve
from scipy.linalg import expm
from scipy.integrate import quad

from zmp_plan.lip_mpc import continious_LIP_dynamics, discrete_LIP_dynamics, LIPInterpolator, LIPMPC, generate_zmp_reference
from bullet_sims import talos_conf as conf



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
        self.x_initial = np.array([0.0, 0.0, 0.0, 0.0])
        self.ZMP_ref_k = np.array([[0.1, 0.1] for _ in range(10)])
    

    def test_continious_LIP_dynamics(self):
        A, B = continious_LIP_dynamics(self.conf.g, self.conf.h)
        A_expected = np.array([[0, 1], [self.conf.g / self.conf.h, 0]])
        A_full_expected = np.zeros((4, 4))
        A_full_expected[:2, :2] = A_expected
        A_full_expected[2:, 2:] = A_expected
        B_expected = np.array([[0, 0], [-self.conf.g / self.conf.h, 0], [0, 0], [0, -self.conf.g / self.conf.h]])
        
        self.assertTrue(np.allclose(A, A_full_expected))
        self.assertTrue(np.allclose(B, B_expected))


    def test_discrete_LIP_dynamics(self):
        Ad, Bd = discrete_LIP_dynamics(self.conf.g, self.conf.h, self.conf.dt)
        omega = np.sqrt(self.conf.g / self.conf.h)
        
        A_continuous = np.array([[0, 1], [self.conf.g / self.conf.h, 0]])
        Ad_expected = expm(A_continuous * self.conf.dt)
        Bd_expected = np.array([[1 - np.cosh(omega * self.conf.dt)], [-omega * np.sinh(omega * self.conf.dt)]])
        
        self.assertTrue(np.allclose(Ad, Ad_expected))
        self.assertTrue(np.allclose(Bd, Bd_expected))


    def test_LIPInterpolator(self):
        interpolator = LIPInterpolator(self.x_initial, self.conf)
        u = np.array([0.1, 0.1])
        x_next = interpolator.integrate(u)
        self.assertEqual(x_next.shape, (4,))
        c, c_dot, c_ddot = interpolator.comState()
        self.assertEqual(c.shape, (3,))
        self.assertEqual(c_dot.shape, (3,))
        self.assertEqual(c_ddot.shape, (3,))
        # FIXME


    def test_LIPMPC(self):
        mpc = LIPMPC(self.conf)
        u = mpc.buildSolveOCP(self.x_initial, self.ZMP_ref_k, 5)
        self.assertEqual(u.shape, (2,))
    

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