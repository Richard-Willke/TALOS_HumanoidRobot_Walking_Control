"""
Provides various helper functions and classes for modeling and controlling the linear inverted pendulum (LIP) dynamics.

Functions:
    continious_LIP_dynamics: Returns the static matrices A, B of the continuous LIP dynamics.
    discrete_LIP_dynamics: Returns the matrices Ad, Bd of the discretized LIP dynamics.
    generate_zmp_reference: Generates the ZMP reference given a sequence of footsteps.

Classes:
    LIPInterpolator: Integrates the LIP model using continuous dynamics to interpolate the solution to height.
    LIPMPC: Builds and solves the optimal control problem (OCP) for the LIP model.
"""
import numpy as np
from typing import List

from pydrake.all import MathematicalProgram, Solve
from bullet_sims import talos_conf as conf
from foot_plan.footstep_planner import FootStep

################################################################################
# Helper fnc
################################################################################

def continious_LIP_dynamics(g, h):
    """
    Returns the static matrices A, B of the continuous LIP dynamics.

    Args:
        g (float): Gravitational constant.
        h (float): Height of the pendulum.

    Returns:
        tuple: Matrices A and B.
            - A (numpy.ndarray): Shape (4, 4)
            - B (numpy.ndarray): Shape (4, 2)

    """
    A_s = np.array([[0, 1],[g / h, 0]])
    A = np.zeros((4, 4))
    A[:2, :2] = A_s
    A[2:, 2:] = A_s

    B = np.array([[0,0], 
                  [-g / h,0],
                  [0,0], 
                  [0,-g / h]])
    return A, B


def discrete_LIP_dynamics(g, h, dt):
    """
    Returns the matrices Ad, Bd of the discretized LIP dynamics.

    Args:
        g (float): Gravitational constant.
        h (float): Height of the pendulum.
        dt (float): Time step for discretization.

    Returns:
        tuple: Matrices Ad and Bd.
            - Ad (numpy.ndarray): Shape (2, 2)
            - Bd (numpy.ndarray): Shape (2, 1)
    """
    omega = np.sqrt(g/h)
    Ad = np.array([[np.cosh(omega*dt),         1/omega*np.sinh(omega*dt)], 
                   [omega*np.sinh(omega*dt),   np.cosh(omega*dt)]])
    Bd = np.array([[1-np.cosh(omega*dt)], 
                   [-omega*np.sinh(omega*dt)]])
    return Ad, Bd

################################################################################
# LIPInterpolator
################################################################################

class LIPInterpolator:
    """
    Integrates the linear inverted pendulum model using the continuous dynamics.
    Used to interpolate the solution to height.

    System state:
        x = [x, dx, y, dy] represent the state: position and velocity of x and y of the CoM.
        u = [u_x, u_y] represent the control: position of x and y of ZMP

    Methods:
        integrate(u): Integrates the dynamics with control input u.
        comState(): Returns the center of mass (CoM) state.
        dcm(): Returns the divergent component of motion (DCM).
        zmp(): Returns the zero moment point (ZMP).
    """
    def __init__(self, x_inital: np.ndarray, conf):
        """initializes the LIP interpolator with the initial state and configuration parameters.

        Args:
            x_inital: Initial state vector.
            conf (module): Configuration parameters.

        Returns:
            LIPInterpolator object.
        """
        self.conf = conf
        self.dt = conf.dt
        self.x = x_inital
        self.A, self.B = continious_LIP_dynamics(conf.g, conf.h)
        self.A_d, self.B_d = discrete_LIP_dynamics(conf.g, conf.h, self.dt)
        self.u = np.array([x_inital[0], x_inital[2]])

    def integrate(self, u: np.ndarray):
        """
        Integrates the dynamics with control input u.

        Args:
            u : Control input vector. Shape (2,). Represents ZMP position [u_x, u_y].

        Returns:
            x (numpy.ndarray): Updated state vector. Shape (4,). Represents CoM position and velocity [x, dx, y, dy].
        """
        x_dot = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.x += x_dot * self.dt
        self.u = u
        return self.x
    
    def comState(self):
        """
        Returns the center of mass (CoM) state.

        Args:
            None

        Returns:
            tuple: CoM position, velocity, and acceleration.
                - c (numpy.ndarray): CoM position. Shape (3,)
                - c_dot (numpy.ndarray): CoM velocity. Shape (3,)
                - c_ddot (numpy.ndarray): CoM acceleration. Shape (3,)
        """
        c = np.array([self.x[0], self.x[2], conf.h])
        c_dot = np.array([self.x[1], self.x[3], 0])
        c_ddot = conf.g/conf.h * np.array([self.x[0] - self.u[0], 
                                           self.x[2] - self.u[1], 
                                           0])
        return c, c_dot, c_ddot
    
    def dcm(self):
        """
        Returns the divergent component of motion (DCM).

        Args:
            None

        Returns:
            numpy.ndarray: DCM vector. Shape (2,)
        """
        omega = np.sqrt(conf.g / conf.h)
        dcm = self.x[:2] + self.x[2:]/omega
        return dcm
    
    def zmp(self):
        """
        Returns the zero moment point (ZMP).

        Args:
            None

        Returns:
            numpy.ndarray: ZMP vector. Shape (3,)
        """
        c, _, c_ddot = self.comState()
        zmp = c - conf.h/conf.g * c_ddot
        return zmp
    
################################################################################
# LIPMPC
################################################################################

def generate_zmp_reference(foot_steps: List[FootStep], no_samples_per_step: int):
    """
    Generates the ZMP reference given a sequence of footsteps. There are a
    series of of foot steps (foot_steps), and we plan to stay on the foot step for a certain
    number of samples (no_samples_per_step).

    Therefore this function generates the ZMP reference over the horizon with
    respect to the sampling step.

    Args:
        foot_steps: Sequence of footstep objects.
        no_samples_per_step: Number of samples per step.

    Returns:
        list: ZMP reference over the horizon.
    """
    zmp_ref = []
    for foot_step_now in foot_steps:
        for _ in range(no_samples_per_step):
            x, y = foot_step_now.pose.translation[0], foot_step_now.pose.translation[1]
            zmp_ref.append([x, y])
    return zmp_ref

class LIPMPC:
    """
    Builds and solves the optimal control problem (OCP) for the LIP model.

    Args:
        conf (module): Configuration parameters.

    Methods:
        buildSolveOCP(x_k, ZMP_ref_k, terminal_idx): Builds and solves the OCP.
    """
    def __init__(self, conf):
        self.conf = conf
        self.dt = 0.1       
        self.no_samples = conf.no_mpc_samples_per_horizon

        self.X_k = None
        self.U_k = None
        self.ZMP_ref_k = None

        self.Ad, self.Bd = discrete_LIP_dynamics(conf.g, conf.h, self.dt)
        

    def buildSolveOCP(self, x_k: np.ndarray, ZMP_ref_k: np.ndarray, terminal_idx: int):
        """
        Builds and solves the optimal control problem (OCP).

        Args:
            x_k: Initial MPC state. Shape (4,). Represents [x, dx, y, dy].
            ZMP_ref_k: ZMP reference over the horizon. Shape (no_samples, 2)
            terminal_idx: Index within the horizon to apply terminal constraint.

        Returns:
            numpy.ndarray: Control vector. Shape (2,). Represents [u_x, u_y], the calculated ZMP position.
        """        
        nx = 4 
        nu = 2 
        prog = MathematicalProgram()
        
        state = prog.NewContinuousVariables(self.no_samples, nx, 'state')
        control = prog.NewContinuousVariables(self.no_samples, nu, 'control')
        
        # 1. intial constraint
        for i in range(nx):
            prog.AddLinearEqualityConstraint(state[0, i] == x_k[i])
    
        # 2. at each time step: respect the LIP descretized dynamics
        for i in range(self.no_samples - 1):
            x_next = self.Ad @ state[i, 0:2] + self.Bd @ control[i, 0:1]
            y_next = self.Ad @ state[i, 2:4] + self.Bd @ control[i, 1:2]

            for j in range(2):
                prog.AddLinearEqualityConstraint(state[i + 1, j] == x_next[j])
                prog.AddLinearEqualityConstraint(state[i + 1, j + 2] == y_next[j])

        # 3. at each time step: keep the ZMP within the foot sole (use the footprint and planned step position)
        for i in range(self.no_samples):
            prog.AddBoundingBoxConstraint(ZMP_ref_k[i] - conf.footprint, ZMP_ref_k[i] + conf.footprint, control[i])
            #TODO: Problematic here: ZMP constraint size error

        # 4. if terminal_idx < self.no_samples than we have the terminal state within
        # the current horizon. In this case create the terminal state (foot step pos + zero vel)
        # and apply the state constraint to all states >= terminal_idx within the horizon
        if terminal_idx < self.no_samples:
            for i in range(terminal_idx, self.no_samples):
                for j in range(nx):
                    if j == 1 or j == 3:  # Velocity components
                        prog.AddLinearEqualityConstraint(state[i, j] == 0)
                    else:                 # Position components
                        prog.AddLinearEqualityConstraint(state[i, j] == ZMP_ref_k[terminal_idx][j//2])

    
        # setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
        for i in range(self.no_samples):
            # tracking cost
            zmp_error = control[i] - ZMP_ref_k[i]
            tracking_cost = conf.alpha * zmp_error @ zmp_error
            prog.AddCost(tracking_cost)

            # smoothing cost
            com_velocity_x = state[i, 1]
            com_velocity_y = state[i, 3]
            com_velocity = np.array([com_velocity_x, com_velocity_y])
            smoothing_cost = conf.gamma * com_velocity @ com_velocity
            prog.AddCost(smoothing_cost)

        # solve
        result = Solve(prog)
        if not result.is_success:
            print("Solver failed.")
            print("Solution result:", result.get_solution_result())
            print("Decision variable values:", result.get_x_val())
            print("Constraint violations:", result.get_infeasible_constraints())
            return np.array([np.nan, np.nan])
            
        self.X_k = result.GetSolution(state)
        self.U_k = result.GetSolution(control)
        if np.isnan(self.X_k).any():
            print("failure")
            print("Solution contains NaN values.")
        
        self.ZMP_ref_k = ZMP_ref_k
        return self.U_k[0]