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

from pydrake.all import MathematicalProgram, Solve
from foot_plan.footstep_planner import Side, FootStep, FootStepPlanner

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
        c = np.array([self.x[0], self.x[2], self.conf.h])
        c_dot = np.array([self.x[1], self.x[3], 0])
        c_ddot = self.conf.g/self.conf.h * np.array([self.x[0] - self.u[0], 
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
        omega = np.sqrt(self.conf.g / self.conf.h)
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
        zmp = c - self.conf.h/self.conf.g * c_ddot
        return zmp
    
################################################################################
# LIPMPC
################################################################################

def generate_zmp_reference(foot_steps: list, no_samples_per_step: int):
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
        zmp_ref: ZMP reference over the horizon.
    """
    zmp_ref = []
    for foot_step_now in foot_steps:
        for _ in range(no_samples_per_step):
            zmp_ref.append(foot_step_now.get_center_pose())
    return zmp_ref

def calculate_linear_constraint(half_footprint_x, half_footprint_y, frame):
    """
    Calculates the linear constraint for the ZMP. Outputs the linear constraint matrix and vector.

             lb <=   A * [x; y] <= ub
    
    Args:
        half_footprint_x (float): Half of the footprint size in x direction.
        half_footprint_y (float): Half of the footprint size in y direction.
        frame (pin.SE3): Frame of ZMP in compared with world frame.
    
    Returns:
        A (numpy.ndarray): Linear constraint matrix.
        ub (numpy.ndarray): Linear constraint vector upper bound.
        lb (numpy.ndarray): Linear constraint vector lower bound.
    """
    position = frame.translation
    rotation_x_y = frame.rotation[0:2, 0:2]
    A = rotation_x_y.T
    ub = np.array([position[0] + half_footprint_x, position[1] + half_footprint_y])
    lb = np.array([position[0] - half_footprint_x, position[1] - half_footprint_y])
    return A, ub, lb

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
        

    def buildSolveOCP(self, x_k: np.ndarray, ZMP_ref_k: list, terminal_idx: int):
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
        ZMP_ref_k_pos = []
        for ZMP_ref in ZMP_ref_k:
            x, y = ZMP_ref.translation[0], ZMP_ref.translation[1]
            ZMP_ref_k_pos.append([x, y])
        
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
            A, ub, lb = calculate_linear_constraint(self.conf.footprint[0]/2, self.conf.footprint[1]/2, ZMP_ref_k[i])
            prog.AddLinearConstraint(A, lb, ub, control[i, :])

        # 4. if terminal_idx < self.no_samples than we have the terminal state within
        # the current horizon. In this case create the terminal state (foot step pos + zero vel)
        # and apply the state constraint to all states >= terminal_idx within the horizon
        if terminal_idx < self.no_samples:
            for i in range(terminal_idx, self.no_samples):
                for j in range(nx):
                    if j == 1 or j == 3:  # Velocity components
                        prog.AddLinearEqualityConstraint(state[i, j] == 0)
                    else:                 # Position components
                        prog.AddLinearEqualityConstraint(state[i, j] == ZMP_ref_k_pos[terminal_idx][j//2])

    
        # setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
        for i in range(self.no_samples):
            # tracking cost
            zmp_error = control[i] - ZMP_ref_k_pos[i]
            tracking_cost = self.conf.alpha * zmp_error @ zmp_error
            prog.AddCost(tracking_cost)

            # smoothing cost
            com_velocity_x = state[i, 1]
            com_velocity_y = state[i, 3]
            com_velocity = np.array([com_velocity_x, com_velocity_y])
            smoothing_cost = self.conf.gamma * com_velocity @ com_velocity
            prog.AddCost(smoothing_cost)

            # do the final step
            # state_x = state[i,0]
            # state_y = state[i,2]
            # dist_to_final = np.array([state_x - ZMP_ref_k_pos[-1][0], state_y - ZMP_ref_k_pos[-1][1]])
            # final_cost = 0.02 * dist_to_final @ dist_to_final
            # prog.AddCost(final_cost)

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
    

if __name__ == "__main__":
    import pinocchio as pin
    import matplotlib.pyplot as plt
    from bullet_sims import talos_conf as conf
    foot_planner = FootStepPlanner(conf.step_size_x, conf.step_size_y, conf.lfxp, conf.lfxn, conf.lfyp, conf.lfyn)
    start_pose = pin.SE3.Identity()

    half_footprint_x = 0.1
    half_footprint_y = 0.065

    frame_test_1 = pin.SE3.Identity()
    frame_test_1.translation = np.array([0.1, 0.1, 0])
    frame_test_1.rotation = pin.rpy.rpyToMatrix(0, 0, 0.2)
    A, ub, lb = calculate_linear_constraint(half_footprint_x, half_footprint_y, frame_test_1)
    # use a grid to visualize if value is within the bounds
    x = np.linspace(-0.3, 0.3, 100)
    y = np.linspace(-0.3, 0.3, 100)
    X, Y = np.meshgrid(x, y)
    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z1[i, j] = np.dot(A, np.array([X[i, j], Y[i, j]]))[0]
            Z2[i, j] = np.dot(A, np.array([X[i, j], Y[i, j]]))[1]
    fig, ax = plt.subplots()
    ax.contour(X, Y, Z1, levels=[lb[0], ub[0]], colors='r')
    ax.contour(X, Y, Z2, levels=[lb[1], ub[1]], colors='b')
    Z3 = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if lb[0] <= Z1[i, j] <= ub[0] and lb[1] <= Z2[i, j] <= ub[1]:
                Z3[i, j] = 1
    ax.contourf(X, Y, Z3, levels=[0.9, 1.1], colors=['green','white'], alpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

    # steps2 = foot_planner.plan_steps(start_pose, Side.RIGHT, 10, 0.5, walking_start = True, walking_stop = False)
    # mpc = LIPMPC(conf)
    # x_k = np.array([0, 0, 0, 0])
    # ZMP_ref_k = []
    # ZMP_real_k = []
    # for step in steps2:
    #     for _ in range(conf.no_mpc_samples_per_step):
    #         ZMP_ref_k.append(step.pose)
    # for i in range(10):
    #     for j in range(conf.no_mpc_samples_per_step):
    #         u = mpc.buildSolveOCP(x_k, ZMP_ref_k, 1000)
    #         ZMP_real_k.append(u)

    # plot the ZMP reference and the calculated ZMP