import numpy as np

from pydrake.all import MathematicalProgram, Solve
# >>>>TODO: replace the following to make it independent of the robot configuration
from bullet_sims import talos_conf as conf

################################################################################
# Helper fnc
################################################################################

def continious_LIP_dynamics(g, h):
    """returns the static matrices A,B of the continious LIP dynamics
    """
    #>>>>TODO: Compute
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
    """returns the matrices static Ad,Bd of the discretized LIP dynamics
    """
    #>>>>TODO: Compute
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
    """Integrates the linear inverted pendulum model using the 
    continous dynamics. To interpolate the solution to hight 
    """
    def __init__(self, x_inital, conf):
        self.conf = conf
        self.dt = conf.dt
        self.x = x_inital
        #>>>>TODO: Finish
        self.A, self.B = continious_LIP_dynamics(conf.g, conf.h)
        self.A_d, self.B_d = discrete_LIP_dynamics(conf.g, conf.h, self.dt)
        self.u = np.array([x_inital[0], x_inital[2]])
        
    def integrate(self, u):
        #>>>>TODO: integrate with dt
        x_dot = np.dot(self.A, self.x) + np.dot(self.B, u)
        # x_dot_x = np.dot(self.A, self.x[:2]) + np.dot(self.B, u[:1])
        # x_dot_y = np.dot(self.A, self.x[2:]) + np.dot(self.B, u[1:])
        # x_dot = np.concatenate([x_dot_x, x_dot_y])
        self.x += x_dot*self.dt
        self.u = u
        #print(x_dot)
        return self.x
    
    def comState(self):
        #>>>>TODO: return the center of mass state
        # that is position \in R3, velocity \in R3, acceleration \in R3
        # c, c_dot, c_ddot = None, None, None
        c = np.array([self.x[0], self.x[2], conf.h])
        c_dot = np.array([self.x[1], self.x[3], 0])
        # c_ddot = conf.g/conf.h * (c - np.vstack((self.u, np.array([0]))))
        c_ddot = conf.g/conf.h * np.array([self.x[0] - self.u[0], 
                                           self.x[2] - self.u[1], 
                                           0])
        return c, c_dot, c_ddot
    
    
    def dcm(self):
        #>>>>TODO: return the computed dcm
        # c, c_dot, _ = self.comState()
        omega = np.sqrt(conf.g / conf.h)
        # dcm = c + c_dot / omega
        dcm = self.x[:2] + self.x[2:]/omega
        return dcm
    
    def zmp(self):
        #>>>>TODO: return the zmp
        c, _, c_ddot = self.comState()
        zmp = c - conf.h/conf.g * c_ddot
        return zmp
        
    
################################################################################
# LIPMPC
################################################################################

class LIPMPC:
    def __init__(self, conf):
        self.conf = conf
        # self.dt = conf.dt 
        self.dt = 0.1       
        self.no_samples = conf.no_mpc_samples_per_horizon
        
        # solution and references over the horizon
        self.X_k = None
        self.U_k = None
        self.ZMP_ref_k = None

        self.Ad, self.Bd = discrete_LIP_dynamics(conf.g, conf.h, self.dt)
        
    def buildSolveOCP(self, x_k, ZMP_ref_k, terminal_idx):
        """build and solve ocp

        Args:
            x_k (_type_): inital mpc state
            ZMP_ref_k (_type_): zmp reference over horizon
            terminal_idx (_type_): index within horizon to apply terminal constraint

        Returns:
            _type_: control
        """
        
        #>>>>TODO: build and solve the ocp
        #>>>>Note: start without terminal constraints
        nx = 4 #>>>>TODO: State dimension = ?
        nu = 2 #>>>>TODO: control dimension = ?
        prog = MathematicalProgram()
        
        state = prog.NewContinuousVariables(self.no_samples, nx, 'state')
        control = prog.NewContinuousVariables(self.no_samples, nu, 'control')
        
        # 1. intial constraint
        #>>>>TODO: Add inital state constraint, Hint: x_k
        for i in range(nx):
            prog.AddLinearEqualityConstraint(state[0, i] == x_k[i])
    
        # 2. at each time step: respect the LIP descretized dynamics
        #>>>>TODO: Enforce the dynamics at every time step
        # for i in range(self.no_samples-1):
        #     prog.AddLinearConstraint(state[i+1] == self.Ad@state[i] + self.Bd@control[i])
        for i in range(self.no_samples - 1):
            x_next = self.Ad @ state[i, :2] + self.Bd @ control[i, :1]
            y_next = self.Ad @ state[i, 2:] + self.Bd @ control[i, 1:]

            for j in range(2):
                prog.AddLinearEqualityConstraint(state[i + 1, j] == x_next[j])
                prog.AddLinearEqualityConstraint(state[i + 1, j + 2] == y_next[j])

        # 3. at each time step: keep the ZMP within the foot sole (use the footprint and planned step position)
        #>>>>TODO: Add ZMP upper and lower bound to keep the control (ZMP) within each footprints
        #Hint: first compute upper and lower bound based on zmp_ref then add constraints.
        #Hint: Add constraints at every time step
        for i in range(self.no_samples):
            # print("ZMP_ref_k[i]: ", ZMP_ref_k[i])
            # print("conf.footprint: ", conf.footprint)
            # print("ZMP_ref_k[i] - conf.footprint: ", ZMP_ref_k[i] - conf.footprint)
            prog.AddBoundingBoxConstraint(ZMP_ref_k[i] - conf.footprint, ZMP_ref_k[i] + conf.footprint, control[i])

    
        # 4. if terminal_idx < self.no_samples than we have the terminal state within
        # the current horizon. In this case create the terminal state (foot step pos + zero vel)
        # and apply the state constraint to all states >= terminal_idx within the horizon
        #>>>>TODO: Add the terminal constraint if requires
        #Hint: If you are unsure, you can start testing without this first!
        
        #################################
        if terminal_idx < self.no_samples:
            for i in range(terminal_idx, self.no_samples):
                for j in range(nx):
                    if j == 1 or j == 3:  # Velocity components
                        prog.AddLinearEqualityConstraint(state[i, j] == 0)
                    else:  # Position components
                        prog.AddLinearEqualityConstraint(state[i, j] == ZMP_ref_k[terminal_idx][j//2])
        #############################

    
        # setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
        #>>>>TODO: add the cost at each timestep, hint: prog.AddCost
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
            print("failure")
            
        self.X_k = result.GetSolution(state)
        self.U_k = result.GetSolution(control)
        if np.isnan(self.X_k).any():
            print("failure")
        
        self.ZMP_ref_k = ZMP_ref_k
        return self.U_k[0]
    

def generate_zmp_reference(foot_steps, no_samples_per_step):
    """generate the zmp reference given a sequence of footsteps
    """

    #>>>>TODO: use the previously footstep type to build the reference 
    # trajectory for the zmp

    zmp_ref = []
    # for i, foot_step_now in enumerate(foot_steps):
    #     for j in range(no_samples_per_step):
    #         x, y = foot_step_now[0], foot_step_now[2]
    #         zmp_ref.append([x, y])
    # return zmp_ref
    for foot_step_now in foot_steps:
        for _ in range(no_samples_per_step):
            x, y = foot_step_now.pose.translation[0], foot_step_now.pose.translation[1]
            zmp_ref.append([x, y])
    return zmp_ref
