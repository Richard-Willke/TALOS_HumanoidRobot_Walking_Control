"""Task2: Linear inverted pendulum MPC

The goal of this file is to formulate the optimal control problem (OCP)
in equation 12 but this time as a model predictive controller (MPC).

In this case we will solve the trajectory planning multiple times over 
a shorter horizon of just 2 steps (receding horizon).
Time between two MPC updates is called T_MPC.

In between MPC updates we simulate the Linear inverted pendulum at a smaller
step time T_SIM, with the lates MPC control ouput u.

Our state & control is the same as before
x = [cx, vx, cy, vy]
u = [px, py]

You will need to fill in the TODO to solve the task.
"""

import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import MathematicalProgram, Solve

import matplotlib.animation as animation

################################################################################
# settings
################################################################################

NO_STEPS                = 8         # total number of foot steps
STEP_TIME               = 0.8       # time needed for every step

# Robot Parameters:
# --------------
h                       = 0.80      # fixed CoM height (assuming walking on a flat terrain)
g                       = 9.81      # norm of the gravity vector
foot_length             = 0.10      # foot size in the x-direction
foot_width              = 0.06      # foot size in the y-direciton


# MPC Parameters:
# --------------
T_MPC                   = 0.1                                               # sampling time interval of the MPC
NO_MPC_SAMPLES_PER_STEP = int(round(STEP_TIME/T_MPC))                       # number of mpc updates per step

NO_STEPS_PER_HORIZON  = 2                                                   # how many steps in the horizon
T_HORIZON = NO_STEPS_PER_HORIZON*STEP_TIME                                  # duration of future horizon
NO_MPC_SAMPLES_HORIZON = int(round(NO_STEPS_PER_HORIZON*STEP_TIME/T_MPC))   # number of mpc updates per horizon

# Cost Parameters:
# ---------------
alpha       = 10**(-1)                                  # ZMP error squared cost weight (= tracking cost)
gamma       = 10**(-3)                                  # CoM velocity error squared cost weight (= smoothing cost)

# Simulation Parameters:
# --------------
T_SIM                   = 0.005                         # 200 Hz simulation time

NO_SIM_SAMPLES_PER_MPC = int(round(T_MPC/T_SIM))        # NO SIM samples between MPC updates
NO_MPC_SAMPLES = int(round(NO_STEPS*STEP_TIME/T_MPC))   # Total number of MPC samples
NO_SIM_SAMPLES = int(round(NO_STEPS*STEP_TIME/T_SIM))   # Total number of Simulator samples

################################################################################
# Helper fnc
################################################################################

def generate_foot_steps(foot_step_0, step_size_x, step_size_y, no_steps, turn_angle = 0.2):
    """Write a function that generates footstep of step size = step_size_x in the 
    x direction starting from foot_step_0 located at (x0, y0).
    
    Args:
        foot_step_0 (_type_): first footstep position (x0, y0)
        step_size_x (_type_): step size in x direction
        no_steps (_type_): number of steps to take
    """

    #>>>>TODO: generate the foot step plan with no_steps
    #>>>>Hint: Check the pdf Fig.3 for inspiration
    foot_steps = [[foot_step_0[0], foot_step_0[1]]]
    R = np.array([[np.cos(turn_angle), -np.sin(turn_angle)],
                  [np.sin(turn_angle), np.cos(turn_angle)]])
    for i in range(no_steps):
        R = R @ np.array([[np.cos(turn_angle), -np.sin(turn_angle)],[np.sin(turn_angle), np.cos(turn_angle)]])
        translation_step = R @ np.array([step_size_x, step_size_y*(-1)**(i+1)])
        foot_steps.append([foot_steps[-1][0] + translation_step[0], foot_steps[-1][1] + translation_step[1]])
    foot_steps = np.array(foot_steps) #None
    return foot_steps

def plot_foot_steps(foot_steps, XY_foot_print, ax, turn_angle = 0.2):
    """Write a function that plots footsteps in the xy plane using the given
    footprint (length, width)
    You can use the function ax.fill() to gerneate a colored rectanges.
    Color the left and right steps different and check if the step sequence makes sense.

    Args:
        foot_steps (_type_): the foot step plan
        XY_foot_print (_type_): the dimensions of the foot (x,y)
        ax (_type_): the axis to plot on
    """
    #>>>>TODO: Plot the the footsteps into ax 
    foot_steps = np.array(foot_steps).T
    #print(foot_steps)
    ax.plot(foot_steps[0],foot_steps[1])
    #a = np.array(foot_steps) @ XY_foot_print 
    #print(a)
    R = np.eye(2)
    for i, step in enumerate(foot_steps.T):
        if i < len(foot_steps.T) -2:
            R = R @ np.array([[np.cos(turn_angle), -np.sin(turn_angle)],[np.sin(turn_angle), np.cos(turn_angle)]])
        top_right = np.array([step[0],step[1]]) + R @ np.array([XY_foot_print[0], XY_foot_print[1]])
        bottom_right = np.array([step[0],step[1]]) + R @ np.array([- XY_foot_print[0],XY_foot_print[1]])
        bottom_left = np.array([step[0],step[1]]) + R @ np.array([ - XY_foot_print[0], - XY_foot_print[1]])
        top_left = np.array([step[0],step[1]]) + R @ np.array([ XY_foot_print[0], - XY_foot_print[1]])

        foot = np.array([top_right,
                         bottom_right,
                         bottom_left,
                         top_left]).T
        ax.fill(foot[0],foot[1])
    ax.set_aspect('equal')
    #plt.show()

def generate_zmp_reference(foot_steps, no_samples_per_step, turn_angle = 0.2):
    """generate a function that computes a referecne trajecotry for the ZMP
    (We need this for the tracking cost in the cost function of eq. 12)
    Remember: Our goal is to keep the ZMP at the footstep center within each step.
    So for the # of samples a step is active the zmp_ref should be at that step.
    
    Returns a vector of size (TOTAL_NO_SAMPLES, 2)

    Args:
        foot_steps (_type_): the foot step plan
        no_samples_per_step (_type_): number of sampes per step
    """

    #R = np.eye(2)
    #for step in foot_steps.T:
    #    R = R @ np.array([[np.cos(turn_angle), -np.sin(turn_angle)],[np.sin(turn_angle), np.cos(turn_angle)]])

    #>>>>TODO: Generate the ZMP reference based on given foot_steps
    zmp_ref = np.zeros((len(foot_steps)*no_samples_per_step,2))
    foot_side = []
    R = np.eye(2)
    R_ref = [R]
    for i in range(len(foot_steps)-2):
        R = R @ np.array([[np.cos(turn_angle), -np.sin(turn_angle)],[np.sin(turn_angle), np.cos(turn_angle)]])
        R_ref.append(R)
        foot_side.append(i%2)
    R_ref.append(R)
    R_ref.append(R)
    R_ref = np.array(R_ref)
    foot_side.append(foot_side[-1])
    foot_side.append(foot_side[-1])
    foot_side = np.array(foot_side)
    R_return = np.zeros((len(foot_steps)*no_samples_per_step,2,2))
    #for i in range(len(foot_steps)):
    for j in range(len(foot_steps)*no_samples_per_step):
        zmp_ref[j] = (foot_steps[int(j / no_samples_per_step)])
        R_return[j] = R_ref[int(j / no_samples_per_step)]
    zmp_ref = np.array(zmp_ref).T#None
    #print(zmp_ref)
    return zmp_ref, R_return, foot_side

################################################################################
# Dynamics of the simplified walking model
################################################################################

def continious_LIP_dynamics():
    """returns the matrices A,B of the continious LIP dynamics

    Args:
        g (_type_): gravity
        h (_type_): fixed height

    Returns:
        np.array: A, B
    """
    #>>>>TODO: Generate A, B for the continous linear inverted pendulum
    #>>>>Hint: Look at Eq. 4 and rewrite as a system first order diff. eq.
    A = np.array([[0, 1, 0, 0],
                  [g/h, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, g/h, 0]])
    B = np.array([[0, 0],
                  [-g/h, 0],
                  [0, 0],
                  [0, -g/h]])
    return A, B


def discrete_LIP_dynamics(delta_t):
    """returns the matrices static Ad,Bd of the discretized LIP dynamics

    Args:
        delta_t (_type_): discretization steps
        g (_type_): gravity
        h (_type_): height

    Returns:
        _type_: _description_
    """
    #>>>>TODO: Generate Ad, Bd for the discretized linear inverted pendulum
    omega = np.sqrt(g/h)
    Ad=np.array([[np.cosh(omega * delta_t), np.sinh(omega * delta_t)/omega],
                 [omega* np.sinh(omega*delta_t), np.cosh(omega * delta_t)]]) 
    Bd=np.array([[1 - np.cosh(omega * delta_t)],
                 [- omega * np.sinh(omega*delta_t)]])
    return Ad, Bd


################################################################################
# Simulation
################################################################################

class Simulator:
    """Simulates the Linear inverted pendulum continous dynamics
    Uses simple euler integration to simulate LIP at sample time dt
    """
    def __init__(self, x_inital, dt):
        self.dt = dt
        self.x = x_inital
        self.x = np.array(self.x).reshape(-1, 1)
        self.A, self.B = continious_LIP_dynamics()
        self.D = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])
        
    def simulate(self, u, d=np.zeros(2)):
        """updates the LIP state x using based on command u
        
        Optionally: Takes a disturbance acceleration d to simulate effect
        of external pushes on the LIP.
        """

        #>>>>TODO: Compute x_dot and use euler integration to approximate
        #self.x = np.array([[self.x[0],self.x[2]],
        #                   [self.x[1],self.x[3]]])
        u = np.array(u).reshape(-1, 1)
        d = np.array(d).reshape(-1, 1)
        #print(self.B@u)
        
        x_dot = self.A@self.x + self.B@u + self.D@d #None 
        # the state at t+dt
        #>>>>TODO: The disturbance is added in x_dot as self.D@d
        self.x = self.x + self.dt * x_dot
        return self.x    

################################################################################
# MPC
################################################################################

class MPC:
    """MPC for the Linear inverted pendulum
    """
    def __init__(self, dt, T_horizon):
        self.dt = dt                                        # mpc dt
        self.T_horizon = T_horizon                          # time of horizon
        self.no_samples = int(round(T_horizon/self.dt))     # mpc samples in horizon (nodes)

        self.Ad, self.Bd = discrete_LIP_dynamics(dt)
        
        self.X_k = None                                     # state over current horizon
        self.U_k = None                                     # control over current horizon
        self.ZMP_ref_k = None                               # ZMP reference over current horizon
        
    def buildSolveOCP(self, x_k, ZMP_ref_k, R_ref_k, foot_sides, terminal_idx):
        """ build the MathematicalProgram that solves the mpc problem and 
        returns the first command of U_k

        Args:
            x_k (_type_): the current state of the lip when starting the mpc
            ZMP_ref_k (_type_): the reference over the current horizon, shape=(no_samples, 2)
            terminal_idx (_type_): index of the terminal constraint within horizon (or bigger than horizon if no constraint)
            
        """
        
        # variables
        nx = 4 #None #>>>>TODO: State dimension = ?
        nu = 2 #None #>>>>TODO: control dimension = ?
        prog = MathematicalProgram()
        
        state = prog.NewContinuousVariables(self.no_samples, nx, 'state')
        control = prog.NewContinuousVariables(self.no_samples, nu, 'control')
        
        # 1. intial constraint
        #>>>>TODO: Add inital state constraint, Hint: x_k
        prog.AddConstraint(state[0,0] == x_k[0,0])
        prog.AddConstraint(state[0,1] == x_k[1,0])
        prog.AddConstraint(state[0,2] == x_k[2,0])
        prog.AddConstraint(state[0,3] == x_k[3,0])
    
        # 2. at each time step: respect the LIP descretized dynamics
        #>>>>TODO: Enforce the dynamics at every time step
        for i in range(NO_MPC_SAMPLES_HORIZON-1):
            x_next = self.Ad @ np.array([[state[i][0],state[i][2]],[state[i][1],state[i][3]]]) + self.Bd @ np.array([control[i]])

            prog.AddConstraint(state[i+1,0] == x_next[0,0]) # x
            prog.AddConstraint(state[i+1,1] == x_next[1,0]) # vx
            prog.AddConstraint(state[i+1,2] == x_next[0,1]) # y 
            prog.AddConstraint(state[i+1,3] == x_next[1,1]) # vy
        
        # 3. at each time step: keep the ZMP within the foot sole (use the footprint and planned step position)
        #>>>>TODO: Add ZMP upper and lower bound to keep the control (ZMP) within each footprints
        #Hint: first compute upper and lower bound based on zmp_ref then add constraints.
        #Hint: Add constraints at every time step
        self.ZMP_ref_k = ZMP_ref_k
        #self.ZMP_ref_k = generate_zmp_reference(foot_steps, NO_MPC_SAMPLES)

        ######################################################################
        #for i in range(NO_MPC_SAMPLES_HORIZON):
        #    ub_x = self.ZMP_ref_k[0][i] + foot_length/2
        #    lb_x = self.ZMP_ref_k[0][i] - foot_length/2
        #    ub_y = self.ZMP_ref_k[1][i] + foot_width/2
        #    lb_y = self.ZMP_ref_k[1][i] - foot_width/2
        #    prog.AddBoundingBoxConstraint(lb_x, ub_x, control[i][0])
        #    prog.AddBoundingBoxConstraint(lb_y, ub_y, control[i][1])
        
        
        v1 = None
        v2 = None
        v3 = None
        v4 = None

        for i in range(NO_MPC_SAMPLES_HORIZON-1):
            v1 = np.array([self.ZMP_ref_k[0][i], self.ZMP_ref_k[1][i] ]) + R_ref_k[i] @ np.array([-foot_length/2, -foot_width/2])
            v2 = np.array([self.ZMP_ref_k[0][i], self.ZMP_ref_k[1][i] ]) + R_ref_k[i] @ np.array([ foot_length/2, -foot_width/2])
            v3 = np.array([self.ZMP_ref_k[0][i], self.ZMP_ref_k[1][i] ]) + R_ref_k[i] @ np.array([ foot_length/2,  foot_width/2])
            v4 = np.array([self.ZMP_ref_k[0][i], self.ZMP_ref_k[1][i] ]) + R_ref_k[i] @ np.array([-foot_length/2,  foot_width/2])
            
            A = np.array([
                [v2[1] - v1[1], v1[0] - v2[0]],
                [v3[1] - v2[1], v2[0] - v3[0]],
                [v4[1] - v3[1], v3[0] - v4[0]],
                [v1[1] - v4[1], v4[0] - v1[0]]
            ])

            b = np.array([
                v2[1] * v1[0] - v2[0] * v1[1],
                v3[1] * v2[0] - v3[0] * v2[1],
                v4[1] * v3[0] - v4[0] * v3[1],
                v1[1] * v4[0] - v1[0] * v4[1]
            ])

            for j in range(4):
                prog.AddLinearConstraint(A[j, 0] * control[i][0] + A[j, 1] * control[i][1] <= b[j])

        ######################################################################

        # 4. if terminal_idx < self.no_samples than we have the terminal state within
        # the current horizon. In this case create the terminal state (foot step pos + zero vel)
        # and apply the state constraint to all states >= terminal_idx within the horizon
        #>>>>TODO: Add the terminal constraint if requires
        #Hint: If you are unsure, you can start testing without this first!
        if terminal_idx < self.no_samples:
            terminal_state = np.concatenate((foot_steps, np.zeros(2)))
            state_terminal = foot_steps[terminal_idx]

            for i in range(terminal_idx, self.no_samples):
                prog.AddLinearConstraint(state[i, :] == terminal_state)
                prog.AddConstraint(state[-1,0] == state_terminal[0])
                prog.AddConstraint(state[-1,1] == state_terminal[1])
                prog.AddConstraint(state[-1,2] == state_terminal[2])
                prog.AddConstraint(state[-1,3] == state_terminal[3])


        # setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
        #>>>>TODO: add the cost at each timestep, hint: prog.AddCost
        for k in range(NO_MPC_SAMPLES_HORIZON):
            prog.AddCost(alpha * (control[k][0]-self.ZMP_ref_k[0,k])*(control[k][0]-self.ZMP_ref_k[0,k]))
            prog.AddCost(alpha * (control[k][1]-self.ZMP_ref_k[1,k])*(control[k][1]-self.ZMP_ref_k[1,k]))
            prog.AddCost(gamma * (state[k,1]*state[k,1] + state[k,3]*state[k,3]))

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
    
################################################################################
# run the simulation
################################################################################

# inital state in x0 = [px0, vx0]
x_0 = np.array([0.0, 0.0])
# inital state in y0 = [py0, vy0]
y_0 = np.array([-0.09, 0.0])

# footprint
footprint = np.array([foot_length, foot_width])

# generate the footsteps
step_size_x = 0.2
step_size_y = 0.1
angle = np.pi/16
#>>>>TODO: 1. generate the foot step plan using generate_foot_steps
foot_steps= generate_foot_steps(np.array([x_0[0],y_0[0]]), step_size_x, step_size_y, NO_STEPS, turn_angle=angle) #None

# reapeat the last two foot steps (so the mpc horizon never exceeds the plan!)
foot_steps = np.vstack([
    foot_steps, foot_steps[-1], foot_steps[-1]])

# zmp reference trajecotry
#>>>>TODO: 2. generate the complete ZMP reference using generate_zmp_reference
zmp_ref, R_ref, foot_sides = generate_zmp_reference(foot_steps, NO_MPC_SAMPLES_PER_STEP, turn_angle=angle)

# generate mpc
mpc = MPC(T_MPC, T_HORIZON)

# generate the pendulum simulator
state_0 = np.concatenate([x_0, y_0])
sim = Simulator(state_0, T_SIM)

# setup some vectors for plotting stuff
TIME_VEC = np.nan*np.ones(NO_SIM_SAMPLES)
STATE_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 4])
ZMP_REF_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 2])
ZMP_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 2])

# time to add some disturbance
t_push = 3.2

# execution loop

k = 0   # the number of mpc update
for i in range(NO_SIM_SAMPLES):
    
    # simulation time
    t = i*T_SIM
        
    if i % NO_SIM_SAMPLES_PER_MPC == 0:
        # time to update the mpc
        
        # current state
        #>>>>TODO: get current state from the simulator
        x_k = sim.x #None
    
        #>>>>TODO: extract the current horizon from the complete reference trajecotry ZMP_ref
        ZMP_ref_k = np.array([zmp_ref[0,k:k+NO_MPC_SAMPLES_HORIZON],zmp_ref[1,k:k+NO_MPC_SAMPLES_HORIZON]]) #None
        R_ref_k = R_ref[k:k+NO_MPC_SAMPLES_HORIZON]
        foot_sides_k = foot_sides[k:k+NO_MPC_SAMPLES_HORIZON]

        # check if we have terminal constraint
        idx_terminal_k = NO_MPC_SAMPLES - k
        #>>>>TODO: Update the mpc, get new command
        #print(x_k)
        #print(ZMP_ref_k)
        u_k = mpc.buildSolveOCP(x_k, ZMP_ref_k, R_ref_k, foot_sides, NO_MPC_SAMPLES_HORIZON) #None
        
        k += 1
    
    # simulate a push for 0.05 sec with 1.0 m/s^2 acceleration 
    x_ddot_ext = np.array([0, 0])
    
    #>>>>TODO: when you got everything working try adding a small disturbance
    # if i > int(t_push/T_SIM) and i < int((t_push + 0.05)/T_SIM):
    #    x_ddot_ext = np.array([0, 1.0])
    
    #>>>>TODO: Update the simulation using the current command
    x_k = sim.simulate(u_k, d=np.zeros(2))[:,0]#None
    #print(x_k)
    # save some stuff
    TIME_VEC[i] = t
    STATE_VEC[i] = x_k
    ZMP_VEC[i] = u_k
    ZMP_REF_VEC[i] = mpc.ZMP_ref_k[:,0]
    
ZMP_LB_VEC = ZMP_REF_VEC - footprint[None,:]
ZMP_UB_VEC = ZMP_REF_VEC + footprint[None,:]

#>>>>TODO: Use the recodings in STATE_VEC and ZMP_VEC to compute the 
# LIP acceleration
#>>>>Hint: Use the continious dynamic matrices
STATE_DOT_VEC = np.zeros_like(STATE_VEC) #None
for i in range(len(STATE_VEC)):
    x = STATE_VEC[i, :]
    zmp = ZMP_VEC[i, :]
    
    # Compute the LIP acceleration using the continuous dynamics
    x_ddot = sim.A @ np.array(x)+ sim.B @ zmp

    STATE_DOT_VEC[i, :] = x_ddot

################################################################################
# plot something

#>>>>TODO: plot everything in x-axis
plt.figure()
plt.plot(TIME_VEC, STATE_VEC[:, 0], label='Com position x')
plt.plot(TIME_VEC, ZMP_REF_VEC[:,0], label='ZMP ref')
plt.plot(TIME_VEC, ZMP_LB_VEC[:, 0], label='ZMP lower Bound')
plt.plot(TIME_VEC, ZMP_UB_VEC[:, 0], label='ZMP upper Bound')
plt.plot(TIME_VEC, STATE_VEC[:, 1], linestyle='--',label='Velocity x')
plt.plot(TIME_VEC, STATE_DOT_VEC[:, 1], linestyle='--',label='Acceleration x')
#plt.plot(t, control_solution[:, 0], label='control input x')
plt.xlabel('time')
plt.ylabel('x axis')
plt.legend()

#>>>>TODO: plot everything in y-axis
plt.figure()
plt.plot(TIME_VEC, STATE_VEC[:, 2], label='Com position y')
plt.plot(TIME_VEC, ZMP_REF_VEC[:,1], label='ZMP ref')
plt.plot(TIME_VEC, ZMP_LB_VEC[:, 1], label='ZMP lower Bound')
plt.plot(TIME_VEC, ZMP_UB_VEC[:, 1], label='ZMP upper Bound')
plt.plot(TIME_VEC, STATE_VEC[:, 3], linestyle='--',label='Velocity y')
plt.plot(TIME_VEC, STATE_DOT_VEC[:, 3], linestyle='--',label='Acceleration y')
#plt.plot(t, control_solution[:, 1], label='control input y')
plt.xlabel('time')
plt.ylabel('y axis')
plt.legend()

#>>>>TODO: plot everything in xy-plane
fig, ax = plt.subplots()
plt.plot(STATE_VEC[:, 0], STATE_VEC[:, 2], label='position Com')
plot_foot_steps(foot_steps, footprint, ax)
ax.set_aspect('equal')
plt.xlabel('x axis [m]')
plt.ylabel('y axis [m]')
plt.legend()

plt.show()
