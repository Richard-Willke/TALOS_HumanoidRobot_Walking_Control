# lip_mpc.py Code Documentation

---
### Table of contents
* [**Module Overview**](#1)
* [**Functions**](#2)
    * [continious_LIP_dynamics](#2.1)
    * [discrete_LIP_dynamics](#2.2)
    * [generate_zmp_reference](#2.3)

* [**Class**](#3)
    * [discrete_LIP_dynamics](#3.1)
    * [continious_LIP_dynamics](#3.2)

    * [discrete_LIP_dynamics](#2.2)
    * [continious_LIP_dynamics](#2.1)
    * [discrete_LIP_dynamics](#2.2)
    * [continious_LIP_dynamics](#2.1)
    * [discrete_LIP_dynamics](#2.2)
---



## <h2 id="1">Module Overview</h2>

This module implements a Model Predictive Control (MPC) based on the Linear Inverted Pendulum (LIP) model for robot gait generation. The code includes calculations for continuous and discrete LIP dynamics, as well as interpolation and solving the MPC optimization problem.



---
## <h2 id="2">Functions</h2>

## <h3 id="2.1"><table><tr><td style="background-color: #4169E1; color: white;">continious_LIP_dynamics</td></tr></table></h3>
<!-- ### <h3 id="2.1">continious_LIP_dynamics -->

#### Functionality
Computes the static matrices A and B for continuous LIP dynamics.

#### Input Parameters
- `g` (float): Gravitational acceleration
- `h` (float): Height of the center of mass

#### Output Parameters
- `A` (numpy.ndarray): State transition matrix
- `B` (numpy.ndarray): Control matrix



## <h3 id="2.2"><table><tr><td style="background-color: #4169E1; color: white;">discrete_LIP_dynamics</td></tr></table></h3>
<!-- ### <h3 id="2.2">discrete_LIP_dynamics -->

#### Functionality
Computes the matrices Ad and Bd for discrete LIP dynamics.

#### Input Parameters
- `g` (float): Gravitational acceleration
- `h` (float): Height of the center of mass
- `dt` (float): Time step

#### Output Parameters
- `Ad` (numpy.ndarray): Discrete state transition matrix
- `Bd` (numpy.ndarray): Discrete control matrix



## <h3 id="2.3"><table><tr><td style="background-color: #4169E1; color: white;">generate_zmp_reference</td></tr></table></h3>
<!-- ### <h3 id="1.1">generate_zmp_reference</h3> -->

#### Functionality
Generates the ZMP reference trajectory given a sequence of footsteps.

#### Input Parameters
- `foot_steps` (list): List of footstep positions
- `no_samples_per_step` (int): Number of samples per step

#### Output Parameters
- `zmp_ref` (list): ZMP reference trajectory

---

## <h2 id="3">Class</h2>



















## <h3 id="2.3"><table><tr><td style="background-color: #4169E1; color: white;">LIPInterpolator</td></tr></table></h3>
<!-- ### <h3 id="2.3">LIPInterpolator</h3> -->

#### Class Overview
Implements interpolation using continuous LIP dynamics.

#### Initialization Method

##### Input Parameters
- `x_inital` (numpy.ndarray): Initial state
- `conf` (object): Configuration object containing model parameters




#### Methods


## <h3 id="2.3"><table><tr><td style="background-color: #4169E1; color: white;">LIPInterpolator</td></tr></table></h3>
##### integrate

###### Functionality
Performs state integration using the given control input `u`.

###### Input Parameters
- `u` (numpy.ndarray): Control input

###### Output Parameters
- `self.x` (numpy.ndarray): Updated state

##### comState

###### Functionality
Returns the center of mass (CoM) state, including position, velocity, and acceleration.

###### Output Parameters
- `c` (numpy.ndarray): CoM position
- `c_dot` (numpy.ndarray): CoM velocity
- `c_ddot` (numpy.ndarray): CoM acceleration

##### dcm

###### Functionality
Computes and returns the Divergent Component of Motion (DCM).

###### Output Parameters
- `dcm` (numpy.ndarray): Divergent Component of Motion

##### zmp

###### Functionality
Computes and returns the Zero Moment Point (ZMP).

###### Output Parameters
- `zmp` (numpy.ndarray): Zero Moment Point





### <h3 id="1.1">LIPMPC

#### Class Overview
Implements Model Predictive Control (MPC) based on the LIP model.

#### Initialization Method

##### Input Parameters
- `conf` (object): Configuration object containing model parameters

#### Methods

##### buildSolveOCP

###### Functionality
Builds and solves the Optimal Control Problem (OCP).

###### Input Parameters
- `x_k` (numpy.ndarray): Initial MPC state
- `ZMP_ref_k` (list): ZMP reference trajectory over the horizon
- `terminal_idx` (int): Index within the horizon to apply the terminal constraint

###### Output Parameters
- `control` (numpy.ndarray): Optimized control input






