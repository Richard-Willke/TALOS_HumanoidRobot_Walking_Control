#!/usr/bin/env python3
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

class SwingFootTrajectory:
    """SwingFootTrajectory
    Interpolate Foot trajectory between two swing foot poses
    
    TODO: Actually there we only have R3 interpolation, we should add orientation interpolation as well
    """
    def __init__(self, T0: pin.SE3, T1: pin.SE3, duration: float, height=0.05):
        """initialize SwingFootTrajectory

        Args:
            T0: Inital foot pose
            T1: Final foot pose
            duration: step of expected duration
            height: step height. Defaults to 0.05.

        Returns:
            SwingFootTrajectory object
        """
        self._height = height
        self._t_elapsed = 0.0
        self._duration = duration
        self.reset(T0, T1)

    def reset(self, T0, T1):
        '''reset back to zero, update poses

        Args:
            T0: Inital foot pose
            T1: Final foot pose

        Returns:
            None
        '''
        self.T0 = T0
        self.T1 = T1

        t = self._duration
        t_mid = t / 2

        self.coeffs_x = self.compute_quintic_coeffs(T0.translation[0], T1.translation[0], 0, 0, 0, 0, t)
        self.coeffs_y = self.compute_quintic_coeffs(T0.translation[1], T1.translation[1], 0, 0, 0, 0, t)
        self.coeffs_z = self.compute_quintic_coeffs_with_midpoint(T0.translation[2], T1.translation[2], 0, 0, 0, 0, t, self._height, t_mid)

        # for the interpolation of the orientation, we should use exponential map and log map to interpolate
        # We assume there are no rotations in x and y axis (could be constraint by Side object)
        self.ang_log_z = pin.log3(T0.rotation.T @ T1.rotation)[2]
        self.coeffs_u = self.compute_quintic_coeffs(0, self.ang_log_z, 0, 0, 0, 0, t)

    def compute_quintic_coeffs(self, p0, pf, v0, vf, a0, af, T):
        """compute quintic coefficients

        Args:
            p0: initial position
            pf: final position
            v0: initial velocity
            vf: final velocity
            a0: initial acceleration
            af: final acceleration
            T: duration
        
        Returns:
            coefficients(np.array): coefficients of quintic polynomial
        """
            
        A = np.array([
            [0, 0, 0, 0, 0, 1],                             # x(0)
            [T**5, T**4, T**3, T**2, T, 1],                 # x(T)
            [0, 0, 0, 0, 1, 0],                             # x_dot(0)
            [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],            # x_dot(T)
            [0, 0, 0, 2, 0, 0],                             # x_ddot(0)
            [20*T**3, 12*T**2, 6*T, 2, 0, 0]                # x_ddot(T)
        ])
        b = np.array([p0, pf, v0, vf, a0, af])
        coeffs = np.linalg.solve(A, b)
        return coeffs

    def compute_quintic_coeffs_with_midpoint(self, p0, pf, v0, vf, a0, af, T, p_mid, T_mid):
        """compute quintic coefficients with midpoint

        Args:
            p0: initial position
            pf: final position
            v0: initial velocity
            vf: final velocity
            a0: initial acceleration
            af: final acceleration
            T: duration
            p_mid: midpoint position
            T_mid: midpoint time

        Returns:
            coefficients(np.array): coefficients of quintic polynomial
        """
        A = np.array([
            [0, 0, 0, 0, 0, 0, 1],                                              # Initial Position
            [T_mid**6, T_mid**5, T_mid**4, T_mid**3, T_mid**2, T_mid, 1],       # Middle Position
            [T**6, T**5, T**4, T**3, T**2, T, 1],                               # Final Position
            [0, 0, 0, 0, 0, 1, 0],                                              # Initial Velocity
            [6*T**5, 5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],                        # Final Velocity
            [0, 0, 0, 0, 2, 0, 0],                                              # Initial Acceleration
            [30*T**4, 20*T**3, 12*T**2, 6*T, 2, 0, 0]                           # Final Acceleration
        ])
        b = np.array([p0, p_mid, pf, v0, vf, a0, af])
        coeffs = np.linalg.solve(A, b)
        return coeffs


    def isDone(self):
        """check if trajectory is done
        
        Args:
            None

        Returns:
            bool: True if last time step has been reached
        """
        is_close = np.isclose(self._t_elapsed, self._duration)
        if is_close:
            return True
        
        is_more = self._t_elapsed > self._duration
        if is_more:
            print("[Warning]: SwingFootTrajectory has passed the end time")
            return True
        else:
            return False
    
    def evaluate(self, t):
        """evaluate the result at time t

        Args:
            t: time
        
        Returns:
            pose(np.array): position
            vel(np.array): velocity
            acc(np.array): acceleration
        """
        if t > self._duration:
            t = self._duration

        pos_x = np.polyval(self.coeffs_x, t)
        pos_y = np.polyval(self.coeffs_y, t)
        pos_z = np.polyval(self.coeffs_z, t)
        pos_u = np.polyval(self.coeffs_u, t)

        vel_x = np.polyval(np.polyder(self.coeffs_x), t)
        vel_y = np.polyval(np.polyder(self.coeffs_y), t)
        vel_z = np.polyval(np.polyder(self.coeffs_z), t)
        vel_u = np.polyval(np.polyder(self.coeffs_u), t)

        acc_x = np.polyval(np.polyder(np.polyder(self.coeffs_x)), t)
        acc_y = np.polyval(np.polyder(np.polyder(self.coeffs_y)), t)
        acc_z = np.polyval(np.polyder(np.polyder(self.coeffs_z)), t)
        acc_u = np.polyval(np.polyder(np.polyder(self.coeffs_u)), t)

        pos = np.array([pos_x, pos_y, pos_z])
        rot = self.T0.rotation @ pin.exp3(np.array([0, 0, pos_u]))
        pose = pin.SE3(rot, pos)
        vel = np.array([vel_x, vel_y, vel_z, 0, 0, vel_u])
        acc = np.array([acc_x, acc_y, acc_z, 0, 0, acc_u])

        self._t_elapsed = t
        return pose, vel, acc

if __name__=="__main__":
    T0 = pin.SE3(np.eye(3), np.array([0, 0, 0]))
    T1 = pin.SE3(pin.rpy.rpyToMatrix(0,0,1.5), np.array([0.2, 0, 0]))

    duration = 3.0
    height = 0.05
    Trajectory = SwingFootTrajectory(T0, T1, duration, height)
    time_list = np.linspace(0, duration, 100)

    pos_list = []
    ang_list = []
    vel_list = []
    acc_list = []

    for t in time_list:
        pose, vel, acc = Trajectory.evaluate(t)
        pos = pose.translation
        ang = pin.log3(pose.rotation)
        pos_list.append(pos)
        ang_list.append(ang)
        vel_list.append(vel)
        acc_list.append(acc)

    figure, ax = plt.subplots(3, 2, figsize=(15, 6))
    ax[0][0].plot(time_list, np.array(pos_list)[:, 0], label='Pos x')
    ax[0][0].plot(time_list, np.array(pos_list)[:, 1], label='Pos y')
    ax[0][0].plot(time_list, np.array(pos_list)[:, 2], label='Pos z')
    ax[0][1].plot(time_list, np.array(ang_list)[:, 0], label='Ang x')
    ax[0][1].plot(time_list, np.array(ang_list)[:, 1], label='Ang y')
    ax[0][1].plot(time_list, np.array(ang_list)[:, 2], label='Ang z')
    
    ax[1][0].plot(time_list, np.array(vel_list)[:, 0], label='Vel x')
    ax[1][0].plot(time_list, np.array(vel_list)[:, 1], label='Vel y')
    ax[1][0].plot(time_list, np.array(vel_list)[:, 2], label='Vel z')
    ax[1][1].plot(time_list, np.array(vel_list)[:, 3], label='Ang Vel x')
    ax[1][1].plot(time_list, np.array(vel_list)[:, 4], label='Ang Vel y')
    ax[1][1].plot(time_list, np.array(vel_list)[:, 5], label='Ang Vel z')

    ax[2][0].plot(time_list, np.array(acc_list)[:, 0], label='Acc x')
    ax[2][0].plot(time_list, np.array(acc_list)[:, 1], label='Acc y')
    ax[2][0].plot(time_list, np.array(acc_list)[:, 2], label='Acc z')
    ax[2][1].plot(time_list, np.array(acc_list)[:, 3], label='Ang Acc x')
    ax[2][1].plot(time_list, np.array(acc_list)[:, 4], label='Ang Acc y')
    ax[2][1].plot(time_list, np.array(acc_list)[:, 5], label='Ang Acc z')

    for i in range(3):
        for j in range(2):
            ax[i][j].legend(loc='upper right')
            ax[i][j].grid(True)

    plt.show()