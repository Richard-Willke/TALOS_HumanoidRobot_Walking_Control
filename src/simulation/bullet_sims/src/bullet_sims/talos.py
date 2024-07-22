"""
Provides the Talos class for the Talos robot simulation.
"""
#!/usr/bin/env python3
import numpy as np
import pinocchio as pin
from typing import Type, TypeVar

# simulator
from simulator.robot import Robot

# whole-body controller
from tsid_wrap.tsid_wrapper import TSIDWrapper

# robot configs
from bullet_sims import talos_conf as conf

# footstep_planner
from foot_plan.footstep_planner import Side

# ROS visualizations
import rospy
import tf2_ros
import pybullet as pb

from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
import tf

class Talos:
    """
    Talos robot class combining whole-body control with pybullet simulation, and provides functions to read and set sensor values.

    Attributes:
        conf: Configuration parameters.
        sim: The simulation environment.
        stack: The TSIDWrapper instance for whole-body control.
        robot: The Robot instance for simulation.
        model: The robot model.
        data: Data for the robot model.
        support_foot: The support foot side.
        swing_foot: The swing foot side.
        zmp: Zero Moment Point (ZMP) position.
        dcm: Divergent Component of Motion (DCM) position.
        pub_joint: ROS publisher for joint states.
        broadcaster: ROS TransformBroadcaster for base link.
        pub_zmp_dcm: ROS publisher for ZMP and DCM markers.
        pub_wrench_left: ROS publisher for left foot wrench.
        pub_wrench_right: ROS publisher for right foot wrench.
    """
    ###########################################################################################
    #                                          Init                                           #
    ###########################################################################################
    def __init__(self, simulator):
        self.conf = conf
        self.sim = simulator
        self.stack = TSIDWrapper(conf)
        
        # spawn robot in simulation
        z_init = 1.1
        basePosition = np.array([0, 0, z_init])
        baseQuaternion = np.array([0, 0, 0, 1])

        self.robot = Robot(simulator, 
                           conf.urdf, 
                           self.stack.model, 
                           basePosition, 
                           baseQuaternion, 
                           np.zeros(37), 
                           useFixedBase=False, 
                           verbose=True)
        
        self.model = self.stack.model
        self.data = self.robot._model.createData()
        
        ########################################################################
        # state
        ########################################################################
        self.support_foot = Side.RIGHT
        self.swing_foot = Side.LEFT
        
        ########################################################################
        # estimators
        ########################################################################
        self.zmp, self.zmp_left, self.zmp_right = self.compute_ZMP()
        self.dcm = self.compute_DCM()
        
        ########################################################################
        # sensors
        ########################################################################
        # ft sensors
        pb.enableJointForceTorqueSensor(self.robot.id(), self.robot.jointNameIndexMap()['leg_right_6_joint'], True)
        pb.enableJointForceTorqueSensor(self.robot.id(), self.robot.jointNameIndexMap()['leg_left_6_joint'], True)
        
        ########################################################################
        # visualizations
        ########################################################################
        # Publisher
        self.pub_joint = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.pub_zmp_dcm = rospy.Publisher('/zmp_and_dcm', MarkerArray, queue_size=10)
        self.pub_wrench_left = rospy.Publisher('/wrench_left', WrenchStamped, queue_size=10)
        self.pub_wrench_right = rospy.Publisher('/wrench_right', WrenchStamped, queue_size=10)
    
    ###########################################################################################
    #                                         Update                                          #
    ###########################################################################################
    def update(self):
        """
        Updates the robot state, estimators, and WBC.
        """
        t = self.sim.simTime()
        dt = self.sim.stepTime()

        self.robot.update()
        pin.forwardKinematics(self.model, self.data, self.robot._q)
        pin.updateFramePlacements(self.model, self.data)
        
        # update the estimators
        self._update_zmp_estimate()
        self._update_dcm_estimate()
        
        # update wbc and send back to pybullet
        self._solve(t, dt)
        
    
    ###########################################################################################
    #                                          Foot                                           #
    ###########################################################################################
    def setSupportFoot(self, side):
        """
        Sets the support foot of the robot on the given side.

        Args:
            side (Side): The side (LEFT or RIGHT) to set as the support foot.
        """        
        self.support_foot = side

        if self.support_foot == Side.LEFT:
            self.stack.add_contact_LF()
            self.stack.remove_contact_RF()
        elif self.support_foot == Side.RIGHT:
            self.stack.add_contact_RF()
            self.stack.remove_contact_LF()

    
    def setSwingFoot(self, side):
        """
        Sets the swing foot of the robot on the given side.

        Args:
            side (Side): The side (LEFT or RIGHT) to set as the swing foot.
        """
        self.swing_foot = side
        
        if self.swing_foot == Side.LEFT:
            self.stack.remove_contact_LF()
            self.stack.add_contact_RF()
        elif self.swing_foot == Side.RIGHT:
            self.stack.remove_contact_RF()
            self.stack.add_contact_LF()

        
    def updateSwingFootRef(self, T_swing_w: pin.SE3, V_swing_w: np.ndarray, A_swing_w: np.ndarray):
        """
        Updates the swing foot motion reference.

        Args:
            T_swing_w: Pose reference. pin.SE3(R, p)
                                       Rotation Matrix R, T_swing_w.rotation, Shape (3, 3)
                                       Translation Vector p, T_swing_w.translation, Shape (3, )
            V_swing_w: Velocity reference. Shape (3,)
            A_swing_w: Acceleration reference. Shape (3,)
        """        
        if self.swing_foot == Side.LEFT:
            self.stack.set_LF_pos_ref(T_swing_w, V_swing_w, A_swing_w)
        elif self.swing_foot == Side.RIGHT:
            self.stack.set_RF_pos_ref(T_swing_w, V_swing_w, A_swing_w)


    def swingFootPose(self):
        """
        Returns the pose of the current swing foot.

        Returns:
            numpy.ndarray: Pose of the current swing foot. pin.SE3(R, p)
                           Rotation Matrix R, T_swing_w.rotation, Shape (3, 3)
                           Translation Vector p, T_swing_w.translation, Shape (3, )
        """
        if self.swing_foot == Side.LEFT:
            return self.stack.get_placement_LF()
        elif self.swing_foot == Side.RIGHT:
            return self.stack.get_placement_RF()

    

    def supportFootPose(self):
        """
        Returns the pose of the current support foot.

        Returns:
            numpy.ndarray: Pose of the current support foot. pin.SE3(R, p)
                           Rotation Matrix R, T_support_w.rotation, Shape (3, 3)
                           Translation Vector p, T_support_w.translation, Shape (3, )
        """
        if self.support_foot == Side.LEFT:
            return self.stack.get_placement_LF()
        elif self.support_foot == Side.RIGHT:
            return self.stack.get_placement_RF()



    ###########################################################################################
    #                                         Publish                                         #
    ###########################################################################################
    def publish(self):
        """
        Publishes various ROS topics for visualization.
        """
        self.publish_joint_states()

        T_b_w, T_dv, T_da = self.stack.baseState(self.dq_sol)
        self.publish_base_link(T_b_w)

        self.publish_wrenches()
        self.publish_dcm_zmp_markers()


    def publish_joint_states(self):
        """
        Publishes the joint states of the robot.
        """
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.frame_id = 'base_link'
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = [
            'leg_left_1_joint', 'leg_left_2_joint', 'leg_left_3_joint', 'leg_left_4_joint', 'leg_left_5_joint', 'leg_left_6_joint', 
            'leg_right_1_joint', 'leg_right_2_joint', 'leg_right_3_joint', 'leg_right_4_joint', 'leg_right_5_joint', 'leg_right_6_joint',
            'torso_1_joint', 'torso_2_joint',
            'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint',
            'arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint', 'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint', 'arm_right_7_joint',
            'head_1_joint', 'head_2_joint', 
        ]
        joint_state.position = self.robot._q[7:]     # q
        joint_state.velocity = self.robot._v[6:]     # q_dot
        self.pub_joint.publish(joint_state)
    

    def publish_base_link(self, T_b_w):
        """
        Publishes the base link transform.

        Args:
            T_b_w (numpy.ndarray): Base link transform. pin.SE3(R, p)
                                   Rotation Matrix R, T_b_w.rotation, Shape (3, 3)
                                   Translation Vector p, T_b_w.translation, Shape (3, )
        """
        msg = TransformStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'world'
        msg.child_frame_id = 'base_link'

        translation = T_b_w.translation
        rotation = T_b_w.rotation

        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        quaternion = tf.transformations.quaternion_from_matrix(T)

        msg.transform.translation.x = translation[0]
        msg.transform.translation.y = translation[1]
        msg.transform.translation.z = translation[2]
        msg.transform.rotation.x = quaternion[0]
        msg.transform.rotation.y = quaternion[1]
        msg.transform.rotation.z = quaternion[2]
        msg.transform.rotation.w = quaternion[3]
        self.broadcaster.sendTransform(msg)

    
    def publish_wrenches(self):
        """
        Publishes the wrenches for the left and right feet.
        """

        wrench_r_ankle, wrench_l_ankle = self.record_ankle_wrenches()
        if wrench_l_ankle is not None:
            ws = WrenchStamped()
            ws.header.stamp = rospy.Time.now()
            ws.header.frame_id = "leg_left_6_link"
            ws.wrench.force.x = wrench_l_ankle.linear[0]
            ws.wrench.force.y = wrench_l_ankle.linear[1]
            ws.wrench.force.z = wrench_l_ankle.linear[2]
            ws.wrench.torque.x = wrench_l_ankle.angular[0]
            ws.wrench.torque.y = wrench_l_ankle.angular[1]
            ws.wrench.torque.z = wrench_l_ankle.angular[2]
            self.pub_wrench_left.publish(ws)


        if wrench_r_ankle is not None:
            ws = WrenchStamped()
            ws.header.stamp = rospy.Time.now()
            ws.header.frame_id = "leg_right_6_link"
            ws.wrench.force.x = wrench_r_ankle.linear[0]
            ws.wrench.force.y = wrench_r_ankle.linear[1]
            ws.wrench.force.z = wrench_r_ankle.linear[2]
            ws.wrench.torque.x = wrench_r_ankle.angular[0]
            ws.wrench.torque.y = wrench_r_ankle.angular[1]
            ws.wrench.torque.z = wrench_r_ankle.angular[2]
            self.pub_wrench_right.publish(ws)


    def publish_dcm_zmp_markers(self):
        """
        Publishes DCM and ZMP markers for visualization.
        """
        zmp_marker = Marker()
        zmp_marker.header = Header()
        zmp_marker.header.stamp = rospy.Time.now()
        zmp_marker.header.frame_id = 'world'
        zmp_marker.id = 1
        zmp_marker.type = Marker.SPHERE
        zmp_marker.pose.position.x = self.zmp[0]
        zmp_marker.pose.position.y = self.zmp[1]
        zmp_marker.pose.position.z = 0.0
        zmp_marker.scale.x = 0.01
        zmp_marker.scale.y = 0.01
        zmp_marker.scale.z = 0.01
        zmp_marker.color.a = 1.0
        zmp_marker.color.r = 1.0
        zmp_marker.color.g = 0.0
        zmp_marker.color.b = 0.0

        zmp_left = Marker()
        zmp_left.header = Header()
        zmp_left.header.stamp = rospy.Time.now()
        zmp_left.header.frame_id = 'world'
        zmp_left.id = 3
        zmp_left.type = Marker.SPHERE
        zmp_left.pose.position.x = self.zmp_left[0]
        zmp_left.pose.position.y = self.zmp_left[1]
        zmp_left.pose.position.z = 0.0
        zmp_left.scale.x = 0.01
        zmp_left.scale.y = 0.01
        zmp_left.scale.z = 0.01
        zmp_left.color.a = 0.5
        zmp_left.color.r = 0.0
        zmp_left.color.g = 0.0
        zmp_left.color.b = 1.0

        zmp_right = Marker()
        zmp_right.header = Header()
        zmp_right.header.stamp = rospy.Time.now()
        zmp_right.header.frame_id = 'world'
        zmp_right.id = 4
        zmp_right.type = Marker.SPHERE
        zmp_right.pose.position.x = self.zmp_right[0]
        zmp_right.pose.position.y = self.zmp_right[1]
        zmp_right.pose.position.z = 0.0
        zmp_right.scale.x = 0.01
        zmp_right.scale.y = 0.01
        zmp_right.scale.z = 0.01
        zmp_right.color.a = 0.5
        zmp_right.color.r = 0.0
        zmp_right.color.g = 0.0
        zmp_right.color.b = 1.0

        dcm_marker = Marker()
        dcm_marker.header = Header()
        dcm_marker.header.stamp = rospy.Time.now()
        dcm_marker.header.frame_id = 'world'
        dcm_marker.id = 2
        dcm_marker.type = Marker.SPHERE
        dcm_marker.pose.position.x = self.dcm[0]
        dcm_marker.pose.position.y = self.dcm[1]
        dcm_marker.pose.position.z = 0.0
        dcm_marker.scale.x = 0.01
        dcm_marker.scale.y = 0.01
        dcm_marker.scale.z = 0.01
        dcm_marker.color.a = 1.0
        dcm_marker.color.r = 0.0
        dcm_marker.color.g = 1.0
        dcm_marker.color.b = 0.0

        marker_array = MarkerArray()
        marker_array.markers.append(zmp_marker)
        marker_array.markers.append(dcm_marker)
        marker_array.markers.append(zmp_left)
        marker_array.markers.append(zmp_right)
        self.pub_zmp_dcm.publish(marker_array)

    
    

    ############################################################################
    # private funcitons
    ############################################################################

    def _solve(self, t, dt):
        """
        Solves the WBC and commands the torque to the pybullet robot.

        Args:
            t (float): Current simulation time.
            dt (float): Simulation time step.
        """
        # get the current state
        q = self.robot.q()
        v = self.robot.v()
        
        # solve the whole body qp
        tau_sol, self.dq_sol = self.stack.update(q, v, t)
        self.robot.setActuatedJointTorques(tau_sol)

    
    def _update_zmp_estimate(self):
        """update the estimated zmp position. Shape (2,)
        """
        # self.zmp = self.compute_ZMP()
        self.zmp, self.zmp_left, self.zmp_right = self.compute_ZMP()

        
    def _update_dcm_estimate(self):
        """update the estimated dcm position. Shape (2,)
        """
        self.dcm = self.compute_DCM()


    def record_ankle_wrenches(self):
        """
        Records the ankle wrenches.

        Returns:
            tuple: Wrenches for the right and left ankles.
        """
        wren = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()['leg_right_6_joint'])[2]
        wnp = np.array([-wren[0], -wren[1], -wren[2], -wren[3], -wren[4], -wren[5]])
        wr_rankle = pin.Force(wnp)

        wren = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()['leg_left_6_joint'])[2]
        wnp = np.array([-wren[0], -wren[1], -wren[2], -wren[3], -wren[4], -wren[5]])
        wl_lankle = pin.Force(wnp)

        return wr_rankle, wl_lankle
    

    def record_ankle_positions(self):
        """
        Records the ankle positions.

        Returns:
            tuple: Positions for the left and right ankles, and the left and right ankle joints.
        """
        data = self.robot._model.createData()
        pin.framesForwardKinematics(self.robot._model, data, self.robot.q())

        H_w_lsole = data.oMf[self.robot._model.getFrameId("left_sole_link")]
        H_w_rsole = data.oMf[self.robot._model.getFrameId("right_sole_link")]

        H_w_lankle = data.oMf[self.robot._model.getFrameId("leg_left_6_joint")]
        H_w_rankle = data.oMf[self.robot._model.getFrameId("leg_right_6_joint")]

        return H_w_lsole, H_w_rsole, H_w_lankle, H_w_rankle
    

    def compute_ZMP(self, d=conf.d_ZMP):
        """
        Computes the ZMP based on force torque sensor readings.

        Args:
            d (float): Distance for ZMP computation.

        Returns:
            numpy.ndarray: ZMP position. Shape (2,)
        """
        # wr_rankle: right ankle wrench
        # f_x_right: [force] measured by the [right] ankle-[x]
        # tau_x_right: [torque] measured by the [right] ankle-[x]

        wr_rankle, wl_lankle = self.record_ankle_wrenches()
        H_w_lsole, H_w_rsole, H_w_lankle, H_w_rankle = self.record_ankle_positions()

        f_r = wr_rankle.linear
        t_r = wr_rankle.angular
        d_r = (H_w_rankle.translation - H_w_rsole.translation)[2]
        ZMP_r_sole = np.array([(-t_r[1] - f_r[0] * d_r)/ f_r[2],
                            (t_r[0] - f_r[1] * d_r)/ f_r[2], 
                            0])
        ZMP_r = H_w_rsole * ZMP_r_sole
        
        f_l = wl_lankle.linear
        t_l = wl_lankle.angular
        d_l = (H_w_lankle.translation - H_w_lsole.translation)[2]
        ZMP_l_sole = np.array([(-t_l[1] - f_l[0] * d_l)/ f_l[2],
                            (t_l[0] - f_l[1] * d_l)/ f_l[2], 
                            0])
        ZMP_l = H_w_lsole * ZMP_l_sole

        ZMP_all = np.array([(ZMP_r[0] * f_r[2] + ZMP_l[0] * f_l[2])/ (f_r[2] + f_l[2]),
                            (ZMP_r[1] * f_r[2] + ZMP_l[1] * f_l[2])/ (f_r[2] + f_l[2]),
                            0])

        # return ZMP_all
        return ZMP_all, ZMP_l, ZMP_r
    


    def compute_DCM(self, g=9.81):
        """
        Computes the DCM based on the current center of mass state.

        Args:
            g (float): Gravitational constant.

        Returns:
            numpy.ndarray: DCM position. Shape (2,)
        """
        # x is projection to the ground -> baseCoMPosition()
        xp = self.robot.baseCoMPosition()
        xp_dot = self.robot.baseCoMVelocity()

        omega = np.sqrt(g / xp[2])
        DCM = xp[:2] + xp_dot[:2] / omega
        return DCM