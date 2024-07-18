#!/usr/bin/env python3
import numpy as np
import pinocchio as pin

# simulator
from simulator.robot import Robot

# whole-body controller
from bullet_sims.tsid_wrapper import TSIDWrapper

# robot configs
from walking_control import talos_conf as conf

# footstep_planner
from walking_control.footstep_planner import Side

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
    """Talos robot
    combines wbc with pybullet, functions to read and set
    sensor values.
    """
    ###########################################################################################
    #                                          Init                                           #
    ###########################################################################################
    def __init__(self, simulator):
        self.conf = conf
        self.sim = simulator
        
        #>>>>TODO: Like allways create the tsid wrapper
        self.stack = TSIDWrapper(conf)
        
        # spawn robot in simulation
        #>>>>TODO: Create the pybullet robot for the simulation
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
        self.zmp = self.compute_ZMP()
        self.dcm = self.compute_DCM()
        
        ########################################################################
        # sensors
        ########################################################################
        # ft sensors
        #>>>>TODO: Turn on the force torque sensor in the robots feet
        pb.enableJointForceTorqueSensor(self.robot.id(), self.robot.jointNameIndexMap()['leg_right_6_joint'], True)
        pb.enableJointForceTorqueSensor(self.robot.id(), self.robot.jointNameIndexMap()['leg_left_6_joint'], True)

        
        ########################################################################
        # visualizations
        ########################################################################
        
        #>>>> TODO: joint state publisher
        self.pub_joint = rospy.Publisher('/joint_states', JointState, queue_size=10)

        #>>>> TODO: floating base broadcaster
        self.broadcaster = tf2_ros.TransformBroadcaster()
        
        #>>>> TODO: zmp and dcm point publisher 
        #>>>> Hint: Use visualization_msgs::MarkerArray, SPHERE to visualize 
        self.pub_zmp_dcm = rospy.Publisher('/zmp_and_dcm', MarkerArray, queue_size=10)
        
        #>>>> TODO: wrench publisher for left and right foot
        #>>>> Hint: use geometry_msgs::Wrench
        self.pub_wrench_left = rospy.Publisher('/wrench_left', WrenchStamped, queue_size=10)
        self.pub_wrench_right = rospy.Publisher('/wrench_right', WrenchStamped, queue_size=10)
           
    
    ###########################################################################################
    #                                         Update                                          #
    ###########################################################################################
    def update(self):
        """updates the robot
        """
        t = self.sim.simTime()
        dt = self.sim.stepTime()

        #>>>> TODO: update the pybullet robot
        self.robot.update()
        pin.forwardKinematics(self.model, self.data, self.robot._q)       # FIXME
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
        """sets the the support foot of the robot on given side
        """
        
        # The support foot is in rigid contact with the ground and should 
        # hold the weight of the robot
        self.support_foot = side
        
        #>>>> TODO: Activate the foot contact on the support foot
        #>>>> TODO: At the same time deactivate the motion task on the support foot
        if self.support_foot == Side.LEFT:
            self.stack.add_contact_LF()
            self.stack.remove_contact_RF()
        elif self.support_foot == Side.RIGHT:
            self.stack.add_contact_RF()
            self.stack.remove_contact_LF()

    
    def setSwingFoot(self, side):
        """sets the swing foot of the robot on given side
        """
        
        # The swing foot is not in contact and can move
        self.swing_foot = side
        
        #>>>> TODO: Deactivate the foot contact on the swing foot
        #>>>> TODO: At the same time turn on the motion task on the swing foot
        if self.swing_foot == Side.LEFT:
            self.stack.remove_contact_LF()
            self.stack.add_contact_RF()
        elif self.swing_foot == Side.RIGHT:
            self.stack.remove_contact_RF()
            self.stack.add_contact_LF()

        
    def updateSwingFootRef(self, T_swing_w, V_swing_w, A_swing_w):
        """updates the swing foot motion reference
        """
        
        #>>>> TODO: set the pose, velocity and acceleration on the swing foots
        # motion task
        if self.swing_foot == Side.LEFT:
            # self.stack.set_motion_LF(T_swing_w, V_swing_w, A_swing_w)
            self.stack.set_LF_pos_ref(T_swing_w, V_swing_w, A_swing_w)
        elif self.swing_foot == Side.RIGHT:
            # self.stack.set_motion_RF(T_swing_w, V_swing_w, A_swing_w)
            self.stack.set_RF_pos_ref(T_swing_w, V_swing_w, A_swing_w)


    def swingFootPose(self):
        """return the pose of the current swing foot
        """
        #>>>>TODO: return correct foot pose
        if self.swing_foot == Side.LEFT:
            return self.stack.get_placement_LF()
        elif self.swing_foot == Side.RIGHT:
            return self.stack.get_placement_RF()
        # return None
    

    def supportFootPose(self):
        """return the pose of the current support foot
        """
        #>>>>TODO: return correct foot pose
        if self.support_foot == Side.LEFT:
            return self.stack.get_placement_LF()
        elif self.support_foot == Side.RIGHT:
            return self.stack.get_placement_RF()
        # return None


    ###########################################################################################
    #                                         Publish                                         #
    ###########################################################################################
    def publish(self):
        # t = rospy.Time.now()
        
        #>>>> TODO: publish the jointstate
        self.publish_joint_states()
        
        #>>>> TODO: broadcast odometry
        T_b_w, T_dv, T_da = self.stack.baseState(self.dq_sol)
        self.publish_base_link(T_b_w)
        
        #>>>> TODO: publish feet wrenches
        self.publish_wrenches()

        #>>>> TODO: publish dcm and zmp marker
        self.publish_dcm_zmp_markers()


    def publish_joint_states(self):
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
        wrench_left = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()['leg_left_6_joint'])[2]
        wrench_right = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()['leg_right_6_joint'])[2]

        wrench_left_msg = WrenchStamped()
        wrench_left_msg.header = Header()
        wrench_left_msg.header.stamp = rospy.Time.now()
        wrench_left_msg.header.frame_id = 'leg_left_6_link'
        wrench_left_msg.wrench.force.x = wrench_left[0]
        wrench_left_msg.wrench.force.y = wrench_left[1]
        wrench_left_msg.wrench.force.z = wrench_left[2]
        wrench_left_msg.wrench.torque.x = wrench_left[3]
        wrench_left_msg.wrench.torque.y = wrench_left[4]
        wrench_left_msg.wrench.torque.z = wrench_left[5]
        self.pub_wrench_left.publish(wrench_left_msg)

        wrench_right_msg = WrenchStamped()
        wrench_right_msg.header = Header()
        wrench_right_msg.header.stamp = rospy.Time.now()
        wrench_right_msg.header.frame_id = 'leg_right_6_link'
        wrench_right_msg.wrench.force.x = wrench_right[0]
        wrench_right_msg.wrench.force.y = wrench_right[1]
        wrench_right_msg.wrench.force.z = wrench_right[2]
        wrench_right_msg.wrench.torque.x = wrench_right[3]
        wrench_right_msg.wrench.torque.y = wrench_right[4]
        wrench_right_msg.wrench.torque.z = wrench_right[5]
        self.pub_wrench_right.publish(wrench_right_msg)

    def publish_dcm_zmp_markers(self):
        zmp_marker = Marker()
        zmp_marker.header = Header()
        zmp_marker.header.stamp = rospy.Time.now()
        zmp_marker.header.frame_id = 'world'
        zmp_marker.id = 1
        zmp_marker.type = Marker.SPHERE
        zmp_marker.pose.position.x = self.zmp[0]
        zmp_marker.pose.position.y = self.zmp[1]
        zmp_marker.pose.position.z = 0.0
        zmp_marker.scale.x = 0.05
        zmp_marker.scale.y = 0.05
        zmp_marker.scale.z = 0.05
        zmp_marker.color.a = 1.0
        zmp_marker.color.r = 1.0
        zmp_marker.color.g = 0.0
        zmp_marker.color.b = 0.0

        dcm_marker = Marker()
        dcm_marker.header = Header()
        dcm_marker.header.stamp = rospy.Time.now()
        dcm_marker.header.frame_id = 'world'
        dcm_marker.id = 2
        dcm_marker.type = Marker.SPHERE
        dcm_marker.pose.position.x = self.dcm[0]
        dcm_marker.pose.position.y = self.dcm[1]
        dcm_marker.pose.position.z = 0.0
        dcm_marker.scale.x = 0.05
        dcm_marker.scale.y = 0.05
        dcm_marker.scale.z = 0.05
        dcm_marker.color.a = 1.0
        dcm_marker.color.r = 0.0
        dcm_marker.color.g = 1.0
        dcm_marker.color.b = 0.0

        marker_array = MarkerArray()
        marker_array.markers.append(zmp_marker)
        marker_array.markers.append(dcm_marker)
        self.pub_zmp_dcm.publish(marker_array)

    
    

    ############################################################################
    # private funcitons
    ############################################################################

    def _solve(self, t, dt):
        # get the current state
        q = self.robot.q()
        v = self.robot.v()
        
        # solve the whole body qp
        #>>>> TODO: sovle the wbc and command the torque to pybullet robot
        tau_sol, self.dq_sol = self.stack.update(q, v, t)
        self.robot.setActuatedJointTorques(tau_sol)

    
    def _update_zmp_estimate(self):
        """update the estimated zmp position
        """
        #>>>> TODO: compute the zmp based on force torque sensor readings
        self.zmp = self.compute_ZMP()

        
    def _update_dcm_estimate(self):
        """update the estimated dcm position
        """
        #>>>> TODO: compute the com based on current center of mass state
        self.dcm = self.compute_DCM()


    def record_ankle_wrenches(self):
        wren = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()['leg_right_6_joint'])[2]
        wnp = np.array([-wren[0], -wren[1], -wren[2], -wren[3], -wren[4], -wren[5]])
        wr_rankle = pin.Force(wnp)

        wren = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()['leg_left_6_joint'])[2]
        wnp = np.array([-wren[0], -wren[1], -wren[2], -wren[3], -wren[4], -wren[5]])
        wl_lankle = pin.Force(wnp)

        return wr_rankle, wl_lankle
    

    def record_ankle_positions(self):
        data = self.robot._model.createData()
        pin.framesForwardKinematics(self.robot._model, data, self.robot.q())

        H_w_lsole = data.oMf[self.robot._model.getFrameId("left_sole_link")]
        H_w_rsole = data.oMf[self.robot._model.getFrameId("right_sole_link")]

        H_w_lankle = data.oMf[self.robot._model.getFrameId("leg_left_6_joint")]
        H_w_rankle = data.oMf[self.robot._model.getFrameId("leg_right_6_joint")]

        return H_w_lsole, H_w_rsole, H_w_lankle, H_w_rankle
    

    def compute_ZMP(self, d=conf.d_ZMP):
        # wr_rankle: right ankle wrench
        # f_x_right: [force] measured by the [right] ankle-[x]
        # tau_x_right: [torque] measured by the [right] ankle-[x]

        # Ankle-Frame
        wr_rankle, wl_lankle = self.record_ankle_wrenches()

        f_x_right, f_y_right, f_z_right = wr_rankle.linear[0], wr_rankle.linear[1], wr_rankle.linear[2]
        tau_x_right, tau_y_right, tau_z_right = wr_rankle.angular[0], wr_rankle.angular[1], wr_rankle.angular[2]
        f_x_left, f_y_left, f_z_left = wl_lankle.linear[0], wl_lankle.linear[1], wl_lankle.linear[2]
        tau_x_left, tau_y_left, tau_z_left = wl_lankle.angular[0], wl_lankle.angular[1], wl_lankle.angular[2]

        p_x_right = ( - tau_y_right - f_x_right * d) / f_z_right
        p_y_right = (tau_x_right - f_y_right * d) / f_z_right
        p_x_left = ( - tau_y_left - f_x_left * d) / f_z_left
        p_y_left = (tau_x_left - f_y_left * d) / f_z_left

        # World-Frame
        H_w_lsole, H_w_rsole, H_w_lankle, H_w_rankle = self.record_ankle_positions()
        # p_x_right, p_y_right, _ = H_w_rsole.inverse() @ [p_x_right, p_y_right, 0]
        p_x_right, p_y_right, _, _ = (np.linalg.inv(H_w_rsole) @ [p_x_right, p_y_right, 0, 1])
        p_x_left, p_y_left, _, _ = (np.linalg.inv(H_w_lsole) @ [p_x_left, p_y_left, 0, 1])

        p_x = (p_x_right * f_z_right + p_x_left * f_z_left) / (f_z_right + f_z_left)
        p_y = (p_y_right * f_z_right + p_y_left * f_z_left) / (f_z_right + f_z_left)
        ZMP = np.array([p_x, p_y])
        return ZMP

    def compute_DCM(self, g=9.81):
        # x is projection to the ground -> baseCoMPosition()
        xp = self.robot.baseCoMPosition()
        xp_dot = self.robot.baseCoMVelocity()

        omega = np.sqrt(g / xp[2])
        DCM = xp[:2] + xp_dot[:2] / omega
        return DCM