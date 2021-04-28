"""
testing offboard positon control with a simple takeoff script
"""

import rospy
from mavros_msgs.msg import State
from geometry_msgs.msg import Quaternion, Vector3, PoseStamped, TwistStamped,Point
import math
import numpy as np
from std_msgs.msg import Header
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import String
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from mavros_msgs.msg import AttitudeTarget
from sensor_msgs.msg import JointState

################################################
    #LQR
###################################################
MASS_POLE = 2
LENGTH_POLE = 0.5
RADIUS = 0.01
RADIUS_POLE = 0.025
G = 9.8
PKG = 'px4'
orientation = quaternion_from_euler(0, 0, -3.14 / 2)

class OffbPosCtl:
    curr_drone_pose = PoseStamped()
    waypointIndex = 0
    distThreshold = 0.5
    sim_ctr = 1

    des_pose = PoseStamped()
    isReadyToFly = False
    # location
    orientation = quaternion_from_euler(0, 0, -3.14 / 2)
    orientation_2 = quaternion_from_euler(0, 0, 3.14 / 2)
    orientation_3 = quaternion_from_euler(0, 0, 3.14)
    locations = np.matrix([[2, 0, 1, 0 * orientation[0], 0 * orientation[1], 0 * orientation[2], 0 * orientation[3]]])

    def mavrosTopicStringRoot(self, uavID=0):
        mav_topic_string = '/mavros/'
        return mav_topic_string

    def __init__(self):
        rospy.init_node('offboard_test', anonymous=True)
        pose_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        drone_pose_subscriber = rospy.Subscriber('/mavros/local_position/pose', PoseStamped,
                                                 callback=self.drone_pose_cb)
        rover_pose_subscriber = rospy.Subscriber('/mavros/local_position/pose', PoseStamped,
                                                 callback=self.rover_pose_cb)
        state_sub = rospy.Subscriber('/mavros/state', State, callback=self.drone_state_cb)
        attach = rospy.Publisher('/attach', String, queue_size=10)


        # For the Pole states
        self.att = AttitudeTarget()
        self.att_setpoint_pub = rospy.Publisher(
            'mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1)
        self._sub_invpend_state = rospy.Subscriber('/iris/joint_states', JointState, self.jstates_callback)
        self.curr_drone_vel = TwistStamped()
        self.vel_pole = 0
        self.pos_pole = 0
        # desired yawrate target
        self.des_yawrate = 0.1
        self.yawrate_tol = 0.02
        self.att.body_rate = Vector3()
        self.att.header = Header()


        NUM_UAV = 2
        mode_proxy = [None for i in range(NUM_UAV)]
        arm_proxy = [None for i in range(NUM_UAV)]

        # Comm for drones
        for uavID in range(0, NUM_UAV):
            mode_proxy[uavID] = rospy.ServiceProxy(self.mavrosTopicStringRoot(uavID) + '/set_mode', SetMode)
            arm_proxy[uavID] = rospy.ServiceProxy(self.mavrosTopicStringRoot(uavID) + '/cmd/arming', CommandBool)

        rate = rospy.Rate(200)  # Hz
        rate.sleep()
        self.des_pose = self.copy_pose(self.curr_drone_pose)
        shape = self.locations.shape

        while not rospy.is_shutdown():
            print(self.sim_ctr, shape[0], self.waypointIndex)
            success = [None for i in range(NUM_UAV)]
            for uavID in range(0, NUM_UAV):
                try:
                    success[uavID] = mode_proxy[uavID](1, 'OFFBOARD')
                except rospy.ServiceException as e:
                    print("mavros/set_mode service call failed: %s" % e)

            success = [None for i in range(NUM_UAV)]
            for uavID in range(0, NUM_UAV):
                rospy.wait_for_service(self.mavrosTopicStringRoot(uavID) + '/cmd/arming')

            for uavID in range(0, NUM_UAV):
                try:
                    success[uavID] = arm_proxy[uavID](True)
                except rospy.ServiceException as e:
                    print("mavros1/set_mode service call failed: %s" % e)
            # if self.waypointIndex is shape[0]:
            #     self.waypointIndex = 0
            #     self.sim_ctr += 1

            if self.isReadyToFly:
                des = self.set_desired_pose().position
                curr = self.curr_drone_pose.pose.position
                dist = math.sqrt(
                    (curr.x - des.x) * (curr.x - des.x) + (curr.y - des.y) * (curr.y - des.y) + (curr.z - des.z) * (
                            curr.z - des.z))
                print(dist)
                # if dist < 1:
                #     print("check")
                #     break
                # self.waypointIndex += 1
            pose_pub.publish(self.des_pose)
            self.statefeedback()
            x, y, a = self.LQR()
            self.att.body_rate.x = 1 * x
            self.att.body_rate.y = 1 * y
            self.att.body_rate.z = 0

            self.att.thrust = 10 * a
            self.att_setpoint_pub.publish(self.att)
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def jstates_callback(self, data):
        """ Callback function for subscribing /invpend/joint_states topic """
        self.pos_pole = data.position[0]
        self.vel_pole = data.velocity[0]

    def statefeedback(self):
        self.x = self.curr_drone_pose.pose.position.x
        self.y = self.curr_drone_pose.pose.position.y
        self.z = self.curr_drone_pose.pose.position.z
        o = self.curr_drone_pose.pose.orientation
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion([o.x, o.y, o.z, o.w])
        self.xdot = self.curr_drone_vel.twist.linear.x
        self.ydot = self.curr_drone_vel.twist.linear.y
        self.zdot = self.curr_drone_vel.twist.linear.z
        self.r = LENGTH_POLE * math.cos(self.pos_pole)
        self.s = LENGTH_POLE * math.sin(self.pos_pole)
        self.rdot = LENGTH_POLE * math.cos(self.vel_pole)
        self.sdot = LENGTH_POLE * math.sin(self.vel_pole)

    def LQR(self):
        self.Ky = np.array(
            [3.162277660168370, 4.110266480060687, 13.966488802439859, 50.213969275831396, 14.052305698647750])
        self.Kx = np.array(
            [-3.162277660168276, -4.110266480060570, 13.966488802439772, -50.213969275830540, -14.052305698647515])
        self.Kz = np.array([10.0, 4.4721])

        att_rate_x = np.dot(-self.Kx, np.array([self.x - 20, self.xdot, self.roll, self.r, self.rdot]))
        att_rate_y = np.dot(-self.Ky, np.array([self.y - 20, self.ydot, self.pitch, self.s, self.sdot]))
        a = np.dot(-self.Kz, np.array([self.z - 20, self.zdot])) + G

        print("x =", att_rate_x, "y =", att_rate_y, "a=", a)
        return att_rate_x, att_rate_y, a
    def set_desired_pose(self):
        self.des_pose.pose.position.x = self.locations[self.waypointIndex, 0]
        self.des_pose.pose.position.y = self.locations[self.waypointIndex, 1]
        self.des_pose.pose.position.z = self.locations[self.waypointIndex, 2]
        self.des_pose.pose.orientation.x = self.locations[self.waypointIndex, 3]
        self.des_pose.pose.orientation.y = self.locations[self.waypointIndex, 4]
        self.des_pose.pose.orientation.z = self.locations[self.waypointIndex, 5]
        self.des_pose.pose.orientation.w = self.locations[self.waypointIndex, 6]
        return self.des_pose.pose

    def copy_pose(self, pose):
        pt = pose.pose.position
        quat = pose.pose.orientation
        copied_pose = PoseStamped()
        copied_pose.header.frame_id = pose.header.frame_id
        copied_pose.pose.position = Point(pt.x, pt.y, pt.z)
        copied_pose.pose.orientation = Quaternion(quat.x, quat.y, quat.z, quat.w)
        return copied_pose

    def drone_pose_cb(self, msg):
        self.curr_drone_pose = msg

    def drone_vel_cb(self, msg):
        self.curr_drone_vel = msg

    def rover_pose_cb(self, msg):
        self.curr_rover_pose = msg

    def drone_state_cb(self, msg):
        print(msg.mode)
        if (msg.mode == 'OFFBOARD'):
            self.isReadyToFly = True
            print("readyToFly")


if __name__ == "__main__":
    OffbPosCtl()
