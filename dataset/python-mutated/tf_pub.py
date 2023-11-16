import rospy
import tf
from nav_msgs.msg import Odometry

class Ego_listener:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        rospy.init_node('status_listener', anonymous=True)
        rospy.Subscriber('odom', Odometry, self.odom_callback)
        rospy.spin()

    def odom_callback(self, msg):
        if False:
            i = 10
            return i + 15
        self.is_odom = True
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.orientation_x = msg.pose.pose.orientation.x
        self.orientation_y = msg.pose.pose.orientation.y
        self.orientation_z = msg.pose.pose.orientation.z
        self.orientation_w = msg.pose.pose.orientation.w
        br = tf.TransformBroadcaster()
        br.sendTransform((msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z), (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w), rospy.Time.now(), 'Ego', 'map')
if __name__ == '__main__':
    try:
        tl = Ego_listener()
    except rospy.ROSInternalException:
        pass