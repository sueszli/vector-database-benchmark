from lib.mgeo.class_defs import *
import os
import sys
import rospy
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)

class get_mgeo:

    def __init__(self):
        if False:
            return 10
        rospy.init_node('test', anonymous=True)
        self.link_pub = rospy.Publisher('link', PointCloud, queue_size=1)
        self.node_pub = rospy.Publisher('node', PointCloud, queue_size=1)
        load_path = os.path.normpath(os.path.join(current_path, 'lib/mgeo_data/R_KR_PG_K-City'))
        mgeo_planner_map = MGeo.create_instance_from_json(load_path)
        node_set = mgeo_planner_map.node_set
        link_set = mgeo_planner_map.link_set
        self.nodes = node_set.nodes
        self.links = link_set.lines
        self.link_msg = self.getAllLinks()
        self.node_msg = self.getAllNode()
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.link_pub.publish(self.link_msg)
            self.node_pub.publish(self.node_msg)
            rate.sleep()

    def getAllLinks(self):
        if False:
            print('Hello World!')
        all_link = PointCloud()
        all_link.header.frame_id = 'map'
        for (k, v) in self.links.items():
            for i in range(v.get_last_idx() + 1):
                link_xyz = v.get_point_dict(i)['coord']
                all_link.points.append(Point32(link_xyz[0], link_xyz[1], link_xyz[2]))
        return all_link

    def getAllNode(self):
        if False:
            for i in range(10):
                print('nop')
        all_node = PointCloud()
        all_node.header.frame_id = 'map'
        for (_, v) in self.nodes.items():
            node_xyz = v.to_dict()['point']
            all_node.points.append(Point32(node_xyz[0], node_xyz[1], node_xyz[2]))
        return all_node
if __name__ == '__main__':
    test_track = get_mgeo()