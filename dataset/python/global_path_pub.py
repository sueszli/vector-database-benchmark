#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import sys
import os

import heapq
from std_msgs.msg import Int16
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

# 현재 파일 경로와 시스템 경로를 추가해 준다
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)

# 정의한 클래스를 import 해온다.
from lib.mgeo.class_defs import *

# global_path_pub 클래스 정의
class global_path_pub:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('global_path_pub', anonymous=True)

        # global_path_pub 노드 생성
        rospy.Subscriber('/rec_global_path', Int16, self.rec_global_path)
        self.is_rec_global_path = False

        self.global_path_pub = rospy.Publisher('/global_path', Path, queue_size=1)
        self.global_path_msg = Path()
        self.global_path_msg.header.frame_id = '/map'

        # json 파일을 읽어와서 mgeo_planner_map을 만든다
        load_path = os.path.normpath(os.path.join(current_path, 'lib/mgeo_data/R_KR_PG_K-City'))
        mgeo_planner_map = MGeo.create_instance_from_json(load_path)

        # 노드와 링크를 가져온다
        self.nodes = mgeo_planner_map.node_set.nodes
        self.links = mgeo_planner_map.link_set.lines

        # 노드 패스를 설정한다
        self.node_path = ['A119BS010184', 'A119BS010269', 'A119BS010148', 'A119BS010695']

        # 시작 노드와 끝 노드 사이의 최단 경로를 찾는다
        # 최단 경로를 찾아서 global_path_msg에 추가해 준다
        for i in range(0, len(self.node_path) - 1):
            self.find_shortest_path(self.node_path[i], self.node_path[i + 1])

        rate = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            if self.is_rec_global_path:
                break

            self.global_path_pub.publish(self.global_path_msg)

            rate.sleep()

    def rec_global_path(self, msg):
        if msg.data == 1:
            self.is_rec_global_path = True

    def find_shortest_path(self, start_node_id, end_node_id):
        distances = dict()
        from_node = {}
        from_link = {}
        for node_id in self.nodes.keys():
            distances[node_id] = float('inf')

        distances[start_node_id] = 0
        queue = []
        heapq.heappush(queue, [distances[start_node_id], start_node_id])

        while queue:
            current_distance, current_node_id = heapq.heappop(queue)

            if current_node_id == end_node_id:
                break

            if distances[current_node_id] < current_distance:
                continue

            for link in self.nodes[current_node_id].get_to_links():
                adjacent_node_id = link.to_node.idx
                distance = current_distance + link.cost

                if distance < distances[adjacent_node_id]:
                    distances[adjacent_node_id] = distance
                    from_node[adjacent_node_id] = current_node_id
                    from_link[adjacent_node_id] = link.idx
                    heapq.heappush(queue, [distance, adjacent_node_id])

        link_path = []
        shortest_path = end_node_id

        while shortest_path != start_node_id:
            link_path.insert(0, from_link[shortest_path])
            shortest_path = from_node[shortest_path]

        for link_id in link_path:
            link = self.links[link_id]
            for point in link.points:
                pose = PoseStamped()
                pose.pose.position.x = point[0]
                pose.pose.position.y = point[1]
                pose.pose.orientation.w = 1
                self.global_path_msg.poses.append(pose)

if __name__ == '__main__':
    global_path_pub = global_path_pub()
