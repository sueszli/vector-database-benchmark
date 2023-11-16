import os
import sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))
from class_defs import *
import numpy as np
import json

def load_node_and_link(node_save_info_list, line_save_info_list):
    if False:
        i = 10
        return i + 15
    node_set = NodeSet()
    link_set = LineSet()
    for save_info in node_save_info_list:
        idx = save_info['idx']
        point = save_info['point']
        node = Node(idx)
        node.point = np.array(point)
        node_set.nodes.append(node)
    for save_info in line_save_info_list:
        idx = save_info['idx']
        from_node_idx = save_info['from_node_idx']
        to_node_idx = save_info['to_node_idx']
        points = save_info['points']
        lazy_init = save_info['lazy_init']
        from_node = node_set.nodes[from_node_idx]
        to_node = node_set.nodes[to_node_idx]
        if from_node_idx > len(node_set.nodes):
            for i in range(len(node_set.nodes)):
                node = node_set.nodes[i]
                if from_node_idx == node.idx:
                    print('i = {}, from_node_idx = {}'.format(i, from_node_idx))
            raise BaseException('[ERROR] from_node_idx = {} > node_set.nodes = {}'.format(from_node_idx, len(node_set.nodes)))
        if to_node_idx > len(node_set.nodes):
            raise BaseException('[ERROR] to_node_idx   = {} > node_set.nodes = {}'.format(to_node_idx, len(node_set.nodes)))
        link = Link(idx=idx, lazy_point_init=lazy_init)
        link.set_from_node(from_node)
        link.set_to_node(to_node)
        link.set_points(np.array(points))
        link_set.lines.append(link)
    for save_info in line_save_info_list:
        idx = save_info['idx']
        link = link_set.lines[idx]
        if link.is_it_for_lane_change():
            lane_ch_from_link_idx = save_info['lane_ch_from_link_idx']
            lane_ch_to_link_idx = save_info['lane_ch_to_link_idx']
            lane_ch_from_link = link_set.lines[lane_ch_from_link_idx]
            lane_ch_to_link = link_set.lines[lane_ch_to_link_idx]
            lane_ch_link_path = [lane_ch_from_link, lane_ch_to_link]
            link.set_values_for_lane_change_link(lane_ch_link_path)
    for link in link_set.lines:
        link.calculate_cost()
    return (node_set, link_set)