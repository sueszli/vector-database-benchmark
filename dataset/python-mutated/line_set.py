import os, sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))
import numpy as np
from class_defs.node_set import NodeSet
from class_defs.node import Node
from class_defs.key_maker import KeyMaker

class LineSet(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.lines = dict()
        self.key_maker = KeyMaker(prefix='LN')

    def append_line(self, line_obj, create_new_key=False):
        if False:
            i = 10
            return i + 15
        if create_new_key:
            idx = self.key_maker.get_new()
            while idx in self.lines.keys():
                idx = self.key_maker.get_new()
            line_obj.idx = idx
        self.lines[line_obj.idx] = line_obj

    def remove_line(self, line_obj):
        if False:
            print('Hello World!')
        if line_obj.idx in self.lines.keys():
            self.lines.pop(line_obj.idx)

    def draw_plot(self, axes):
        if False:
            while True:
                i = 10
        for (idx, line) in self.lines.items():
            line.draw_plot(axes)

    def erase_plot(self):
        if False:
            return 10
        for (idx, line) in self.lines.items():
            line.erase_plot()

    def get_ref_points(self):
        if False:
            for i in range(10):
                print('nop')
        ref_points = list()
        for (idx, line) in self.lines.items():
            if line == None:
                continue
            mid_point = int(len(line.points) / 2.0)
            point_start = line.get_point_dict(0)
            point_mid = line.get_point_dict(mid_point)
            point_end = line.get_point_dict(-1)
            ref_points.append(point_start)
            ref_points.append(point_mid)
            ref_points.append(point_end)
        return ref_points

    def create_node_set_for_all_lines(self, node_set=None, dist_threshold=0.1):
        if False:
            print('Hello World!')
        '\n        각 line의 끝에 node를 생성한다.\n        이 때 argument로 전달된 거리값 이내에 다른 선이 존재하면, 같은 node로 판별하고 연결한다 \n        '
        if node_set is None:
            node_set = NodeSet()
        ref_points = self.get_ref_points()
        for (idx, current_link) in self.lines.items():
            if current_link.get_from_node() is None:
                new_node = Node()
                new_node.point = current_link.points[0]
                node_set.append_node(new_node, create_new_key=True)
                current_link.set_from_node(new_node)
                for pts in ref_points:
                    if current_link is pts['line_ref']:
                        continue
                    if pts['type'] == 'mid':
                        continue
                    dist = np.sqrt((pts['coord'][0] - new_node.point[0]) ** 2 + (pts['coord'][1] - new_node.point[1]) ** 2)
                    if dist < dist_threshold:
                        if pts['type'] == 'end':
                            pts['line_ref'].set_to_node(new_node)
                        else:
                            pts['line_ref'].set_from_node(new_node)
            if current_link.get_to_node() == None:
                new_node = Node()
                new_node.point = current_link.points[-1]
                node_set.append_node(new_node, create_new_key=True)
                current_link.set_to_node(new_node)
                for pts in ref_points:
                    if current_link is pts['line_ref']:
                        continue
                    if pts['type'] == 'mid':
                        continue
                    dist = np.sqrt((pts['coord'][0] - new_node.point[0]) ** 2 + (pts['coord'][1] - new_node.point[1]) ** 2)
                    if dist < dist_threshold:
                        if pts['type'] == 'start':
                            pts['line_ref'].set_from_node(new_node)
                        else:
                            pts['line_ref'].set_to_node(new_node)
        return node_set

    def set_vis_mode_all_different_color(self, on_off):
        if False:
            i = 10
            return i + 15
        '\n        NOTE: list, dict를 모두 지원하게 만들었으므로, 향후 변경이 필요없다\n        '
        for var in self.lines:
            if isinstance(self.lines, list):
                line = var
            elif isinstance(self.lines, dict):
                line = self.lines[var]
            line.set_vis_mode_all_different_color(on_off)

    @staticmethod
    def merge_two_sets(setA, setB):
        if False:
            return 10
        new_set = LineSet()
        setA.lines.update(setB.lines)
        return setA

    def merge_line_set(self, a_lines):
        if False:
            i = 10
            return i + 15
        for line in a_lines:
            if line not in self.lines.keys():
                self.lines[line] = a_lines[line]
                self.lines[line].copy_attributes(self.lines[line], a_lines[line])
        return self.lines