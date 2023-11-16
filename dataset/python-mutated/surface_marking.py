import os, sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))
from class_defs.base_plane import BasePlane
import numpy as np
from collections import OrderedDict

class SurfaceMarking(BasePlane):
    """
    노면표시를 나타내는 클래스. 두 가지 역할을 수행한다
    1) Mesh 생성 (예: Speedbump Mesh Guide 생성)
    2) PlannerMap에서 해당 표시를 인식 (현재 링크와 관련 있는 노면 표시를 조회 가능)
    """

    def __init__(self, points=None, idx=None):
        if False:
            while True:
                i = 10
        super(SurfaceMarking, self).__init__(points, idx)
        self.link_id_list = []
        self.road_id = ''
        self.link_list = list()
        self.type = None
        self.sub_type = None
        self.type_code_def = ''
        '이하는 MPL에서의 draw를 위함'
        self.plotted_obj = None
        self.reset_vis_mode_manual_appearance()

    def add_link_ref(self, link):
        if False:
            print('Hello World!')
        if link not in self.link_list:
            self.link_list.append(link)
        if self not in link.surface_markings:
            link.surface_markings.append(self)

    def draw_plot(self, axes):
        if False:
            return 10
        if self.vis_mode_line_width is not None and self.vis_mode_line_color is not None:
            self.plotted_obj = axes.plot(self.points[:, 0], self.points[:, 1], linewidth=self.vis_mode_line_width, color=self.vis_mode_line_color, markersize=1, marker='o')
            return
        else:
            self.plotted_obj = axes.plot(self.points[:, 0], self.points[:, 1], markersize=1, marker='o', color='b')

    def erase_plot(self):
        if False:
            while True:
                i = 10
        if self.plotted_obj is not None:
            for obj in self.plotted_obj:
                if obj.axes is not None:
                    obj.remove()

    def hide_plot(self):
        if False:
            for i in range(10):
                print('nop')
        if self.plotted_obj is not None:
            for obj in self.plotted_obj:
                obj.set_visible(False)

    def unhide_plot(self):
        if False:
            return 10
        if self.plotted_obj is not None:
            for obj in self.plotted_obj:
                obj.set_visible(True)

    def set_vis_mode_manual_appearance(self, width, color):
        if False:
            return 10
        self.vis_mode_line_width = width
        self.vis_mode_line_color = color

    def reset_vis_mode_manual_appearance(self):
        if False:
            return 10
        self.set_vis_mode_manual_appearance(None, None)

    @staticmethod
    def to_dict(obj):
        if False:
            i = 10
            return i + 15
        'json 파일 등으로 저장할 수 있는 dict 데이터로 변경한다'
        dict_data = {'idx': obj.idx, 'points': obj.points.tolist(), 'link_id_list': obj.link_id_list, 'road_id': obj.road_id, 'type': obj.type, 'sub_type': obj.sub_type}
        return dict_data

    @staticmethod
    def from_dict(dict_data, link_set=None):
        if False:
            return 10
        'json 파일등으로부터 읽은 dict 데이터에서 Signal 인스턴스를 생성한다'
        'STEP #1 파일 내 정보 읽기'
        idx = dict_data['idx']
        points = np.array(dict_data['points'])
        link_id_list = dict_data['link_id_list']
        road_id = dict_data['road_id']
        sm_type = dict_data['type']
        sm_subtype = dict_data['sub_type']
        'STEP #2 인스턴스 생성'
        obj = SurfaceMarking(points=points, idx=idx)
        obj.link_id_list = link_id_list
        obj.road_id = road_id
        obj.type = sm_type
        obj.sub_type = sm_subtype
        'STEP #3 인스턴스 참조 연결'
        if link_set is not None:
            for link_id in link_id_list:
                if link_id in link_set.lines.keys():
                    link = link_set.lines[link_id]
                    obj.add_link_ref(link)
        return obj

    def item_prop(self):
        if False:
            print('Hello World!')
        prop_data = OrderedDict()
        prop_data['idx'] = {'type': 'string', 'value': self.idx}
        prop_data['points'] = {'type': 'list<list<float>>', 'value': self.points.tolist()}
        prop_data['type'] = {'type': 'string', 'value': self.type}
        prop_data['sub_type'] = {'type': 'string', 'value': self.sub_type}
        prop_data['type_code_def'] = {'type': 'string', 'value': self.type_code_def}
        return prop_data