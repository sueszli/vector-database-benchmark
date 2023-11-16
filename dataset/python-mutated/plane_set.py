import os, sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.normpath(os.path.join(current_path, '../')))
from .plane import Plane

class PlaneSet(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.planes = list()

    def reorganize(self):
        if False:
            i = 10
            return i + 15
        for (i, plane) in enumerate(self.planes):
            plane.idx = i

    def add_plane(self, plane):
        if False:
            while True:
                i = 10
        self.planes.append(plane)

    def remove_plane(self, plane_to_delete):
        if False:
            for i in range(10):
                print('nop')
        self.planes.remove(plane_to_delete)

    def create_a_new_empty_plane(self):
        if False:
            i = 10
            return i + 15
        new_id = len(self.planes)
        self.add_plane(Plane(new_id))

    def get_last_plane(self):
        if False:
            for i in range(10):
                print('nop')
        return self.planes[-1]

    def save_as_json(self, filename):
        if False:
            i = 10
            return i + 15
        import json
        obj_to_save = []
        for plane in self.planes:
            if not plane.is_closed():
                continue
            obj_to_save.append({'node_idx': plane.get_node_idx_list()})
        with open(filename, 'w') as f:
            json.dump(obj_to_save, f)

    def load_from_json(self, node_set_obj, filename):
        if False:
            print('Hello World!')
        import json
        with open(filename, 'r') as f:
            list_of_info_for_each_plane = json.load(f)
        self.planes = list()
        for info in list_of_info_for_each_plane:
            self.create_a_new_empty_plane()
            self.get_last_plane().init_from_node_idx_list(node_set_obj, info['node_idx'])
        self.create_a_new_empty_plane()

    def _print(self):
        if False:
            for i in range(10):
                print('nop')
        for plane in self.planes:
            print(plane.to_string())