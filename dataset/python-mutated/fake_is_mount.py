import os
from typing import List

class FakeIsMount:

    def __init__(self, mount_points):
        if False:
            print('Hello World!')
        self.mount_points = mount_points

    def is_mount(self, path):
        if False:
            return 10
        if path == '/':
            return True
        path = os.path.normpath(path)
        if path in self.mount_points_list():
            return True
        return False

    def mount_points_list(self):
        if False:
            while True:
                i = 10
        return set(['/'] + self.mount_points)

    def add_mount_point(self, path):
        if False:
            i = 10
            return i + 15
        self.mount_points.append(path)