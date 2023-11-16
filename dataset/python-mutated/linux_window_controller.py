from serpent.window_controller import WindowController
import subprocess
import shlex
import re

class LinuxWindowController(WindowController):

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def locate_window(self, name):
        if False:
            print('Hello World!')
        return subprocess.check_output(shlex.split(f'xdotool search --onlyvisible --name "^{name}$"')).decode('utf-8').strip()

    def move_window(self, window_id, x, y):
        if False:
            print('Hello World!')
        subprocess.call(shlex.split(f'xdotool windowmove {window_id} {x} {y}'))

    def resize_window(self, window_id, width, height):
        if False:
            while True:
                i = 10
        subprocess.call(shlex.split(f'xdotool windowsize {window_id} {width} {height}'))

    def focus_window(self, window_id):
        if False:
            i = 10
            return i + 15
        subprocess.call(shlex.split(f'xdotool windowactivate {window_id}'))

    def bring_window_to_top(self, window_id):
        if False:
            i = 10
            return i + 15
        subprocess.call(shlex.split(f'xdotool windowactivate {window_id}'))

    def is_window_focused(self, window_id):
        if False:
            i = 10
            return i + 15
        focused_window_id = subprocess.check_output(shlex.split('xdotool getwindowfocus')).decode('utf-8').strip()
        return focused_window_id == window_id

    def get_focused_window_name(self):
        if False:
            i = 10
            return i + 15
        focused_window_id = subprocess.check_output(shlex.split('xdotool getwindowfocus')).decode('utf-8').strip()
        return subprocess.check_output(shlex.split(f'xdotool getwindowname {focused_window_id}')).decode('utf-8').strip()

    def get_window_geometry(self, window_id):
        if False:
            print('Hello World!')
        geometry = dict()
        window_geometry = subprocess.check_output(shlex.split(f'xdotool getwindowgeometry {window_id}')).decode('utf-8').strip()
        size = re.match('\\s+Geometry: ([0-9]+x[0-9]+)', window_geometry.split('\n')[2]).group(1).split('x')
        geometry['width'] = int(size[0])
        geometry['height'] = int(size[1])
        window_information = subprocess.check_output(shlex.split(f'xwininfo -id {window_id}')).decode('utf-8').strip()
        geometry['x_offset'] = int(re.match('\\s+Absolute upper-left X:\\s+([0-9]+)', window_information.split('\n')[2]).group(1))
        geometry['y_offset'] = int(re.match('\\s+Absolute upper-left Y:\\s+([0-9]+)', window_information.split('\n')[3]).group(1))
        return geometry