import time
import threading
from .iomediator import IoMediator
SEND_LOCK = threading.Lock()

class WindowGrabber:

    def __init__(self, dialog):
        if False:
            for i in range(10):
                print('nop')
        self.dialog = dialog

    def start(self):
        if False:
            return 10
        time.sleep(0.1)
        IoMediator.listeners.append(self)

    def handle_keypress(self, raw_key, modifiers, key, *args):
        if False:
            i = 10
            return i + 15
        pass

    def handle_mouseclick(self, root_x, root_y, rel_x, rel_y, button, window_info):
        if False:
            for i in range(10):
                print('nop')
        IoMediator.listeners.remove(self)
        self.dialog.receive_window_info(window_info)