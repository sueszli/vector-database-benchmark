from pyboy.plugins.base_plugin import PyBoyPlugin
from pyboy.utils import WindowEvent

class AutoPause(PyBoyPlugin):
    argv = [('--autopause', {'action': 'store_true', 'help': 'Enable auto-pausing when window looses focus'})]

    def handle_events(self, events):
        if False:
            for i in range(10):
                print('nop')
        for event in events:
            if event == WindowEvent.WINDOW_UNFOCUS:
                events.append(WindowEvent.PAUSE)
            elif event == WindowEvent.WINDOW_FOCUS:
                events.append(WindowEvent.UNPAUSE)
        return events

    def enabled(self):
        if False:
            while True:
                i = 10
        return self.pyboy_argv.get('autopause')