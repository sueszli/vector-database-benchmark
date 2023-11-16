import logging
import os
import time
from pyboy.plugins.base_plugin import PyBoyPlugin
from pyboy.utils import WindowEvent
logger = logging.getLogger(__name__)
try:
    from PIL import Image
except ImportError:
    Image = None
FPS = 60

class ScreenRecorder(PyBoyPlugin):

    def __init__(self, *args):
        if False:
            print('Hello World!')
        super().__init__(*args)
        self.recording = False
        self.frames = []

    def handle_events(self, events):
        if False:
            i = 10
            return i + 15
        for event in events:
            if event == WindowEvent.SCREEN_RECORDING_TOGGLE:
                self.recording ^= True
                if not self.recording:
                    self.save()
                else:
                    logger.info('ScreenRecorder started')
                break
        return events

    def post_tick(self):
        if False:
            for i in range(10):
                print('nop')
        if self.recording:
            self.add_frame(self.pyboy.botsupport_manager().screen().screen_image())

    def add_frame(self, frame):
        if False:
            i = 10
            return i + 15
        self.frames.append(frame)

    def save(self, path=None, fps=60):
        if False:
            return 10
        logger.info('ScreenRecorder saving...')
        if path is None:
            directory = os.path.join(os.path.curdir, 'recordings')
            if not os.path.exists(directory):
                os.makedirs(directory, mode=493)
            path = os.path.join(directory, time.strftime(f'{self.pyboy.cartridge_title()}-%Y.%m.%d-%H.%M.%S.gif'))
        if len(self.frames) > 0:
            self.frames[0].save(path, save_all=True, interlace=False, loop=0, optimize=True, append_images=self.frames[1:], duration=int(round(1000 / fps, -1)))
            logger.info('Screen recording saved in {}'.format(path))
        else:
            logger.error('Screen recording failed: no frames')
        self.frames = []

    def enabled(self):
        if False:
            while True:
                i = 10
        if Image is None:
            logger.warning(f'{__name__}: Missing dependency "Pillow". Recording disabled')
            return False
        return True