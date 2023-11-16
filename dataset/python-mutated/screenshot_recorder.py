import logging
import os
import time
from pyboy.logger import logger
from pyboy.plugins.base_plugin import PyBoyPlugin
from pyboy.utils import WindowEvent
logger = logging.getLogger(__name__)
try:
    from PIL import Image
except ImportError:
    Image = None
FPS = 60

class ScreenshotRecorder(PyBoyPlugin):

    def handle_events(self, events):
        if False:
            return 10
        for event in events:
            if event == WindowEvent.SCREENSHOT_RECORD:
                self.save()
                break
        return events

    def save(self, path=None):
        if False:
            print('Hello World!')
        if path is None:
            directory = os.path.join(os.path.curdir, 'screenshots')
            if not os.path.exists(directory):
                os.makedirs(directory, mode=493)
            path = os.path.join(directory, time.strftime(f'{self.pyboy.cartridge_title()}-%Y.%m.%d-%H.%M.%S.png'))
        self.pyboy.botsupport_manager().screen().screen_image().save(path)
        logger.info('Screenshot saved in {}'.format(path))

    def enabled(self):
        if False:
            while True:
                i = 10
        if Image is None:
            logger.warning(f'{__name__}: Missing dependency "Pillow". Screenshots disabled')
            return False
        return True