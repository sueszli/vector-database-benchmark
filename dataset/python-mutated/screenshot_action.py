import os
from tribler_apptester.action import Action

class ScreenshotAction(Action):
    """
    This action takes a screenshot of the user interface.
    """

    def action_code(self):
        if False:
            for i in range(10):
                print('nop')
        return "timestamp = int(time.time())\npixmap = QPixmap(window.rect().size())\nwindow.render(pixmap, QPoint(), QRegion(window.rect()))\nimg_name = 'screenshot_%%d.jpg' %% timestamp\nscreenshots_dir = '%s'\nif not os.path.exists(screenshots_dir):\n    os.mkdir(screenshots_dir)\npixmap.save(os.path.join(screenshots_dir, img_name))\n        " % os.path.join(os.getcwd(), 'screenshots').replace('\\', '\\\\')

    def required_imports(self):
        if False:
            return 10
        return ['import time', 'import os', 'from PyQt5.QtGui import QPixmap, QRegion', 'from PyQt5.QtCore import QPoint']