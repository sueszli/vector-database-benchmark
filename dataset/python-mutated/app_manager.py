from typing import Optional
from PyQt5.QtWidgets import QApplication
from tribler.gui.utilities import connect

class AppManager:
    """
    A helper class that calls QApplication.quit()

    You should never call `QApplication.quit()` directly. Call `app_manager.quit_application()` instead.
    It is necessary to avoid runtime errors like "wrapped C/C++ object of type ... has been deleted".

    After `app_manager.quit_application()` was called, it is not safe to access Qt objects anymore.
    If a signal can be emitted during the application shutdown, you can check `app_manager.quitting_app` flag
    inside the signal handler to be sure that it is still safe to access Qt objects.
    """

    def __init__(self, app: Optional[QApplication]=None):
        if False:
            return 10
        self.quitting_app = False
        if app is not None:
            connect(app.aboutToQuit, self.on_about_to_quit)

    def on_about_to_quit(self):
        if False:
            print('Hello World!')
        self.quitting_app = True

    def quit_application(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.quitting_app:
            self.quitting_app = True
            QApplication.quit()