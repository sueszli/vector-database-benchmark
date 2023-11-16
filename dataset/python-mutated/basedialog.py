"""
Base variable explorer dialog
"""
from qtpy.QtWidgets import QDialog
import qstylizer.style

class BaseDialog(QDialog):

    def __init__(self, parent=None):
        if False:
            return 10
        QDialog.__init__(self, parent)
        css = qstylizer.style.StyleSheet()
        css.QPushButton.setValues(padding='3px 15px 3px 15px')
        self.setStyleSheet(css.toString())

    def set_dynamic_width_and_height(self, screen_geometry, width_ratio=0.5, height_ratio=0.5):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update width and height using an updated screen geometry.\n        Use a ratio for the width and height of the dialog.\n        '
        screen_width = int(screen_geometry.width() * width_ratio)
        screen_height = int(screen_geometry.height() * height_ratio)
        self.resize(screen_width, screen_height)
        x = int(screen_geometry.center().x() - self.width() / 2)
        y = int(screen_geometry.center().y() - self.height() / 2)
        self.move(x, y)

    def show(self):
        if False:
            return 10
        super(BaseDialog, self).show()
        window = self.window()
        windowHandle = window.windowHandle()
        screen = windowHandle.screen()
        geometry = screen.geometry()
        self.set_dynamic_width_and_height(geometry)
        screen.geometryChanged.connect(self.set_dynamic_width_and_height)