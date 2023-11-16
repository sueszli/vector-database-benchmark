from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton

class TextButton(QPushButton):

    def __init__(self, *args, height=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        .. versionadded:: 3.9\n            The *height* argument.\n        '
        super().__init__(*args, **kwargs)
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect, True)
        if height:
            self.setFixedHeight(height)