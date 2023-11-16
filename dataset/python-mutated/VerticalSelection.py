from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QTransform
from urh.ui.painting.Selection import Selection

class VerticalSelection(Selection):

    def __init__(self, *args, fillcolor, opacity, parent=None):
        if False:
            return 10
        super().__init__(*args, fillcolor=fillcolor, opacity=opacity, parent=parent)

    @property
    def length(self):
        if False:
            print('Hello World!')
        return self.height

    @property
    def is_empty(self) -> bool:
        if False:
            while True:
                i = 10
        return self.height == 0

    @property
    def start(self):
        if False:
            return 10
        if self.height < 0:
            return self.y + self.height
        else:
            return self.y

    @start.setter
    def start(self, value):
        if False:
            print('Hello World!')
        self.setY(value)

    @property
    def end(self):
        if False:
            for i in range(10):
                print('nop')
        if self.height < 0:
            return self.y
        else:
            return self.y + self.height

    @end.setter
    def end(self, value):
        if False:
            while True:
                i = 10
        self.height = value - self.start

    def clear(self):
        if False:
            i = 10
            return i + 15
        self.height = 0
        super().clear()

    def get_selected_edge(self, pos: QPointF, transform: QTransform):
        if False:
            print('Hello World!')
        return super()._get_selected_edge(pos, transform, horizontal_selection=False)