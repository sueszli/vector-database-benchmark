__license__ = 'GPL v3'
__copyright__ = '2008, Kovid Goyal <kovid at kovidgoyal.net>'
from qt.core import QGraphicsView, QSize

class BookView(QGraphicsView):
    MINIMUM_SIZE = QSize(400, 500)

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        QGraphicsView.__init__(self, *args)
        self.preferred_size = self.MINIMUM_SIZE

    def minimumSizeHint(self):
        if False:
            print('Hello World!')
        return self.MINIMUM_SIZE

    def sizeHint(self):
        if False:
            for i in range(10):
                print('nop')
        return self.preferred_size

    def resize_for(self, width, height):
        if False:
            for i in range(10):
                print('nop')
        self.preferred_size = QSize(width, height)