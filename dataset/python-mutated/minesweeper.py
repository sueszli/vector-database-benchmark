from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import random
import time
IMG_BOMB = QImage('./images/bug.png')
IMG_FLAG = QImage('./images/flag.png')
IMG_START = QImage('./images/rocket.png')
IMG_CLOCK = QImage('./images/clock-select.png')
NUM_COLORS = {1: QColor('#f44336'), 2: QColor('#9C27B0'), 3: QColor('#3F51B5'), 4: QColor('#03A9F4'), 5: QColor('#00BCD4'), 6: QColor('#4CAF50'), 7: QColor('#E91E63'), 8: QColor('#FF9800')}
LEVELS = [(8, 10), (16, 40), (24, 99)]
STATUS_READY = 0
STATUS_PLAYING = 1
STATUS_FAILED = 2
STATUS_SUCCESS = 3
STATUS_ICONS = {STATUS_READY: './images/plus.png', STATUS_PLAYING: './images/smiley.png', STATUS_FAILED: './images/cross.png', STATUS_SUCCESS: './images/smiley-lol.png'}

class Pos(QWidget):
    expandable = pyqtSignal(int, int)
    clicked = pyqtSignal()
    ohno = pyqtSignal()

    def __init__(self, x, y, *args, **kwargs):
        if False:
            print('Hello World!')
        super(Pos, self).__init__(*args, **kwargs)
        self.setFixedSize(QSize(20, 20))
        self.x = x
        self.y = y

    def reset(self):
        if False:
            print('Hello World!')
        self.is_start = False
        self.is_mine = False
        self.adjacent_n = 0
        self.is_revealed = False
        self.is_flagged = False
        self.update()

    def paintEvent(self, event):
        if False:
            i = 10
            return i + 15
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = event.rect()
        if self.is_revealed:
            color = self.palette().color(QPalette.Background)
            (outer, inner) = (color, color)
        else:
            (outer, inner) = (Qt.gray, Qt.lightGray)
        p.fillRect(r, QBrush(inner))
        pen = QPen(outer)
        pen.setWidth(1)
        p.setPen(pen)
        p.drawRect(r)
        if self.is_revealed:
            if self.is_start:
                p.drawPixmap(r, QPixmap(IMG_START))
            elif self.is_mine:
                p.drawPixmap(r, QPixmap(IMG_BOMB))
            elif self.adjacent_n > 0:
                pen = QPen(NUM_COLORS[self.adjacent_n])
                p.setPen(pen)
                f = p.font()
                f.setBold(True)
                p.setFont(f)
                p.drawText(r, Qt.AlignHCenter | Qt.AlignVCenter, str(self.adjacent_n))
        elif self.is_flagged:
            p.drawPixmap(r, QPixmap(IMG_FLAG))

    def flag(self):
        if False:
            return 10
        self.is_flagged = True
        self.update()
        self.clicked.emit()

    def reveal(self):
        if False:
            return 10
        self.is_revealed = True
        self.update()

    def click(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.is_revealed:
            self.reveal()
            if self.adjacent_n == 0:
                self.expandable.emit(self.x, self.y)
        self.clicked.emit()

    def mouseReleaseEvent(self, e):
        if False:
            i = 10
            return i + 15
        if e.button() == Qt.RightButton and (not self.is_revealed):
            self.flag()
        elif e.button() == Qt.LeftButton:
            self.click()
            if self.is_mine:
                self.ohno.emit()

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(MainWindow, self).__init__(*args, **kwargs)
        (self.b_size, self.n_mines) = LEVELS[1]
        w = QWidget()
        hb = QHBoxLayout()
        self.mines = QLabel()
        self.mines.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.clock = QLabel()
        self.clock.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        f = self.mines.font()
        f.setPointSize(24)
        f.setWeight(75)
        self.mines.setFont(f)
        self.clock.setFont(f)
        self._timer = QTimer()
        self._timer.timeout.connect(self.update_timer)
        self._timer.start(1000)
        self.mines.setText('%03d' % self.n_mines)
        self.clock.setText('000')
        self.button = QPushButton()
        self.button.setFixedSize(QSize(32, 32))
        self.button.setIconSize(QSize(32, 32))
        self.button.setIcon(QIcon('./images/smiley.png'))
        self.button.setFlat(True)
        self.button.pressed.connect(self.button_pressed)
        l = QLabel()
        l.setPixmap(QPixmap.fromImage(IMG_BOMB))
        l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hb.addWidget(l)
        hb.addWidget(self.mines)
        hb.addWidget(self.button)
        hb.addWidget(self.clock)
        l = QLabel()
        l.setPixmap(QPixmap.fromImage(IMG_CLOCK))
        l.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hb.addWidget(l)
        vb = QVBoxLayout()
        vb.addLayout(hb)
        self.grid = QGridLayout()
        self.grid.setSpacing(5)
        vb.addLayout(self.grid)
        w.setLayout(vb)
        self.setCentralWidget(w)
        self.init_map()
        self.update_status(STATUS_READY)
        self.reset_map()
        self.update_status(STATUS_READY)
        self.show()

    def init_map(self):
        if False:
            i = 10
            return i + 15
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                w = Pos(x, y)
                self.grid.addWidget(w, y, x)
                w.clicked.connect(self.trigger_start)
                w.expandable.connect(self.expand_reveal)
                w.ohno.connect(self.game_over)

    def reset_map(self):
        if False:
            return 10
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                w = self.grid.itemAtPosition(y, x).widget()
                w.reset()
        positions = []
        while len(positions) < self.n_mines:
            (x, y) = (random.randint(0, self.b_size - 1), random.randint(0, self.b_size - 1))
            if (x, y) not in positions:
                w = self.grid.itemAtPosition(y, x).widget()
                w.is_mine = True
                positions.append((x, y))

        def get_adjacency_n(x, y):
            if False:
                for i in range(10):
                    print('nop')
            positions = self.get_surrounding(x, y)
            n_mines = sum((1 if w.is_mine else 0 for w in positions))
            return n_mines
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                w = self.grid.itemAtPosition(y, x).widget()
                w.adjacent_n = get_adjacency_n(x, y)
        while True:
            (x, y) = (random.randint(0, self.b_size - 1), random.randint(0, self.b_size - 1))
            w = self.grid.itemAtPosition(y, x).widget()
            if (x, y) not in positions:
                w = self.grid.itemAtPosition(y, x).widget()
                w.is_start = True
                for w in self.get_surrounding(x, y):
                    if not w.is_mine:
                        w.click()
                break

    def get_surrounding(self, x, y):
        if False:
            while True:
                i = 10
        positions = []
        for xi in range(max(0, x - 1), min(x + 2, self.b_size)):
            for yi in range(max(0, y - 1), min(y + 2, self.b_size)):
                positions.append(self.grid.itemAtPosition(yi, xi).widget())
        return positions

    def button_pressed(self):
        if False:
            while True:
                i = 10
        if self.status == STATUS_PLAYING:
            self.update_status(STATUS_FAILED)
            self.reveal_map()
        elif self.status == STATUS_FAILED:
            self.update_status(STATUS_READY)
            self.reset_map()

    def reveal_map(self):
        if False:
            for i in range(10):
                print('nop')
        for x in range(0, self.b_size):
            for y in range(0, self.b_size):
                w = self.grid.itemAtPosition(y, x).widget()
                w.reveal()

    def expand_reveal(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        for xi in range(max(0, x - 1), min(x + 2, self.b_size)):
            for yi in range(max(0, y - 1), min(y + 2, self.b_size)):
                w = self.grid.itemAtPosition(yi, xi).widget()
                if not w.is_mine:
                    w.click()

    def trigger_start(self, *args):
        if False:
            for i in range(10):
                print('nop')
        if self.status != STATUS_PLAYING:
            self.update_status(STATUS_PLAYING)
            self._timer_start_nsecs = int(time.time())

    def update_status(self, status):
        if False:
            print('Hello World!')
        self.status = status
        self.button.setIcon(QIcon(STATUS_ICONS[self.status]))

    def update_timer(self):
        if False:
            return 10
        if self.status == STATUS_PLAYING:
            n_secs = int(time.time()) - self._timer_start_nsecs
            self.clock.setText('%03d' % n_secs)

    def game_over(self):
        if False:
            for i in range(10):
                print('nop')
        self.reveal_map()
        self.update_status(STATUS_FAILED)
if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    app.exec_()