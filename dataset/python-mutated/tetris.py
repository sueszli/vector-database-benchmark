"""
Function:
    俄罗斯方块
Author:
    Charles
微信公众号:
    Charles的皮卡丘
"""
import os
import sys
import random
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from .modules import tetrisShape, InnerBoard, ExternalBoard, SidePanel
'俄罗斯方块'

class TetrisGame(QMainWindow):
    game_type = 'tetris'

    def __init__(self, parent=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(TetrisGame, self).__init__(parent)
        self.rootdir = os.path.split(os.path.abspath(__file__))[0]
        self.is_paused = False
        self.is_started = False
        self.initUI()
    '界面初始化'

    def initUI(self):
        if False:
            print('Hello World!')
        self.setWindowIcon(QIcon(os.path.join(self.rootdir, 'resources/icon.jpg')))
        self.grid_size = 22
        self.fps = 200
        self.timer = QBasicTimer()
        self.setFocusPolicy(Qt.StrongFocus)
        layout_horizontal = QHBoxLayout()
        self.inner_board = InnerBoard()
        self.external_board = ExternalBoard(self, self.grid_size, self.inner_board)
        layout_horizontal.addWidget(self.external_board)
        self.side_panel = SidePanel(self, self.grid_size, self.inner_board)
        layout_horizontal.addWidget(self.side_panel)
        self.status_bar = self.statusBar()
        self.external_board.score_signal[str].connect(self.status_bar.showMessage)
        self.start()
        self.center()
        self.setWindowTitle('俄罗斯方块 —— Charles的皮卡丘')
        self.show()
        self.setFixedSize(self.external_board.width() + self.side_panel.width(), self.side_panel.height() + self.status_bar.height())
    '游戏界面移动到屏幕中间'

    def center(self):
        if False:
            while True:
                i = 10
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)
    '更新界面'

    def updateWindow(self):
        if False:
            print('Hello World!')
        self.external_board.updateData()
        self.side_panel.updateData()
        self.update()
    '开始'

    def start(self):
        if False:
            print('Hello World!')
        if self.is_started:
            return
        self.is_started = True
        self.inner_board.createNewTetris()
        self.timer.start(self.fps, self)
    '暂停/不暂停'

    def pause(self):
        if False:
            while True:
                i = 10
        if not self.is_started:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.timer.stop()
            self.external_board.score_signal.emit('Paused')
        else:
            self.timer.start(self.fps, self)
        self.updateWindow()
    '计时器事件'

    def timerEvent(self, event):
        if False:
            print('Hello World!')
        if event.timerId() == self.timer.timerId():
            removed_lines = self.inner_board.moveDown()
            self.external_board.score += removed_lines
            self.updateWindow()
        else:
            super(TetrisGame, self).timerEvent(event)
    '按键事件'

    def keyPressEvent(self, event):
        if False:
            i = 10
            return i + 15
        if not self.is_started or self.inner_board.current_tetris == tetrisShape().shape_empty:
            super(TetrisGame, self).keyPressEvent(event)
            return
        key = event.key()
        if key == Qt.Key_P:
            self.pause()
            return
        if self.is_paused:
            return
        elif key == Qt.Key_Left:
            self.inner_board.moveLeft()
        elif key == Qt.Key_Right:
            self.inner_board.moveRight()
        elif key == Qt.Key_Up:
            self.inner_board.rotateAnticlockwise()
        elif key == Qt.Key_Space:
            self.external_board.score += self.inner_board.dropDown()
        else:
            super(TetrisGame, self).keyPressEvent(event)
        self.updateWindow()