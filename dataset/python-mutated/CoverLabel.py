"""
Created on 2023/02/22
@author: Irony
@site: https://pyqt.site https://github.com/PyQt5
@email: 892768447@qq.com
@file: CoverLabel.py
@description:
"""
import webbrowser
try:
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QLabel
except ImportError:
    from PySide2.QtGui import QPixmap
    from PySide2.QtWidgets import QLabel
from .Ui_CoverLabel import Ui_CoverLabel

class CoverLabel(QLabel, Ui_CoverLabel):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(CoverLabel, self).__init__(*args, **kwargs)
        self.setupUi(self)

    def init(self, cover_path, play_url, play_count):
        if False:
            i = 10
            return i + 15
        self.cover_path = cover_path
        self.play_url = play_url
        self.setPixmap(QPixmap(cover_path))
        self.labelHeadset.setPixmap(QPixmap('Data/Svg_icon_headset_sm.svg'))
        self.labelPlay.setPixmap(QPixmap('Data/Svg_icon_play_sm.svg'))
        self.labelCount.setStyleSheet('color: #999999;')
        self.labelCount.setText(play_count)

    def setCoverPath(self, path):
        if False:
            return 10
        self.cover_path = path

    def mouseReleaseEvent(self, event):
        if False:
            return 10
        super(CoverLabel, self).mouseReleaseEvent(event)
        webbrowser.open_new_tab(self.play_url)