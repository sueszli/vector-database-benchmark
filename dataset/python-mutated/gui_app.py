import os
import sys
from PyQt5.QtCore import Qt, QDir
from PyQt5.QtGui import QIcon, QPixmap, QGuiApplication
from PyQt5.QtWidgets import QApplication, QWidget
from feeluown.gui.browser import Browser
from feeluown.gui.hotkey import HotkeyManager
from feeluown.gui.image import ImgManager
from feeluown.gui.theme import ThemeManager
from feeluown.gui.tips import TipsManager
from feeluown.gui.watch import WatchManager
from feeluown.gui.ui import Ui
from feeluown.gui.tray import Tray
from feeluown.gui.provider_ui import ProviderUiManager, CurrentProviderUiManager
from feeluown.gui.uimodels.playlist import PlaylistUiManager
from feeluown.gui.uimodels.my_music import MyMusicUiManager
from feeluown.collection import CollectionManager
from .app import App

class GuiApp(App, QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        config = args[1]
        pkg_root_dir = os.path.join(os.path.dirname(__file__), '..')
        icons_dir = os.path.join(pkg_root_dir, 'gui/assets/icons')
        QDir.addSearchPath('icons', icons_dir)
        QGuiApplication.setWindowIcon(QIcon(QPixmap('icons:feeluown.png')))
        QApplication.setDesktopFileName('FeelUOwn')
        QApplication.instance().setQuitOnLastWindowClosed(not config.ENABLE_TRAY)
        QApplication.instance().setApplicationName('FeelUOwn')
        if sys.platform == 'win32':
            font = QApplication.font()
            font.setFamilies(['Segoe UI Symbol', 'Microsoft YaHei', 'sans-serif'])
            font.setPixelSize(13)
            QApplication.setFont(font)
        QWidget.__init__(self)
        App.__init__(self, *args, **kwargs)
        GuiApp.__q_app = QApplication.instance()
        self.setObjectName('app')
        self.coll_mgr = CollectionManager(self)
        self.theme_mgr = ThemeManager(self, parent=self)
        self.tips_mgr = TipsManager(self)
        self.hotkey_mgr = HotkeyManager(self)
        self.img_mgr = ImgManager(self)
        self.watch_mgr = WatchManager(self)
        self.pvd_ui_mgr = self.pvd_uimgr = ProviderUiManager(self)
        self.current_pvd_ui_mgr = CurrentProviderUiManager(self)
        self.pl_uimgr = PlaylistUiManager(self)
        self.mymusic_uimgr = MyMusicUiManager(self)
        self.browser = Browser(self)
        self.ui = Ui(self)
        if self.config.ENABLE_TRAY:
            self.tray = Tray(self)
        self.show_msg = self.ui._message_line.show_msg

    def initialize(self):
        if False:
            return 10
        super().initialize()
        self.hotkey_mgr.initialize()
        self.theme_mgr.initialize()
        if self.config.ENABLE_TRAY:
            self.tray.initialize()
            self.tray.show()
        self.coll_mgr.scan()
        self.watch_mgr.initialize()
        self.browser.initialize()
        QApplication.instance().aboutToQuit.connect(self.about_to_exit)

    def run(self):
        if False:
            i = 10
            return i + 15
        self.show()
        super().run()

    def apply_state(self, state):
        if False:
            while True:
                i = 10
        super().apply_state(state)
        coll_library = self.coll_mgr.get_coll_library()
        self.browser.goto(page=f'/colls/{coll_library.identifier}')
        gui = state.get('gui', {})
        lyric = gui.get('lyric', {})
        self.ui.lyric_window.apply_state(lyric)

    def dump_state(self):
        if False:
            while True:
                i = 10
        state = super().dump_state()
        state['gui'] = {'lyric': self.ui.lyric_window.dump_state()}
        return state

    def closeEvent(self, _):
        if False:
            return 10
        if not self.config.ENABLE_TRAY:
            self.exit()

    def mouseReleaseEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        if not self.rect().contains(e.pos()):
            return
        if e.button() == Qt.BackButton:
            self.browser.back()
        elif e.button() == Qt.ForwardButton:
            self.browser.forward()

    def exit_player(self):
        if False:
            for i in range(10):
                print('nop')
        self.ui.mpv_widget.shutdown()
        super().exit_player()

    def about_to_exit(self):
        if False:
            for i in range(10):
                print('nop')
        super().about_to_exit()
        QApplication.instance().aboutToQuit.disconnect(self.about_to_exit)

    def exit(self):
        if False:
            while True:
                i = 10
        QApplication.exit()