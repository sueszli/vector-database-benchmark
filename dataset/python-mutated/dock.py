"""
Dock widgets for plugins
"""
import qstylizer.style
from qtpy.QtCore import QEvent, QObject, Qt, QSize, Signal
from qtpy.QtWidgets import QDockWidget, QHBoxLayout, QSizePolicy, QTabBar, QToolButton, QWidget
from spyder.api.translations import _
from spyder.api.config.mixins import SpyderConfigurationAccessor
from spyder.utils.icon_manager import ima
from spyder.utils.palette import QStylePalette
from spyder.utils.stylesheet import PanesToolbarStyleSheet, HORIZONTAL_DOCK_TABBAR_STYLESHEET, VERTICAL_DOCK_TABBAR_STYLESHEET

class TabFilter(QObject, SpyderConfigurationAccessor):
    """Filter event attached to each DockWidget QTabBar."""
    CONF_SECTION = 'main'

    def __init__(self, dock_tabbar, main):
        if False:
            for i in range(10):
                print('nop')
        QObject.__init__(self)
        self.dock_tabbar: QTabBar = dock_tabbar
        self.main = main
        self.from_index = None
        self._set_tabbar_stylesheet()
        self.dock_tabbar.setElideMode(Qt.ElideNone)
        self.dock_tabbar.setUsesScrollButtons(True)

    def eventFilter(self, obj, event):
        if False:
            while True:
                i = 10
        'Filter mouse press events.\n\n        Events that are captured and not propagated return True. Events that\n        are not captured and are propagated return False.\n        '
        event_type = event.type()
        if event_type == QEvent.MouseButtonPress:
            self.tab_pressed(event)
            return False
        return False

    def tab_pressed(self, event):
        if False:
            return 10
        'Method called when a tab from a QTabBar has been pressed.'
        self.from_index = self.dock_tabbar.tabAt(event.pos())
        self.dock_tabbar.setCurrentIndex(self.from_index)
        try:
            if event.button() == Qt.RightButton:
                if self.from_index == -1:
                    self.show_nontab_menu(event)
                else:
                    self.show_tab_menu(event)
        except AttributeError:
            pass

    def show_tab_menu(self, event):
        if False:
            while True:
                i = 10
        'Show the context menu assigned to tabs.'
        self.show_nontab_menu(event)

    def show_nontab_menu(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Show the context menu assigned to nontabs section.'
        menu = self.main.createPopupMenu()
        menu.exec_(self.dock_tabbar.mapToGlobal(event.pos()))

    def _set_tabbar_stylesheet(self):
        if False:
            i = 10
            return i + 15
        if self.get_conf('vertical_tabs'):
            self.dock_tabbar.setStyleSheet(str(VERTICAL_DOCK_TABBAR_STYLESHEET))
        else:
            self.dock_tabbar.setStyleSheet(str(HORIZONTAL_DOCK_TABBAR_STYLESHEET))

class DragButton(QToolButton):
    """
    Drag button for the title bar.

    This button pass all its mouse events to its parent.
    """

    def __init__(self, parent, button_size):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.parent = parent
        self.setIconSize(button_size)
        self.setAutoRaise(True)
        self.setIcon(ima.icon('drag_dock_widget'))
        self.setToolTip(_('Drag and drop pane to a different position'))
        self.setStyleSheet(self._stylesheet)

    def mouseReleaseEvent(self, event):
        if False:
            return 10
        self.parent.mouseReleaseEvent(event)

    def mousePressEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.parent.mousePressEvent(event)

    @property
    def _stylesheet(self):
        if False:
            print('Hello World!')
        css = qstylizer.style.StyleSheet()
        css.QToolButton.setValues(borderRadius='0px', border='0px')
        return css.toString()

class CloseButton(QToolButton):
    """Close button for the title bar."""

    def __init__(self, parent, button_size):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.parent = parent
        self.setIconSize(button_size)
        self.setAutoRaise(True)
        self.setIcon(ima.icon('lock_open'))
        self.setToolTip(_('Lock pane'))
        self._apply_stylesheet(QStylePalette.COLOR_BACKGROUND_3, 0)

    def _apply_stylesheet(self, bgcolor, bradius):
        if False:
            for i in range(10):
                print('nop')
        css = qstylizer.style.StyleSheet()
        css.QToolButton.setValues(width=PanesToolbarStyleSheet.BUTTON_WIDTH, borderRadius=f'{bradius}px', border='0px', backgroundColor=bgcolor)
        self.setStyleSheet(css.toString())

    def enterEvent(self, event):
        if False:
            i = 10
            return i + 15
        self.setCursor(Qt.ArrowCursor)
        self._apply_stylesheet(QStylePalette.COLOR_BACKGROUND_5, 3)
        self.parent._apply_stylesheet(QStylePalette.COLOR_BACKGROUND_3)
        self.setIcon(ima.icon('lock'))
        super().enterEvent(event)

    def mousePressEvent(self, event):
        if False:
            while True:
                i = 10
        self._apply_stylesheet(QStylePalette.COLOR_BACKGROUND_6, 3)
        super().mousePressEvent(event)

    def leaveEvent(self, event):
        if False:
            while True:
                i = 10
        self._apply_stylesheet(QStylePalette.COLOR_BACKGROUND_3, 0)
        self.parent._apply_stylesheet(QStylePalette.COLOR_BACKGROUND_5)
        self.setIcon(ima.icon('lock_open'))
        super().leaveEvent(event)

class DockTitleBar(QWidget):
    """
    Custom title bar for our dock widgets.

    Inspired from
    https://stackoverflow.com/a/40894225/438386
    """

    def __init__(self, parent):
        if False:
            i = 10
            return i + 15
        super(DockTitleBar, self).__init__(parent)
        button_size = QSize(20, 20)
        drag_button = DragButton(self, button_size)
        left_spacer = QWidget(self)
        left_spacer.setToolTip(drag_button.toolTip())
        left_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        right_spacer = QWidget(self)
        right_spacer.setToolTip(drag_button.toolTip())
        right_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        close_button = CloseButton(self, button_size)
        close_button.clicked.connect(parent.remove_title_bar)
        hlayout = QHBoxLayout(self)
        hlayout.setSpacing(0)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.addWidget(left_spacer)
        hlayout.addWidget(drag_button)
        hlayout.addWidget(right_spacer)
        hlayout.addWidget(close_button)
        self._apply_stylesheet(QStylePalette.COLOR_BACKGROUND_3)

    def mouseReleaseEvent(self, event):
        if False:
            while True:
                i = 10
        self.setCursor(Qt.OpenHandCursor)
        self._apply_stylesheet(QStylePalette.COLOR_BACKGROUND_5)
        QWidget.mouseReleaseEvent(self, event)

    def mousePressEvent(self, event):
        if False:
            return 10
        self.setCursor(Qt.ClosedHandCursor)
        self._apply_stylesheet(QStylePalette.COLOR_BACKGROUND_6)
        QWidget.mousePressEvent(self, event)

    def enterEvent(self, event):
        if False:
            i = 10
            return i + 15
        self.setCursor(Qt.OpenHandCursor)
        self._apply_stylesheet(QStylePalette.COLOR_BACKGROUND_5)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if False:
            return 10
        'Remove customizations when leaving widget.'
        self.unsetCursor()
        self._apply_stylesheet(QStylePalette.COLOR_BACKGROUND_3)
        super().leaveEvent(event)

    def _apply_stylesheet(self, bgcolor):
        if False:
            while True:
                i = 10
        css = qstylizer.style.StyleSheet()
        css.QWidget.setValues(height=PanesToolbarStyleSheet.BUTTON_HEIGHT, backgroundColor=bgcolor)
        self.setStyleSheet(css.toString())

class SpyderDockWidget(QDockWidget):
    """Subclass to override needed methods"""
    ALLOWED_AREAS = Qt.AllDockWidgetAreas
    LOCATION = Qt.LeftDockWidgetArea
    FEATURES = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable
    sig_plugin_closed = Signal()
    sig_title_bar_shown = Signal(bool)

    def __init__(self, title, parent):
        if False:
            for i in range(10):
                print('nop')
        super(SpyderDockWidget, self).__init__(title, parent)
        self.title = title
        self.setFeatures(self.FEATURES)
        self.main = parent
        self.empty_titlebar = QWidget(self)
        self.titlebar = DockTitleBar(self)
        self.dock_tabbar = None
        layout = QHBoxLayout(self.empty_titlebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.empty_titlebar.setLayout(layout)
        self.empty_titlebar.setMinimumSize(0, 0)
        self.empty_titlebar.setMaximumSize(0, 0)
        self.set_title_bar()
        self.remove_title_bar()
        self.visibilityChanged.connect(self.install_tab_event_filter)

    def closeEvent(self, event):
        if False:
            return 10
        '\n        Reimplement Qt method to send a signal on close so that "Panes" main\n        window menu can be updated correctly\n        '
        self.sig_plugin_closed.emit()

    def install_tab_event_filter(self, value):
        if False:
            print('Hello World!')
        '\n        Install an event filter to capture mouse events in the tabs of a\n        QTabBar holding tabified dockwidgets.\n        '
        dock_tabbar = None
        try:
            tabbars = self.main.findChildren(QTabBar)
        except RuntimeError:
            tabbars = []
        for tabbar in tabbars:
            for tab in range(tabbar.count()):
                title = tabbar.tabText(tab)
                if title == self.title:
                    dock_tabbar = tabbar
                    break
        if dock_tabbar is not None:
            self.dock_tabbar = dock_tabbar
            if getattr(self.dock_tabbar, 'filter', None) is None:
                self.dock_tabbar.filter = TabFilter(self.dock_tabbar, self.main)
                self.dock_tabbar.installEventFilter(self.dock_tabbar.filter)

    def remove_title_bar(self):
        if False:
            for i in range(10):
                print('nop')
        'Set empty qwidget on title bar.'
        self.sig_title_bar_shown.emit(False)
        self.setTitleBarWidget(self.empty_titlebar)

    def set_title_bar(self):
        if False:
            for i in range(10):
                print('nop')
        'Set custom title bar.'
        self.sig_title_bar_shown.emit(True)
        self.setTitleBarWidget(self.titlebar)