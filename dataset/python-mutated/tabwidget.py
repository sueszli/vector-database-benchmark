"""The tab widget used for TabbedBrowser from browser.py."""
import functools
import contextlib
import dataclasses
from typing import Optional, Dict, Any
from qutebrowser.qt.core import pyqtSignal, pyqtSlot, Qt, QSize, QRect, QPoint, QTimer, QUrl
from qutebrowser.qt.widgets import QTabWidget, QTabBar, QSizePolicy, QProxyStyle, QStyle, QStylePainter, QStyleOptionTab, QCommonStyle
from qutebrowser.qt.gui import QIcon, QPalette, QColor
from qutebrowser.utils import qtutils, objreg, utils, usertypes, log
from qutebrowser.config import config, stylesheet
from qutebrowser.misc import objects, debugcachestats
from qutebrowser.browser import browsertab

class TabWidget(QTabWidget):
    """The tab widget used for TabbedBrowser.

    Signals:
        tab_index_changed: Emitted when the current tab was changed.
                           arg 0: The index of the tab which is now focused.
                           arg 1: The total count of tabs.
        new_tab_requested: Emitted when a new tab is requested.
    """
    tab_index_changed = pyqtSignal(int, int)
    new_tab_requested = pyqtSignal('QUrl', bool, bool)
    MUTE_STRING = '[M] '
    AUDIBLE_STRING = '[A] '

    def __init__(self, win_id, parent=None):
        if False:
            return 10
        super().__init__(parent)
        bar = TabBar(win_id, self)
        self.setStyle(TabBarStyle())
        self.setTabBar(bar)
        bar.tabCloseRequested.connect(self.tabCloseRequested)
        bar.tabMoved.connect(functools.partial(QTimer.singleShot, 0, self.update_tab_titles))
        bar.currentChanged.connect(self._on_current_changed)
        bar.new_tab_requested.connect(self._on_new_tab_requested)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setDocumentMode(True)
        self.setUsesScrollButtons(True)
        bar.setDrawBase(False)
        self._init_config()
        config.instance.changed.connect(self._init_config)

    @config.change_filter('tabs')
    def _init_config(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize attributes based on the config.'
        self.setMovable(True)
        self.setTabsClosable(False)
        position = config.val.tabs.position
        selection_behavior = config.val.tabs.select_on_remove
        self.setTabPosition(position)
        self.setElideMode(config.val.tabs.title.elide)
        tabbar = self.tab_bar()
        tabbar.vertical = position in [QTabWidget.TabPosition.West, QTabWidget.TabPosition.East]
        tabbar.setSelectionBehaviorOnRemove(selection_behavior)
        tabbar.refresh()

    def tab_bar(self) -> 'TabBar':
        if False:
            while True:
                i = 10
        'Get the TabBar for this TabWidget.'
        bar = self.tabBar()
        assert isinstance(bar, TabBar), bar
        return bar

    def _tab_by_idx(self, idx: int) -> Optional[browsertab.AbstractTab]:
        if False:
            print('Hello World!')
        'Get the tab at the given index.'
        tab = self.widget(idx)
        if tab is not None:
            assert isinstance(tab, browsertab.AbstractTab), tab
        return tab

    def set_tab_indicator_color(self, idx, color):
        if False:
            while True:
                i = 10
        'Set the tab indicator color.\n\n        Args:\n            idx: The tab index.\n            color: A QColor.\n        '
        bar = self.tab_bar()
        bar.set_tab_data(idx, 'indicator-color', color)
        bar.update(bar.tabRect(idx))

    def tab_indicator_color(self, idx):
        if False:
            print('Hello World!')
        'Get the tab indicator color for the given index.'
        return self.tab_bar().tab_indicator_color(idx)

    def set_page_title(self, idx, title):
        if False:
            print('Hello World!')
        'Set the tab title user data.'
        tabbar = self.tab_bar()
        if config.cache['tabs.tooltips']:
            tabbar.setTabToolTip(idx, title)
        tabbar.set_tab_data(idx, 'page-title', title)
        self.update_tab_title(idx)

    def page_title(self, idx):
        if False:
            while True:
                i = 10
        'Get the tab title user data.'
        return self.tab_bar().page_title(idx)

    def update_tab_title(self, idx, field=None):
        if False:
            return 10
        'Update the tab text for the given tab.\n\n        Args:\n            idx: The tab index to update.\n            field: A field name which was updated. If given, the title\n                   is only set if the given field is in the template.\n        '
        assert idx != -1
        tab = self._tab_by_idx(idx)
        assert tab is not None
        if tab.data.pinned:
            fmt = config.cache['tabs.title.format_pinned']
        else:
            fmt = config.cache['tabs.title.format']
        if field is not None and (fmt is None or '{' + field + '}' not in fmt):
            return

        def right_align(num):
            if False:
                print('Hello World!')
            return str(num).rjust(len(str(self.count())))

        def left_align(num):
            if False:
                for i in range(10):
                    print('nop')
            return str(num).ljust(len(str(self.count())))
        bar = self.tab_bar()
        cur_idx = bar.currentIndex()
        if idx == cur_idx:
            rel_idx = left_align(idx + 1) + ' '
        else:
            rel_idx = ' ' + right_align(abs(idx - cur_idx))
        fields = self.get_tab_fields(idx)
        fields['current_title'] = fields['current_title'].replace('&', '&&')
        fields['index'] = idx + 1
        fields['aligned_index'] = right_align(idx + 1)
        fields['relative_index'] = rel_idx
        title = '' if fmt is None else fmt.format(**fields)
        if bar.tabText(idx) != title:
            bar.setTabText(idx, title)

    def get_tab_fields(self, idx):
        if False:
            return 10
        'Get the tab field data.'
        tab = self._tab_by_idx(idx)
        assert tab is not None
        page_title = self.page_title(idx)
        fields: Dict[str, Any] = {}
        fields['id'] = tab.tab_id
        fields['current_title'] = page_title
        fields['title_sep'] = ' - ' if page_title else ''
        fields['perc_raw'] = tab.progress()
        fields['backend'] = objects.backend.name
        fields['private'] = ' [Private Mode] ' if tab.is_private else ''
        try:
            if tab.audio.is_muted():
                fields['audio'] = TabWidget.MUTE_STRING
            elif tab.audio.is_recently_audible():
                fields['audio'] = TabWidget.AUDIBLE_STRING
            else:
                fields['audio'] = ''
        except browsertab.WebTabError:
            fields['audio'] = ''
        if tab.load_status() == usertypes.LoadStatus.loading:
            fields['perc'] = '[{}%] '.format(tab.progress())
        else:
            fields['perc'] = ''
        try:
            url = self.tab_url(idx)
        except qtutils.QtValueError:
            fields['host'] = ''
            fields['current_url'] = ''
            fields['protocol'] = ''
        else:
            fields['host'] = url.host()
            fields['current_url'] = url.toDisplayString()
            fields['protocol'] = url.scheme()
        y = tab.scroller.pos_perc()[1]
        if y <= 0:
            scroll_pos = 'top'
        elif y >= 100:
            scroll_pos = 'bot'
        else:
            scroll_pos = '{:2}%'.format(y)
        fields['scroll_pos'] = scroll_pos
        return fields

    @contextlib.contextmanager
    def _toggle_visibility(self):
        if False:
            while True:
                i = 10
        "Toggle visibility while running.\n\n        Every single call to setTabText calls the size hinting functions for\n        every single tab, which are slow. Since we know we are updating all\n        the tab's titles, we can delay this processing by making the tab\n        non-visible. To avoid flickering, disable repaint updates while we\n        work.\n        "
        bar = self.tab_bar()
        toggle = self.count() > 10 and (not bar.drag_in_progress) and bar.isVisible()
        if toggle:
            bar.setUpdatesEnabled(False)
            bar.setVisible(False)
        yield
        if toggle:
            bar.setVisible(True)
            bar.setUpdatesEnabled(True)

    def update_tab_titles(self):
        if False:
            while True:
                i = 10
        'Update all texts.'
        with self._toggle_visibility():
            for idx in range(self.count()):
                self.update_tab_title(idx)

    def tabInserted(self, idx):
        if False:
            while True:
                i = 10
        'Update titles when a tab was inserted.'
        super().tabInserted(idx)
        self.update_tab_titles()

    def tabRemoved(self, idx):
        if False:
            i = 10
            return i + 15
        'Update titles when a tab was removed.'
        super().tabRemoved(idx)
        self.update_tab_titles()

    def addTab(self, page, icon_or_text, text_or_empty=None):
        if False:
            print('Hello World!')
        "Override addTab to use our own text setting logic.\n\n        Unfortunately QTabWidget::addTab has these two overloads:\n            - QWidget * page, const QIcon & icon, const QString & label\n            - QWidget * page, const QString & label\n\n        This means we'll get different arguments based on the chosen overload.\n\n        Args:\n            page: The QWidget to add.\n            icon_or_text: Either the QIcon to add or the label.\n            text_or_empty: Either the label or None.\n\n        Return:\n            The index of the newly added tab.\n        "
        if text_or_empty is None:
            text = icon_or_text
            new_idx = super().addTab(page, '')
        else:
            icon = icon_or_text
            text = text_or_empty
            new_idx = super().addTab(page, icon, '')
        self.set_page_title(new_idx, text)
        return new_idx

    def insertTab(self, idx, page, icon_or_text, text_or_empty=None):
        if False:
            print('Hello World!')
        "Override insertTab to use our own text setting logic.\n\n        Unfortunately QTabWidget::insertTab has these two overloads:\n            - int index, QWidget * page, const QIcon & icon,\n              const QString & label\n            - int index, QWidget * page, const QString & label\n\n        This means we'll get different arguments based on the chosen overload.\n\n        Args:\n            idx: Where to insert the widget.\n            page: The QWidget to add.\n            icon_or_text: Either the QIcon to add or the label.\n            text_or_empty: Either the label or None.\n\n        Return:\n            The index of the newly added tab.\n        "
        if text_or_empty is None:
            text = icon_or_text
            new_idx = super().insertTab(idx, page, '')
        else:
            icon = icon_or_text
            text = text_or_empty
            new_idx = super().insertTab(idx, page, icon, '')
        self.set_page_title(new_idx, text)
        return new_idx

    @pyqtSlot(int)
    def _on_current_changed(self, index):
        if False:
            for i in range(10):
                print('nop')
        'Emit the tab_index_changed signal if the current tab changed.'
        self.tab_bar().on_current_changed()
        self.update_tab_titles()
        self.tab_index_changed.emit(index, self.count())

    @pyqtSlot()
    def _on_new_tab_requested(self):
        if False:
            return 10
        'Open a new tab.'
        self.new_tab_requested.emit(config.val.url.default_page, False, False)

    def tab_url(self, idx):
        if False:
            return 10
        'Get the URL of the tab at the given index.\n\n        Return:\n            The tab URL as QUrl.\n        '
        tab = self._tab_by_idx(idx)
        url = QUrl() if tab is None else tab.url()
        qtutils.ensure_valid(url)
        return url

    def update_tab_favicon(self, tab: browsertab.AbstractTab) -> None:
        if False:
            return 10
        'Update favicon of the given tab.'
        idx = self.indexOf(tab)
        icon = tab.icon() if tab.data.should_show_icon() else QIcon()
        self.setTabIcon(idx, icon)
        if config.val.tabs.tabs_are_windows:
            window = self.window()
            assert window is not None
            window.setWindowIcon(tab.icon())

    def setTabIcon(self, idx: int, icon: QIcon) -> None:
        if False:
            print('Hello World!')
        'Always show tab icons for pinned tabs in some circumstances.'
        tab = self._tab_by_idx(idx)
        if icon.isNull() and config.cache['tabs.favicons.show'] != 'never' and config.cache['tabs.pinned.shrink'] and (not self.tab_bar().vertical) and (tab is not None) and tab.data.pinned:
            style = self.style()
            assert style is not None
            icon = style.standardIcon(QStyle.StandardPixmap.SP_FileIcon)
        super().setTabIcon(idx, icon)

class TabBar(QTabBar):
    """Custom tab bar with our own style.

    FIXME: Dragging tabs doesn't look as nice as it does in QTabBar.  However,
    fixing this would be a lot of effort, so we'll postpone it until we're
    reimplementing drag&drop for other reasons.

    https://github.com/qutebrowser/qutebrowser/issues/126

    Attributes:
        vertical: When the tab bar is currently vertical.
        win_id: The window ID this TabBar belongs to.

    Signals:
        new_tab_requested: Emitted when a new tab is requested.
    """
    STYLESHEET = '\n        TabBar {\n            font: {{ conf.fonts.tabs.unselected }};\n            background-color: {{ conf.colors.tabs.bar.bg }};\n        }\n\n        TabBar::tab:selected {\n            font: {{ conf.fonts.tabs.selected }};\n        }\n    '
    new_tab_requested = pyqtSignal()

    def __init__(self, win_id, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._win_id = win_id
        self._our_style = TabBarStyle()
        self.setStyle(self._our_style)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.vertical = False
        self._auto_hide_timer = QTimer()
        self._auto_hide_timer.setSingleShot(True)
        self._auto_hide_timer.timeout.connect(self.maybe_hide)
        self._on_show_switching_delay_changed()
        self.setAutoFillBackground(True)
        self.drag_in_progress: bool = False
        stylesheet.set_register(self)
        self.ensurePolished()
        config.instance.changed.connect(self._on_config_changed)
        self._set_icon_size()
        QTimer.singleShot(0, self.maybe_hide)
        self._minimum_tab_size_hint_helper = functools.lru_cache(maxsize=2 ** 9)(self._minimum_tab_size_hint_helper_uncached)
        debugcachestats.register(name=f'tab width cache (win_id={win_id})')(self._minimum_tab_size_hint_helper)
        self._minimum_tab_height = functools.lru_cache(maxsize=1)(self._minimum_tab_height_uncached)

    def __repr__(self):
        if False:
            return 10
        return utils.get_repr(self, count=self.count())

    def _tab_widget(self):
        if False:
            for i in range(10):
                print('nop')
        "Get the TabWidget we're in."
        parent = self.parent()
        assert isinstance(parent, TabWidget), parent
        return parent

    def _current_tab(self):
        if False:
            i = 10
            return i + 15
        'Get the current tab object.'
        return self._tab_widget().currentWidget()

    @pyqtSlot(str)
    def _on_config_changed(self, option: str) -> None:
        if False:
            i = 10
            return i + 15
        if option.startswith('fonts.tabs.'):
            self.ensurePolished()
            self._set_icon_size()
        elif option == 'tabs.favicons.scale':
            self._set_icon_size()
        elif option == 'tabs.show_switching_delay':
            self._on_show_switching_delay_changed()
        elif option == 'tabs.show':
            self.maybe_hide()
        if option.startswith('colors.tabs.'):
            self.update()
        if option in ['tabs.indicator.padding', 'tabs.padding', 'tabs.indicator.width', 'tabs.min_width', 'tabs.pinned.shrink', 'fonts.tabs.selected', 'fonts.tabs.unselected']:
            self._minimum_tab_size_hint_helper.cache_clear()
            self._minimum_tab_height.cache_clear()

    def _on_show_switching_delay_changed(self):
        if False:
            for i in range(10):
                print('nop')
        'Set timer interval when tabs.show_switching_delay got changed.'
        self._auto_hide_timer.setInterval(config.val.tabs.show_switching_delay)

    def on_current_changed(self):
        if False:
            return 10
        'Show tab bar when current tab got changed.'
        self.maybe_hide()
        if config.val.tabs.show == 'switching':
            self.show()
            self._auto_hide_timer.start()

    @pyqtSlot()
    def maybe_hide(self):
        if False:
            while True:
                i = 10
        'Hide the tab bar if needed.'
        show = config.val.tabs.show
        tab = self._current_tab()
        if show in ['never', 'switching'] or (show == 'multiple' and self.count() == 1) or (tab and tab.data.fullscreen):
            self.hide()
        else:
            self.show()

    def set_tab_data(self, idx, key, value):
        if False:
            print('Hello World!')
        'Set tab data as a dictionary.'
        if not 0 <= idx < self.count():
            raise IndexError('Tab index ({}) out of range ({})!'.format(idx, self.count()))
        data = self.tabData(idx)
        if data is None:
            data = {}
        data[key] = value
        self.setTabData(idx, data)

    def tab_data(self, idx, key):
        if False:
            return 10
        'Get tab data for a given key.'
        if not 0 <= idx < self.count():
            raise IndexError('Tab index ({}) out of range ({})!'.format(idx, self.count()))
        data = self.tabData(idx)
        if data is None:
            data = {}
        return data[key]

    def tab_indicator_color(self, idx):
        if False:
            return 10
        'Get the tab indicator color for the given index.'
        try:
            return self.tab_data(idx, 'indicator-color')
        except KeyError:
            return QColor()

    def page_title(self, idx):
        if False:
            while True:
                i = 10
        'Get the tab title user data.\n\n        Args:\n            idx: The tab index to get the title for.\n        '
        try:
            return self.tab_data(idx, 'page-title')
        except KeyError:
            return ''

    def refresh(self):
        if False:
            return 10
        'Properly repaint the tab bar and relayout tabs.'
        self.setIconSize(self.iconSize())

    def _set_icon_size(self):
        if False:
            return 10
        'Set the tab bar favicon size.'
        size = self.fontMetrics().height() - 2
        size = int(size * config.val.tabs.favicons.scale)
        self.setIconSize(QSize(size, size))

    def mouseReleaseEvent(self, e):
        if False:
            return 10
        'Override mouseReleaseEvent to know when drags stop.'
        self.drag_in_progress = False
        super().mouseReleaseEvent(e)

    def mousePressEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        'Override mousePressEvent to close tabs if configured.\n\n        Also keep track of if we are currently in a drag.'
        self.drag_in_progress = True
        button = config.val.tabs.close_mouse_button
        if e.button() == Qt.MouseButton.RightButton and button == 'right' or (e.button() == Qt.MouseButton.MiddleButton and button == 'middle'):
            e.accept()
            idx = self.tabAt(e.pos())
            if idx == -1:
                action = config.val.tabs.close_mouse_button_on_bar
                if action == 'ignore':
                    return
                elif action == 'new-tab':
                    self.new_tab_requested.emit()
                    return
                elif action == 'close-current':
                    idx = self.currentIndex()
                elif action == 'close-last':
                    idx = self.count() - 1
            self.tabCloseRequested.emit(idx)
            return
        super().mousePressEvent(e)

    def minimumTabSizeHint(self, index: int, ellipsis: bool=True) -> QSize:
        if False:
            return 10
        "Set the minimum tab size to indicator/icon/... text.\n\n        Args:\n            index: The index of the tab to get a size hint for.\n            ellipsis: Whether to use ellipsis to calculate width\n                      instead of the tab's text.\n                      Forced to False for pinned tabs.\n        Return:\n            A QSize of the smallest tab size we can make.\n        "
        icon = self.tabIcon(index)
        if icon.isNull():
            icon_width = 0
        else:
            icon_width = min(icon.actualSize(self.iconSize()).width(), self.iconSize().width()) + TabBarStyle.ICON_PADDING
        pinned = self._tab_pinned(index)
        if not self.vertical and pinned and config.val.tabs.pinned.shrink:
            ellipsis = False
        return self._minimum_tab_size_hint_helper(self.tabText(index), icon_width, ellipsis, pinned)

    def _minimum_tab_size_hint_helper_uncached(self, tab_text: str, icon_width: int, ellipsis: bool, pinned: bool) -> QSize:
        if False:
            print('Hello World!')
        'Helper function to cache tab results.\n\n        Config values accessed in here should be added to _on_config_changed to\n        ensure cache is flushed when needed.\n        '
        text = '…' if ellipsis else tab_text

        def _text_to_width(text):
            if False:
                for i in range(10):
                    print('nop')
            return self.fontMetrics().size(Qt.TextFlag.TextShowMnemonic, text).width()
        text_width = min(_text_to_width(text), _text_to_width(tab_text))
        padding = config.cache['tabs.padding']
        indicator_width = config.cache['tabs.indicator.width']
        indicator_padding = config.cache['tabs.indicator.padding']
        padding_h = padding.left + padding.right
        if indicator_width != 0:
            padding_h += indicator_padding.left + indicator_padding.right
        height = self._minimum_tab_height()
        width = text_width + icon_width + padding_h + indicator_width
        min_width = config.cache['tabs.min_width']
        if not self.vertical and min_width > 0 and (not pinned) or not config.cache['tabs.pinned.shrink']:
            width = max(min_width, width)
        return QSize(width, height)

    def _minimum_tab_height_uncached(self):
        if False:
            for i in range(10):
                print('nop')
        padding = config.cache['tabs.padding']
        return self.fontMetrics().height() + padding.top + padding.bottom

    def _tab_pinned(self, index: int) -> bool:
        if False:
            while True:
                i = 10
        'Return True if tab is pinned.'
        if not 0 <= index < self.count():
            raise IndexError('Tab index ({}) out of range ({})!'.format(index, self.count()))
        widget = self._tab_widget().widget(index)
        if widget is None:
            return False
        return widget.data.pinned

    def tabSizeHint(self, index: int) -> QSize:
        if False:
            return 10
        "Override tabSizeHint to customize qb's tab size.\n\n        https://wiki.python.org/moin/PyQt/Customising%20tab%20bars\n\n        Args:\n            index: The index of the tab.\n\n        Return:\n            A QSize.\n        "
        if self.count() == 0:
            return QSize()
        height = self._minimum_tab_height()
        if self.vertical:
            confwidth = str(config.cache['tabs.width'])
            if confwidth.endswith('%'):
                main_window = objreg.get('main-window', scope='window', window=self._win_id)
                perc = int(confwidth.rstrip('%'))
                width = main_window.width() * perc // 100
            else:
                width = int(confwidth)
            size = QSize(width, height)
        else:
            if config.cache['tabs.pinned.shrink'] and self._tab_pinned(index):
                width = self.minimumTabSizeHint(index, ellipsis=False).width()
            else:
                width = max(self.width(), 10)
                max_width = config.cache['tabs.max_width']
                if max_width > 0:
                    width = min(max_width, width)
            size = QSize(width, height)
        qtutils.ensure_valid(size)
        return size

    def initStyleOption(self, opt, idx):
        if False:
            while True:
                i = 10
        'Override QTabBar.initStyleOption().\n\n        Used to calculate styling clues from a widget for the GUI layer.\n        '
        super().initStyleOption(opt, idx)
        text_rect = self._our_style.subElementRect(QStyle.SubElement.SE_TabBarTabText, opt, self)
        opt.text = self.fontMetrics().elidedText(self.tabText(idx), self.elideMode(), text_rect.width(), Qt.TextFlag.TextShowMnemonic)

    def paintEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Override paintEvent to draw the tabs like we want to.'
        p = QStylePainter(self)
        selected = self.currentIndex()
        for idx in range(self.count()):
            if not event.region().intersects(self.tabRect(idx)):
                continue
            tab = QStyleOptionTab()
            self.initStyleOption(tab, idx)
            setting = 'colors.tabs'
            if self._tab_pinned(idx):
                setting += '.pinned'
            if idx == selected:
                setting += '.selected'
            setting += '.odd' if (idx + 1) % 2 else '.even'
            tab.palette.setColor(QPalette.ColorRole.Window, config.cache[setting + '.bg'])
            tab.palette.setColor(QPalette.ColorRole.WindowText, config.cache[setting + '.fg'])
            indicator_color = self.tab_indicator_color(idx)
            tab.palette.setColor(QPalette.ColorRole.Base, indicator_color)
            p.drawControl(QStyle.ControlElement.CE_TabBarTab, tab)

    def tabInserted(self, idx):
        if False:
            return 10
        'Update visibility when a tab was inserted.'
        super().tabInserted(idx)
        self.maybe_hide()

    def tabRemoved(self, idx):
        if False:
            return 10
        'Update visibility when a tab was removed.'
        super().tabRemoved(idx)
        self.maybe_hide()

    def wheelEvent(self, e):
        if False:
            print('Hello World!')
        'Override wheelEvent to make the action configurable.\n\n        Args:\n            e: The QWheelEvent\n        '
        if config.val.tabs.mousewheel_switching:
            if utils.is_mac:
                index = self.currentIndex()
                if index == -1:
                    return
                dx = e.angleDelta().x()
                dy = e.angleDelta().y()
                delta = dx if abs(dx) > abs(dy) else dy
                offset = -1 if delta > 0 else 1
                index += offset
                if 0 <= index < self.count():
                    self.setCurrentIndex(index)
            else:
                super().wheelEvent(e)
        else:
            tabbed_browser = objreg.get('tabbed-browser', scope='window', window=self._win_id)
            tabbed_browser.wheelEvent(e)

@dataclasses.dataclass
class Layouts:
    """Layout information for tab.

    Used by TabBarStyle._tab_layout().
    """
    text: QRect
    icon: QRect
    indicator: QRect

class TabBarStyle(QProxyStyle):
    """Qt style used by TabBar to fix some issues with the default one.

    This fixes the following things:
        - Remove the focus rectangle Ubuntu draws on tabs.
        - Force text to be left-aligned even though Qt has "centered"
          hardcoded.
    """
    ICON_PADDING = 4

    def __init__(self, style=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(style)

    def _base_style(self) -> QStyle:
        if False:
            print('Hello World!')
        'Get the base style.'
        style = self.baseStyle()
        assert style is not None
        return style

    def _draw_indicator(self, layouts, opt, p):
        if False:
            print('Hello World!')
        'Draw the tab indicator.\n\n        Args:\n            layouts: The layouts from _tab_layout.\n            opt: QStyleOption from drawControl.\n            p: QPainter from drawControl.\n        '
        color = opt.palette.base().color()
        rect = layouts.indicator
        if color.isValid() and rect.isValid():
            p.fillRect(rect, color)

    def _draw_icon(self, layouts, opt, p):
        if False:
            return 10
        'Draw the tab icon.\n\n        Args:\n            layouts: The layouts from _tab_layout.\n            opt: QStyleOption\n            p: QPainter\n        '
        qtutils.ensure_valid(layouts.icon)
        icon_mode = QIcon.Mode.Normal if opt.state & QStyle.StateFlag.State_Enabled else QIcon.Mode.Disabled
        icon_state = QIcon.State.On if opt.state & QStyle.StateFlag.State_Selected else QIcon.State.Off
        icon = opt.icon.pixmap(opt.iconSize, icon_mode, icon_state)
        self._base_style().drawItemPixmap(p, layouts.icon, Qt.AlignmentFlag.AlignCenter, icon)

    def drawControl(self, element, opt, p, widget=None):
        if False:
            print('Hello World!')
        'Override drawControl to draw odd tabs in a different color.\n\n        Draws the given element with the provided painter with the style\n        options specified by option.\n\n        Args:\n            element: ControlElement\n            opt: QStyleOption\n            p: QPainter\n            widget: QWidget\n        '
        if element not in [QStyle.ControlElement.CE_TabBarTab, QStyle.ControlElement.CE_TabBarTabShape, QStyle.ControlElement.CE_TabBarTabLabel]:
            self._base_style().drawControl(element, opt, p, widget)
            return
        layouts = self._tab_layout(opt)
        if layouts is None:
            log.misc.warning('Could not get layouts for tab!')
            return
        if element == QStyle.ControlElement.CE_TabBarTab:
            self.drawControl(QStyle.ControlElement.CE_TabBarTabShape, opt, p, widget)
            self.drawControl(QStyle.ControlElement.CE_TabBarTabLabel, opt, p, widget)
        elif element == QStyle.ControlElement.CE_TabBarTabShape:
            p.fillRect(opt.rect, opt.palette.window())
            self._draw_indicator(layouts, opt, p)
            QCommonStyle.drawControl(self, QStyle.ControlElement.CE_TabBarTabShape, opt, p, widget)
        elif element == QStyle.ControlElement.CE_TabBarTabLabel:
            if not opt.icon.isNull() and layouts.icon.isValid():
                self._draw_icon(layouts, opt, p)
            alignment = config.cache['tabs.title.alignment'] | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextHideMnemonic
            self._base_style().drawItemText(p, layouts.text, int(alignment), opt.palette, bool(opt.state & QStyle.StateFlag.State_Enabled), opt.text, QPalette.ColorRole.WindowText)
        else:
            raise ValueError('Invalid element {!r}'.format(element))

    def pixelMetric(self, metric, option=None, widget=None):
        if False:
            while True:
                i = 10
        'Override pixelMetric to not shift the selected tab.\n\n        Args:\n            metric: PixelMetric\n            option: const QStyleOption *\n            widget: const QWidget *\n\n        Return:\n            An int.\n        '
        if metric in [QStyle.PixelMetric.PM_TabBarTabShiftHorizontal, QStyle.PixelMetric.PM_TabBarTabShiftVertical, QStyle.PixelMetric.PM_TabBarTabHSpace, QStyle.PixelMetric.PM_TabBarTabVSpace, QStyle.PixelMetric.PM_TabBarScrollButtonWidth]:
            return 0
        else:
            return self._base_style().pixelMetric(metric, option, widget)

    def subElementRect(self, sr, opt, widget=None):
        if False:
            i = 10
            return i + 15
        'Override subElementRect to use our own _tab_layout implementation.\n\n        Args:\n            sr: SubElement\n            opt: QStyleOption\n            widget: QWidget\n\n        Return:\n            A QRect.\n        '
        if sr == QStyle.SubElement.SE_TabBarTabText:
            layouts = self._tab_layout(opt)
            if layouts is None:
                log.misc.warning('Could not get layouts for tab!')
                return QRect()
            return layouts.text
        elif sr in [QStyle.SubElement.SE_TabWidgetTabBar, QStyle.SubElement.SE_TabBarScrollLeftButton]:
            return QCommonStyle.subElementRect(self, sr, opt, widget)
        else:
            return self._base_style().subElementRect(sr, opt, widget)

    def _tab_layout(self, opt):
        if False:
            for i in range(10):
                print('nop')
        "Compute the text/icon rect from the opt rect.\n\n        This is based on Qt's QCommonStylePrivate::tabLayout\n        (qtbase/src/widgets/styles/qcommonstyle.cpp) as we can't use the\n        private implementation.\n\n        Args:\n            opt: QStyleOptionTab\n\n        Return:\n            A Layout object with two QRects.\n        "
        padding = config.cache['tabs.padding']
        indicator_padding = config.cache['tabs.indicator.padding']
        text_rect = QRect(opt.rect)
        if not text_rect.isValid():
            return None
        text_rect.adjust(padding.left, padding.top, -padding.right, -padding.bottom)
        indicator_width = config.cache['tabs.indicator.width']
        if indicator_width == 0:
            indicator_rect = QRect()
        else:
            indicator_rect = QRect(opt.rect)
            qtutils.ensure_valid(indicator_rect)
            indicator_rect.adjust(padding.left + indicator_padding.left, padding.top + indicator_padding.top, 0, -(padding.bottom + indicator_padding.bottom))
            indicator_rect.setWidth(indicator_width)
            text_rect.adjust(indicator_width + indicator_padding.left + indicator_padding.right, 0, 0, 0)
        icon_rect = self._get_icon_rect(opt, text_rect)
        if icon_rect.isValid():
            text_rect.adjust(icon_rect.width() + TabBarStyle.ICON_PADDING, 0, 0, 0)
        text_rect = self._base_style().visualRect(opt.direction, opt.rect, text_rect)
        return Layouts(text=text_rect, icon=icon_rect, indicator=indicator_rect)

    def _get_icon_rect(self, opt, text_rect):
        if False:
            for i in range(10):
                print('nop')
        'Get a QRect for the icon to draw.\n\n        Args:\n            opt: QStyleOptionTab\n            text_rect: The QRect for the text.\n\n        Return:\n            A QRect.\n        '
        icon_size = opt.iconSize
        if not icon_size.isValid():
            icon_extent = self.pixelMetric(QStyle.PixelMetric.PM_SmallIconSize)
            icon_size = QSize(icon_extent, icon_extent)
        icon_mode = QIcon.Mode.Normal if opt.state & QStyle.StateFlag.State_Enabled else QIcon.Mode.Disabled
        icon_state = QIcon.State.On if opt.state & QStyle.StateFlag.State_Selected else QIcon.State.Off
        position = config.cache['tabs.position']
        if position in [QTabWidget.TabPosition.East, QTabWidget.TabPosition.West] and config.cache['tabs.favicons.show'] != 'never':
            tab_icon_size = icon_size
        else:
            actual_size = opt.icon.actualSize(icon_size, icon_mode, icon_state)
            tab_icon_size = QSize(min(actual_size.width(), icon_size.width()), min(actual_size.height(), icon_size.height()))
        icon_top = text_rect.center().y() + 1 - tab_icon_size.height() // 2
        icon_rect = QRect(QPoint(text_rect.left(), icon_top), tab_icon_size)
        icon_rect = self._base_style().visualRect(opt.direction, opt.rect, icon_rect)
        return icon_rect