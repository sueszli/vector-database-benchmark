from enum import Enum
from string import Template
from typing import List, Union
import weakref
from PyQt5.QtCore import QFile, QObject, QEvent
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget
from .config import qconfig, Theme, isDarkTheme

class StyleSheetManager(QObject):
    """ Style sheet manager """

    def __init__(self):
        if False:
            return 10
        self.widgets = weakref.WeakKeyDictionary()

    def register(self, source, widget: QWidget, reset=True):
        if False:
            print('Hello World!')
        ' register widget to manager\n\n        Parameters\n        ----------\n        source: str | StyleSheetBase\n            qss source, it could be:\n            * `str`: qss file path\n            * `StyleSheetBase`: style sheet instance\n\n        widget: QWidget\n            the widget to set style sheet\n\n        reset: bool\n            whether to reset the qss source\n        '
        if isinstance(source, str):
            source = StyleSheetFile(source)
        if widget not in self.widgets:
            widget.destroyed.connect(self.deregister)
            widget.installEventFilter(CustomStyleSheetWatcher(widget))
            self.widgets[widget] = StyleSheetCompose([source, CustomStyleSheet(widget)])
        if not reset:
            self.source(widget).add(source)
        else:
            self.widgets[widget] = StyleSheetCompose([source, CustomStyleSheet(widget)])

    def deregister(self, widget: QWidget):
        if False:
            i = 10
            return i + 15
        ' deregister widget from manager '
        if widget not in self.widgets:
            return
        self.widgets.pop(widget)

    def items(self):
        if False:
            print('Hello World!')
        return self.widgets.items()

    def source(self, widget: QWidget):
        if False:
            i = 10
            return i + 15
        ' get the qss source of widget '
        return self.widgets.get(widget, StyleSheetCompose([]))
styleSheetManager = StyleSheetManager()

class QssTemplate(Template):
    """ style sheet template """
    delimiter = '--'

def applyThemeColor(qss: str):
    if False:
        print('Hello World!')
    ' apply theme color to style sheet\n\n    Parameters\n    ----------\n    qss: str\n        the style sheet string to apply theme color, the substituted variable\n        should be equal to the value of `ThemeColor` and starts width `--`, i.e `--ThemeColorPrimary`\n    '
    template = QssTemplate(qss)
    mappings = {c.value: c.name() for c in ThemeColor._member_map_.values()}
    return template.safe_substitute(mappings)

class StyleSheetBase:
    """ Style sheet base class """

    def path(self, theme=Theme.AUTO):
        if False:
            i = 10
            return i + 15
        ' get the path of style sheet '
        raise NotImplementedError

    def content(self, theme=Theme.AUTO):
        if False:
            return 10
        ' get the content of style sheet '
        return getStyleSheetFromFile(self.path(theme))

    def apply(self, widget: QWidget, theme=Theme.AUTO):
        if False:
            return 10
        ' apply style sheet to widget '
        setStyleSheet(widget, self, theme)

class FluentStyleSheet(StyleSheetBase, Enum):
    """ Fluent style sheet """
    MENU = 'menu'
    LABEL = 'label'
    PIVOT = 'pivot'
    BUTTON = 'button'
    DIALOG = 'dialog'
    SLIDER = 'slider'
    INFO_BAR = 'info_bar'
    SPIN_BOX = 'spin_box'
    TAB_VIEW = 'tab_view'
    TOOL_TIP = 'tool_tip'
    CHECK_BOX = 'check_box'
    COMBO_BOX = 'combo_box'
    FLIP_VIEW = 'flip_view'
    LINE_EDIT = 'line_edit'
    LIST_VIEW = 'list_view'
    TREE_VIEW = 'tree_view'
    INFO_BADGE = 'info_badge'
    PIPS_PAGER = 'pips_pager'
    TABLE_VIEW = 'table_view'
    CARD_WIDGET = 'card_widget'
    TIME_PICKER = 'time_picker'
    COLOR_DIALOG = 'color_dialog'
    MEDIA_PLAYER = 'media_player'
    SETTING_CARD = 'setting_card'
    TEACHING_TIP = 'teaching_tip'
    FLUENT_WINDOW = 'fluent_window'
    SWITCH_BUTTON = 'switch_button'
    MESSAGE_DIALOG = 'message_dialog'
    STATE_TOOL_TIP = 'state_tool_tip'
    CALENDAR_PICKER = 'calendar_picker'
    FOLDER_LIST_DIALOG = 'folder_list_dialog'
    SETTING_CARD_GROUP = 'setting_card_group'
    EXPAND_SETTING_CARD = 'expand_setting_card'
    NAVIGATION_INTERFACE = 'navigation_interface'

    def path(self, theme=Theme.AUTO):
        if False:
            i = 10
            return i + 15
        theme = qconfig.theme if theme == Theme.AUTO else theme
        return f':/qfluentwidgets/qss/{theme.value.lower()}/{self.value}.qss'

class StyleSheetFile(StyleSheetBase):
    """ Style sheet file """

    def __init__(self, path: str):
        if False:
            print('Hello World!')
        super().__init__()
        self.filePath = path

    def path(self, theme=Theme.AUTO):
        if False:
            print('Hello World!')
        return self.filePath

class CustomStyleSheet(StyleSheetBase):
    """ Custom style sheet """
    DARK_QSS_KEY = 'darkCustomQss'
    LIGHT_QSS_KEY = 'lightCustomQss'

    def __init__(self, widget: QWidget) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.widget = widget

    def path(self, theme=Theme.AUTO):
        if False:
            i = 10
            return i + 15
        return ''

    def __eq__(self, other: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, CustomStyleSheet):
            return False
        return other.widget is self.widget

    def setCustomStyleSheet(self, lightQss: str, darkQss: str):
        if False:
            i = 10
            return i + 15
        ' set custom style sheet in light and dark theme mode '
        self.setLightStyleSheet(lightQss)
        self.setDarkStyleSheet(darkQss)
        return self

    def setLightStyleSheet(self, qss: str):
        if False:
            return 10
        ' set the style sheet in light mode '
        self.widget.setProperty(self.LIGHT_QSS_KEY, qss)
        return self

    def setDarkStyleSheet(self, qss: str):
        if False:
            i = 10
            return i + 15
        ' set the style sheet in dark mode '
        self.widget.setProperty(self.DARK_QSS_KEY, qss)
        return self

    def lightStyleSheet(self) -> str:
        if False:
            while True:
                i = 10
        return self.widget.property(self.LIGHT_QSS_KEY) or ''

    def darkStyleSheet(self) -> str:
        if False:
            while True:
                i = 10
        return self.widget.property(self.DARK_QSS_KEY) or ''

    def content(self, theme=Theme.AUTO) -> str:
        if False:
            print('Hello World!')
        theme = qconfig.theme if theme == Theme.AUTO else theme
        if theme == Theme.LIGHT:
            return self.lightStyleSheet()
        return self.darkStyleSheet()

class CustomStyleSheetWatcher(QObject):
    """ Custom style sheet watcher """

    def eventFilter(self, obj: QWidget, e: QEvent):
        if False:
            print('Hello World!')
        if e.type() != QEvent.DynamicPropertyChange:
            return super().eventFilter(obj, e)
        name = e.propertyName().data().decode()
        if name in [CustomStyleSheet.LIGHT_QSS_KEY, CustomStyleSheet.DARK_QSS_KEY]:
            addStyleSheet(obj, CustomStyleSheet(obj))
        return super().eventFilter(obj, e)

class StyleSheetCompose(StyleSheetBase):
    """ Style sheet compose """

    def __init__(self, sources: List[StyleSheetBase]):
        if False:
            print('Hello World!')
        super().__init__()
        self.sources = sources

    def content(self, theme=Theme.AUTO):
        if False:
            print('Hello World!')
        return '\n'.join([i.content(theme) for i in self.sources])

    def add(self, source: StyleSheetBase):
        if False:
            return 10
        ' add style sheet source '
        if source is self or source in self.sources:
            return
        self.sources.append(source)

    def remove(self, source: StyleSheetBase):
        if False:
            for i in range(10):
                print('nop')
        ' remove style sheet source '
        if source not in self.sources:
            return
        self.sources.remove(source)

def getStyleSheetFromFile(file: Union[str, QFile]):
    if False:
        print('Hello World!')
    ' get style sheet from qss file '
    f = QFile(file)
    f.open(QFile.ReadOnly)
    qss = str(f.readAll(), encoding='utf-8')
    f.close()
    return qss

def getStyleSheet(source: Union[str, StyleSheetBase], theme=Theme.AUTO):
    if False:
        for i in range(10):
            print('nop')
    ' get style sheet\n\n    Parameters\n    ----------\n    source: str | StyleSheetBase\n        qss source, it could be:\n          * `str`: qss file path\n          * `StyleSheetBase`: style sheet instance\n\n    theme: Theme\n        the theme of style sheet\n    '
    if isinstance(source, str):
        source = StyleSheetFile(source)
    return applyThemeColor(source.content(theme))

def setStyleSheet(widget: QWidget, source: Union[str, StyleSheetBase], theme=Theme.AUTO, register=True):
    if False:
        return 10
    ' set the style sheet of widget\n\n    Parameters\n    ----------\n    widget: QWidget\n        the widget to set style sheet\n\n    source: str | StyleSheetBase\n        qss source, it could be:\n          * `str`: qss file path\n          * `StyleSheetBase`: style sheet instance\n\n    theme: Theme\n        the theme of style sheet\n\n    register: bool\n        whether to register the widget to the style manager. If `register=True`, the style of\n        the widget will be updated automatically when the theme changes\n    '
    if register:
        styleSheetManager.register(source, widget)
    widget.setStyleSheet(getStyleSheet(source, theme))

def setCustomStyleSheet(widget: QWidget, lightQss: str, darkQss: str):
    if False:
        for i in range(10):
            print('nop')
    ' set custom style sheet\n\n    Parameters\n    ----------\n    widget: QWidget\n        the widget to add style sheet\n\n    lightQss: str\n        style sheet used in light theme mode\n\n    darkQss: str\n        style sheet used in light theme mode\n    '
    CustomStyleSheet(widget).setCustomStyleSheet(lightQss, darkQss)

def addStyleSheet(widget: QWidget, source: Union[str, StyleSheetBase], theme=Theme.AUTO, register=True):
    if False:
        for i in range(10):
            print('nop')
    ' add style sheet to widget\n\n    Parameters\n    ----------\n    widget: QWidget\n        the widget to set style sheet\n\n    source: str | StyleSheetBase\n        qss source, it could be:\n          * `str`: qss file path\n          * `StyleSheetBase`: style sheet instance\n\n    theme: Theme\n        the theme of style sheet\n\n    register: bool\n        whether to register the widget to the style manager. If `register=True`, the style of\n        the widget will be updated automatically when the theme changes\n    '
    if register:
        styleSheetManager.register(source, widget, reset=False)
        qss = getStyleSheet(styleSheetManager.source(widget), theme)
    else:
        qss = widget.styleSheet() + '\n' + getStyleSheet(source, theme)
    if qss.rstrip() != widget.styleSheet().rstrip():
        widget.setStyleSheet(qss)

def updateStyleSheet():
    if False:
        print('Hello World!')
    ' update the style sheet of all fluent widgets '
    removes = []
    for (widget, file) in styleSheetManager.items():
        try:
            setStyleSheet(widget, file, qconfig.theme)
        except RuntimeError:
            removes.append(widget)
    for widget in removes:
        styleSheetManager.deregister(widget)

def setTheme(theme: Theme, save=False):
    if False:
        while True:
            i = 10
    ' set the theme of application\n\n    Parameters\n    ----------\n    theme: Theme\n        theme mode\n\n    save: bool\n        whether to save the change to config file\n    '
    qconfig.set(qconfig.themeMode, theme, save)
    updateStyleSheet()
    qconfig.themeChangedFinished.emit()

def toggleTheme(save=False):
    if False:
        print('Hello World!')
    ' toggle the theme of application\n\n    Parameters\n    ----------\n    save: bool\n        whether to save the change to config file\n    '
    theme = Theme.LIGHT if isDarkTheme() else Theme.DARK
    setTheme(theme, save)

class ThemeColor(Enum):
    """ Theme color type """
    PRIMARY = 'ThemeColorPrimary'
    DARK_1 = 'ThemeColorDark1'
    DARK_2 = 'ThemeColorDark2'
    DARK_3 = 'ThemeColorDark3'
    LIGHT_1 = 'ThemeColorLight1'
    LIGHT_2 = 'ThemeColorLight2'
    LIGHT_3 = 'ThemeColorLight3'

    def name(self):
        if False:
            print('Hello World!')
        return self.color().name()

    def color(self):
        if False:
            print('Hello World!')
        color = qconfig.get(qconfig.themeColor)
        (h, s, v, _) = color.getHsvF()
        if isDarkTheme():
            s *= 0.84
            v = 1
            if self == self.DARK_1:
                v *= 0.9
            elif self == self.DARK_2:
                s *= 0.977
                v *= 0.82
            elif self == self.DARK_3:
                s *= 0.95
                v *= 0.7
            elif self == self.LIGHT_1:
                s *= 0.92
            elif self == self.LIGHT_2:
                s *= 0.78
            elif self == self.LIGHT_3:
                s *= 0.65
        elif self == self.DARK_1:
            v *= 0.75
        elif self == self.DARK_2:
            s *= 1.05
            v *= 0.5
        elif self == self.DARK_3:
            s *= 1.1
            v *= 0.4
        elif self == self.LIGHT_1:
            v *= 1.05
        elif self == self.LIGHT_2:
            s *= 0.75
            v *= 1.05
        elif self == self.LIGHT_3:
            s *= 0.65
            v *= 1.05
        return QColor.fromHsvF(h, min(s, 1), min(v, 1))

def themeColor():
    if False:
        while True:
            i = 10
    ' get theme color '
    return ThemeColor.PRIMARY.color()

def setThemeColor(color, save=False):
    if False:
        for i in range(10):
            print('nop')
    ' set theme color\n\n    Parameters\n    ----------\n    color: QColor | Qt.GlobalColor | str\n        theme color\n\n    save: bool\n        whether to save to change to config file\n    '
    color = QColor(color)
    qconfig.set(qconfig.themeColor, color, save=save)
    updateStyleSheet()