from typing import Union
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QPushButton
from ..dialog_box.color_dialog import ColorDialog
from ..widgets.combo_box import ComboBox
from ..widgets.switch_button import SwitchButton, IndicatorPosition
from ..widgets.slider import Slider
from ..widgets.icon_widget import IconWidget
from ..widgets.button import HyperlinkButton
from ...common.style_sheet import FluentStyleSheet
from ...common.config import qconfig, isDarkTheme, ConfigItem, OptionsConfigItem
from ...common.icon import FluentIconBase

class SettingCard(QFrame):
    """ Setting card """

    def __init__(self, icon: Union[str, QIcon, FluentIconBase], title, content=None, parent=None):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        icon: str | QIcon | FluentIconBase\n            the icon to be drawn\n\n        title: str\n            the title of card\n\n        content: str\n            the content of card\n\n        parent: QWidget\n            parent widget\n        '
        super().__init__(parent=parent)
        self.iconLabel = IconWidget(icon, self)
        self.titleLabel = QLabel(title, self)
        self.contentLabel = QLabel(content or '', self)
        self.hBoxLayout = QHBoxLayout(self)
        self.vBoxLayout = QVBoxLayout()
        if not content:
            self.contentLabel.hide()
        self.setFixedHeight(70 if content else 50)
        self.iconLabel.setFixedSize(16, 16)
        self.hBoxLayout.setSpacing(0)
        self.hBoxLayout.setContentsMargins(16, 0, 0, 0)
        self.hBoxLayout.setAlignment(Qt.AlignVCenter)
        self.vBoxLayout.setSpacing(0)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.setAlignment(Qt.AlignVCenter)
        self.hBoxLayout.addWidget(self.iconLabel, 0, Qt.AlignLeft)
        self.hBoxLayout.addSpacing(16)
        self.hBoxLayout.addLayout(self.vBoxLayout)
        self.vBoxLayout.addWidget(self.titleLabel, 0, Qt.AlignLeft)
        self.vBoxLayout.addWidget(self.contentLabel, 0, Qt.AlignLeft)
        self.hBoxLayout.addSpacing(16)
        self.hBoxLayout.addStretch(1)
        self.contentLabel.setObjectName('contentLabel')
        FluentStyleSheet.SETTING_CARD.apply(self)

    def setTitle(self, title: str):
        if False:
            print('Hello World!')
        ' set the title of card '
        self.titleLabel.setText(title)

    def setContent(self, content: str):
        if False:
            for i in range(10):
                print('nop')
        ' set the content of card '
        self.contentLabel.setText(content)
        self.contentLabel.setVisible(bool(content))

    def setValue(self, value):
        if False:
            while True:
                i = 10
        ' set the value of config item '
        pass

    def paintEvent(self, e):
        if False:
            while True:
                i = 10
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        if isDarkTheme():
            painter.setBrush(QColor(255, 255, 255, 13))
            painter.setPen(QColor(0, 0, 0, 50))
        else:
            painter.setBrush(QColor(255, 255, 255, 170))
            painter.setPen(QColor(0, 0, 0, 19))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 6, 6)

class SwitchSettingCard(SettingCard):
    """ Setting card with switch button """
    checkedChanged = pyqtSignal(bool)

    def __init__(self, icon: Union[str, QIcon, FluentIconBase], title, content=None, configItem: ConfigItem=None, parent=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        icon: str | QIcon | FluentIconBase\n            the icon to be drawn\n\n        title: str\n            the title of card\n\n        content: str\n            the content of card\n\n        configItem: ConfigItem\n            configuration item operated by the card\n\n        parent: QWidget\n            parent widget\n        '
        super().__init__(icon, title, content, parent)
        self.configItem = configItem
        self.switchButton = SwitchButton(self.tr('Off'), self, IndicatorPosition.RIGHT)
        if configItem:
            self.setValue(qconfig.get(configItem))
            configItem.valueChanged.connect(self.setValue)
        self.hBoxLayout.addWidget(self.switchButton, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.switchButton.checkedChanged.connect(self.__onCheckedChanged)

    def __onCheckedChanged(self, isChecked: bool):
        if False:
            for i in range(10):
                print('nop')
        ' switch button checked state changed slot '
        self.setValue(isChecked)
        self.checkedChanged.emit(isChecked)

    def setValue(self, isChecked: bool):
        if False:
            return 10
        if self.configItem:
            qconfig.set(self.configItem, isChecked)
        self.switchButton.setChecked(isChecked)
        self.switchButton.setText(self.tr('On') if isChecked else self.tr('Off'))

    def setChecked(self, isChecked: bool):
        if False:
            i = 10
            return i + 15
        self.setValue(isChecked)

    def isChecked(self):
        if False:
            return 10
        return self.switchButton.isChecked()

class RangeSettingCard(SettingCard):
    """ Setting card with a slider """
    valueChanged = pyqtSignal(int)

    def __init__(self, configItem, icon: Union[str, QIcon, FluentIconBase], title, content=None, parent=None):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        configItem: RangeConfigItem\n            configuration item operated by the card\n\n        icon: str | QIcon | FluentIconBase\n            the icon to be drawn\n\n        title: str\n            the title of card\n\n        content: str\n            the content of card\n\n        parent: QWidget\n            parent widget\n        '
        super().__init__(icon, title, content, parent)
        self.configItem = configItem
        self.slider = Slider(Qt.Horizontal, self)
        self.valueLabel = QLabel(self)
        self.slider.setMinimumWidth(268)
        self.slider.setSingleStep(1)
        self.slider.setRange(*configItem.range)
        self.slider.setValue(configItem.value)
        self.valueLabel.setNum(configItem.value)
        self.hBoxLayout.addStretch(1)
        self.hBoxLayout.addWidget(self.valueLabel, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(6)
        self.hBoxLayout.addWidget(self.slider, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.valueLabel.setObjectName('valueLabel')
        configItem.valueChanged.connect(self.setValue)
        self.slider.valueChanged.connect(self.__onValueChanged)

    def __onValueChanged(self, value: int):
        if False:
            print('Hello World!')
        ' slider value changed slot '
        self.setValue(value)
        self.valueChanged.emit(value)

    def setValue(self, value):
        if False:
            return 10
        qconfig.set(self.configItem, value)
        self.valueLabel.setNum(value)
        self.valueLabel.adjustSize()
        self.slider.setValue(value)

class PushSettingCard(SettingCard):
    """ Setting card with a push button """
    clicked = pyqtSignal()

    def __init__(self, text, icon: Union[str, QIcon, FluentIconBase], title, content=None, parent=None):
        if False:
            while True:
                i = 10
        '\n        Parameters\n        ----------\n        text: str\n            the text of push button\n\n        icon: str | QIcon | FluentIconBase\n            the icon to be drawn\n\n        title: str\n            the title of card\n\n        content: str\n            the content of card\n\n        parent: QWidget\n            parent widget\n        '
        super().__init__(icon, title, content, parent)
        self.button = QPushButton(text, self)
        self.hBoxLayout.addWidget(self.button, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.button.clicked.connect(self.clicked)

class PrimaryPushSettingCard(PushSettingCard):
    """ Push setting card with primary color """

    def __init__(self, text, icon, title, content=None, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(text, icon, title, content, parent)
        self.button.setObjectName('primaryButton')

class HyperlinkCard(SettingCard):
    """ Hyperlink card """

    def __init__(self, url, text, icon: Union[str, QIcon, FluentIconBase], title, content=None, parent=None):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        url: str\n            the url to be opened\n\n        text: str\n            text of url\n\n        icon: str | QIcon | FluentIconBase\n            the icon to be drawn\n\n        title: str\n            the title of card\n\n        content: str\n            the content of card\n\n        text: str\n            the text of push button\n\n        parent: QWidget\n            parent widget\n        '
        super().__init__(icon, title, content, parent)
        self.linkButton = HyperlinkButton(url, text, self)
        self.hBoxLayout.addWidget(self.linkButton, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)

class ColorPickerButton(QToolButton):
    """ Color picker button """
    colorChanged = pyqtSignal(QColor)

    def __init__(self, color: QColor, title: str, parent=None, enableAlpha=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent=parent)
        self.title = title
        self.enableAlpha = enableAlpha
        self.setFixedSize(96, 32)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setColor(color)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self.__showColorDialog)

    def __showColorDialog(self):
        if False:
            for i in range(10):
                print('nop')
        ' show color dialog '
        w = ColorDialog(self.color, self.tr('Choose ') + self.title, self.window(), self.enableAlpha)
        w.colorChanged.connect(self.__onColorChanged)
        w.exec()

    def __onColorChanged(self, color):
        if False:
            i = 10
            return i + 15
        ' color changed slot '
        self.setColor(color)
        self.colorChanged.emit(color)

    def setColor(self, color):
        if False:
            return 10
        ' set color '
        self.color = QColor(color)
        self.update()

    def paintEvent(self, e):
        if False:
            return 10
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        pc = QColor(255, 255, 255, 10) if isDarkTheme() else QColor(234, 234, 234)
        painter.setPen(pc)
        color = QColor(self.color)
        if not self.enableAlpha:
            color.setAlpha(255)
        painter.setBrush(color)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)

class ColorSettingCard(SettingCard):
    """ Setting card with color picker """
    colorChanged = pyqtSignal(QColor)

    def __init__(self, configItem, icon: Union[str, QIcon, FluentIconBase], title: str, content: str=None, parent=None, enableAlpha=False):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        configItem: RangeConfigItem\n            configuration item operated by the card\n\n        icon: str | QIcon | FluentIconBase\n            the icon to be drawn\n\n        title: str\n            the title of card\n\n        content: str\n            the content of card\n\n        parent: QWidget\n            parent widget\n\n        enableAlpha: bool\n            whether to enable the alpha channel\n        '
        super().__init__(icon, title, content, parent)
        self.configItem = configItem
        self.colorPicker = ColorPickerButton(qconfig.get(configItem), title, self, enableAlpha)
        self.hBoxLayout.addWidget(self.colorPicker, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.colorPicker.colorChanged.connect(self.__onColorChanged)
        configItem.valueChanged.connect(self.setValue)

    def __onColorChanged(self, color: QColor):
        if False:
            while True:
                i = 10
        qconfig.set(self.configItem, color)
        self.colorChanged.emit(color)

    def setValue(self, color: QColor):
        if False:
            i = 10
            return i + 15
        self.colorPicker.setColor(color)
        qconfig.set(self.configItem, color)

class ComboBoxSettingCard(SettingCard):
    """ Setting card with a combo box """

    def __init__(self, configItem: OptionsConfigItem, icon: Union[str, QIcon, FluentIconBase], title, content=None, texts=None, parent=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        configItem: OptionsConfigItem\n            configuration item operated by the card\n\n        icon: str | QIcon | FluentIconBase\n            the icon to be drawn\n\n        title: str\n            the title of card\n\n        content: str\n            the content of card\n\n        texts: List[str]\n            the text of items\n\n        parent: QWidget\n            parent widget\n        '
        super().__init__(icon, title, content, parent)
        self.configItem = configItem
        self.comboBox = ComboBox(self)
        self.hBoxLayout.addWidget(self.comboBox, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.optionToText = {o: t for (o, t) in zip(configItem.options, texts)}
        for (text, option) in zip(texts, configItem.options):
            self.comboBox.addItem(text, userData=option)
        self.comboBox.setCurrentText(self.optionToText[qconfig.get(configItem)])
        self.comboBox.currentIndexChanged.connect(self._onCurrentIndexChanged)
        configItem.valueChanged.connect(self.setValue)

    def _onCurrentIndexChanged(self, index: int):
        if False:
            for i in range(10):
                print('nop')
        qconfig.set(self.configItem, self.comboBox.itemData(index))

    def setValue(self, value):
        if False:
            print('Hello World!')
        if value not in self.optionToText:
            return
        self.comboBox.setCurrentText(self.optionToText[value])
        qconfig.set(self.configItem, value)