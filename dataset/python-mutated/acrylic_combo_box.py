from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QAction
from .acrylic_menu import AcrylicMenuBase, AcrylicMenuActionListWidget
from .acrylic_line_edit import AcrylicLineEditBase
from ..widgets.combo_box import ComboBoxMenu, ComboBox, EditableComboBox
from ..widgets.menu import MenuAnimationType, RoundMenu, IndicatorMenuItemDelegate
from ..settings import SettingCard
from ...common.config import OptionsConfigItem, qconfig

class AcrylicComboMenuActionListWidget(AcrylicMenuActionListWidget):

    def _topMargin(self):
        if False:
            return 10
        return 2

class AcrylicComboBoxMenu(AcrylicMenuBase, RoundMenu):

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent=parent)
        self.setUpMenu(AcrylicComboMenuActionListWidget(self))
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setItemDelegate(IndicatorMenuItemDelegate())
        self.view.setObjectName('comboListWidget')
        self.setItemHeight(33)

class AcrylicComboBox(ComboBox):
    """ Acrylic combo box """

    def _createComboMenu(self):
        if False:
            print('Hello World!')
        return AcrylicComboBoxMenu(self)

class AcrylicEditableComboBox(AcrylicLineEditBase, EditableComboBox):
    """ Acrylic combo box """

    def _createComboMenu(self):
        if False:
            while True:
                i = 10
        return AcrylicComboBoxMenu(self)

class AcrylicComboBoxSettingCard(SettingCard):
    """ Setting card with a combo box """

    def __init__(self, configItem: OptionsConfigItem, icon, title, content=None, texts=None, parent=None):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        configItem: OptionsConfigItem\n            configuration item operated by the card\n\n        icon: str | QIcon | FluentIconBase\n            the icon to be drawn\n\n        title: str\n            the title of card\n\n        content: str\n            the content of card\n\n        texts: List[str]\n            the text of items\n\n        parent: QWidget\n            parent widget\n        '
        super().__init__(icon, title, content, parent)
        self.configItem = configItem
        self.comboBox = AcrylicComboBox(self)
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
            print('Hello World!')
        qconfig.set(self.configItem, self.comboBox.itemData(index))

    def setValue(self, value):
        if False:
            i = 10
            return i + 15
        if value not in self.optionToText:
            return
        self.comboBox.setCurrentText(self.optionToText[value])
        qconfig.set(self.configItem, value)