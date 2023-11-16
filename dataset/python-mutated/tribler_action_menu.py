from PyQt5.QtWidgets import QMenu
from tribler.gui.defs import CONTEXT_MENU_WIDTH

class TriblerActionMenu(QMenu):
    """
    This menu is displayed when a user right-clicks some items in Tribler, i.e. a download widget.
    Overrides QMenu to provide some custom CSS rules.
    """

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        QMenu.__init__(self, parent)
        self.setStyleSheet('\n        QMenu {\n            background-color: #404040;\n        }\n\n        QMenu::item {\n            color: #D0D0D0;\n            padding: 5px;\n        }\n\n        QMenu::item:selected {\n            background-color: #707070;\n        }\n\n        QMenu::item:disabled {\n            color: #999999;\n        }\n        ')
        self.setMinimumWidth(CONTEXT_MENU_WIDTH)