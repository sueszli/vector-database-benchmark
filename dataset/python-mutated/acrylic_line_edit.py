from .acrylic_menu import AcrylicCompleterMenu, AcrylicLineEditMenu
from ..widgets.line_edit import LineEdit, SearchLineEdit

class AcrylicLineEditBase:
    """ Acrylic line edit base """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

    def setCompleter(self, completer):
        if False:
            print('Hello World!')
        super().setCompleter(completer)
        self.setCompleterMenu(AcrylicCompleterMenu(self))

    def contextMenuEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        menu = AcrylicLineEditMenu(self)
        menu.exec(e.globalPos())

class AcrylicLineEdit(AcrylicLineEditBase, LineEdit):
    """ Acrylic line edit """

class AcrylicSearchLineEdit(AcrylicLineEditBase, SearchLineEdit):
    """ Acrylic search line edit """