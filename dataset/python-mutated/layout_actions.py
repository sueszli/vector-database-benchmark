from enum import Enum
from functools import partial
from qt.core import QIcon, QToolButton
from calibre.gui2.actions import InterfaceAction

class Panel(Enum):
    """ See gui2.init for these """
    SEARCH_BAR = 'sb'
    TAG_BROWSER = 'tb'
    BOOK_DETAILS = 'bd'
    GRID_VIEW = 'gv'
    COVER_BROWSER = 'cb'
    QUICKVIEW = 'qv'

class LayoutActions(InterfaceAction):
    name = 'Layout Actions'
    action_spec = (_('Layout actions'), 'layout.png', _('Add/remove layout items: search bar, tag browser, etc.'), None)
    action_type = 'current'
    popup_type = QToolButton.ToolButtonPopupMode.InstantPopup
    action_add_menu = True
    dont_add_to = frozenset({'context-menu-device', 'menubar-device'})

    def gui_layout_complete(self):
        if False:
            i = 10
            return i + 15
        m = self.qaction.menu()
        m.addAction(_('Hide all'), self.hide_all)
        for (button, name) in zip(self.gui.layout_buttons, self.gui.button_order):
            m.addSeparator()
            ic = QIcon.ic(button.icname)
            m.addAction(ic, _('Show {}').format(button.label), partial(self.set_visible, Panel(name), True))
            m.addAction(ic, _('Hide {}').format(button.label), partial(self.set_visible, Panel(name), False))

    def _change_item(self, button, show=True):
        if False:
            i = 10
            return i + 15
        if button.isChecked() and (not show):
            button.click()
        elif not button.isChecked() and show:
            button.click()

    def _button_from_enum(self, name: Panel):
        if False:
            i = 10
            return i + 15
        for (q, b) in zip(self.gui.button_order, self.gui.layout_buttons):
            if q == name.value:
                return b

    def set_visible(self, name: Panel, show=True):
        if False:
            print('Hello World!')
        '\n        Show or hide the panel. Does nothing if the panel is already in the\n        desired state.\n\n        :param name: specifies which panel using a Panel enum\n        :param show: If True, show the panel, otherwise hide the panel\n        '
        self._change_item(self._button_from_enum(name), show)

    def is_visible(self, name: Panel):
        if False:
            print('Hello World!')
        '\n        Returns True if the panel is visible.\n\n        :param name: specifies which panel using a Panel enum\n        '
        self._button_from_enum(name).isChecked()

    def hide_all(self):
        if False:
            for i in range(10):
                print('nop')
        for name in self.gui.button_order:
            self.set_visible(Panel(name), show=False)

    def show_all(self):
        if False:
            return 10
        for name in self.gui.button_order:
            self.set_visible(Panel(name), show=True)

    def panel_titles(self):
        if False:
            while True:
                i = 10
        '\n        Return a dictionary of Panel Enum items to translated human readable title.\n        Simplifies building dialogs, for example combo boxes of all the panel\n        names or check boxes for each panel.\n\n        :return: {Panel_enum_value: human readable title, ...}\n        '
        return {p: self._button_from_enum(p).label for p in Panel}