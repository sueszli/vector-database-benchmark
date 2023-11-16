from calibre.gui2.actions import InterfaceAction

class BooklistContextMenuAction(InterfaceAction):
    name = 'Booklist context menu'
    action_spec = (_('Book list header menu'), 'context_menu.png', _('Show the book list header context menu'), ())
    action_type = 'current'
    action_add_menu = False
    dont_add_to = frozenset(['context-menu-device', 'menubar-device'])

    def genesis(self):
        if False:
            while True:
                i = 10
        self.qaction.triggered.connect(self.show_context_menu)

    def show_context_menu(self):
        if False:
            for i in range(10):
                print('nop')
        self.gui.library_view.show_column_header_context_menu_from_action()

    def location_selected(self, loc):
        if False:
            i = 10
            return i + 15
        self.qaction.setEnabled(loc == 'library')