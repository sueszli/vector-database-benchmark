from calibre.gui2.actions import InterfaceAction

class AutoscrollBooksAction(InterfaceAction):
    name = 'Autoscroll Books'
    action_spec = (_('Auto scroll through the book list'), 'auto-scroll.png', _('Auto scroll through the book list, particularly useful with the cover browser open'), _('X'))
    dont_add_to = frozenset(('context-menu-device', 'menubar-device'))
    action_type = 'current'

    def genesis(self):
        if False:
            i = 10
            return i + 15
        self.qaction.triggered.connect(self.gui.toggle_auto_scroll)

    def location_selected(self, loc):
        if False:
            while True:
                i = 10
        enabled = loc == 'library'
        self.qaction.setEnabled(enabled)
        self.menuless_qaction.setEnabled(enabled)