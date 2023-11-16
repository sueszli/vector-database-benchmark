from calibre.gui2.actions import InterfaceAction

class BrowseNotesAction(InterfaceAction):
    name = 'Browse Notes'
    action_spec = (_('Browse notes'), 'notes.png', _('Browse notes for authors, tags, etc. in the library'), _('Ctrl+Shift+N'))
    dont_add_to = frozenset(('context-menu-device',))
    action_type = 'current'

    def genesis(self):
        if False:
            i = 10
            return i + 15
        self.d = None
        self.qaction.triggered.connect(self.show_browser)

    def show_browser(self):
        if False:
            for i in range(10):
                print('nop')
        if self.d is not None and self.d.isVisible():
            self.d.raise_()
            self.d.activateWindow()
        else:
            from calibre.gui2.library.notes import NotesBrowser
            self.d = NotesBrowser(self.gui)
            self.d.show()