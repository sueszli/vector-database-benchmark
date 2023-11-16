from qt.core import QMenu, QToolButton
from calibre.gui2.actions import InterfaceAction, show_menu_under_widget

class ManageCategoriesAction(InterfaceAction):
    name = 'Manage categories'
    action_spec = (_('Manage categories'), 'tags.png', _('Manage categories: authors, tags, series, etc.'), None)
    action_type = 'current'
    popup_type = QToolButton.ToolButtonPopupMode.InstantPopup
    action_add_menu = True
    dont_add_to = frozenset(['context-menu-device', 'menubar-device'])

    def genesis(self):
        if False:
            return 10
        self.menu = m = self.qaction.menu()
        m.aboutToShow.connect(self.about_to_show_menu)
        self.hidden_menu = QMenu()
        self.shortcut_action = self.create_menu_action(menu=self.hidden_menu, unique_name='Manage categories', text=_('Manage categories: authors, tags, series, etc.'), icon='tags.png', triggered=self.show_menu)

    def show_menu(self):
        if False:
            print('Hello World!')
        show_menu_under_widget(self.gui, self.menu, self.qaction, self.name)

    def about_to_show_menu(self):
        if False:
            return 10
        db = self.gui.current_db
        self.gui.populate_manage_categories_menu(db, self.menu, add_column_items=True)

    def location_selected(self, loc):
        if False:
            return 10
        enabled = loc == 'library'
        self.qaction.setEnabled(enabled)
        for action in self.menu.actions():
            action.setEnabled(enabled)