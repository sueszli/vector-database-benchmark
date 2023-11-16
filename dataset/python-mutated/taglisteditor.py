from PyQt6 import QtWidgets
from picard.util.tags import TAG_NAMES
from picard.ui.ui_widget_taglisteditor import Ui_TagListEditor
from picard.ui.widgets.editablelistview import AutocompleteItemDelegate, EditableListModel

class TagListEditor(QtWidgets.QWidget):

    def __init__(self, parent):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.ui = Ui_TagListEditor()
        self.ui.setupUi(self)
        list_view = self.ui.tag_list_view
        model = EditableListModel()
        model.user_sortable_changed.connect(self.on_user_sortable_changed)
        self.ui.sort_buttons.setVisible(model.user_sortable)
        list_view.setModel(model)
        list_view.setItemDelegate(AutocompleteItemDelegate(sorted(TAG_NAMES.keys())))
        selection = list_view.selectionModel()
        selection.selectionChanged.connect(self.on_selection_changed)
        self.on_selection_changed([], [])

    def on_selection_changed(self, selected, deselected):
        if False:
            print('Hello World!')
        indexes = self.ui.tag_list_view.selectedIndexes()
        last_row = self.ui.tag_list_view.model().rowCount() - 1
        buttons_enabled = len(indexes) > 0
        move_up_enabled = buttons_enabled and all((i.row() != 0 for i in indexes))
        move_down_enabled = buttons_enabled and all((i.row() != last_row for i in indexes))
        self.ui.tags_remove_btn.setEnabled(buttons_enabled)
        self.ui.tags_move_up_btn.setEnabled(move_up_enabled)
        self.ui.tags_move_down_btn.setEnabled(move_down_enabled)

    def clear(self):
        if False:
            while True:
                i = 10
        self.ui.tag_list_view.update([])

    def update(self, tags):
        if False:
            print('Hello World!')
        self.ui.tag_list_view.update(tags)

    @property
    def tags(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ui.tag_list_view.items

    def on_user_sortable_changed(self, user_sortable):
        if False:
            i = 10
            return i + 15
        self.ui.sort_buttons.setVisible(user_sortable)

    def set_user_sortable(self, user_sortable):
        if False:
            for i in range(10):
                print('nop')
        self.ui.tag_list_view.model().user_sortable = user_sortable