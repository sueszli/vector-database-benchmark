from hscommon.gui.base import GUIObject
from core.gui.base import DupeGuruGUIObject

class DetailsPanel(GUIObject, DupeGuruGUIObject):

    def __init__(self, app):
        if False:
            print('Hello World!')
        GUIObject.__init__(self, multibind=True)
        DupeGuruGUIObject.__init__(self, app)
        self._table = []

    def _view_updated(self):
        if False:
            i = 10
            return i + 15
        self._refresh()
        self.view.refresh()

    def _refresh(self):
        if False:
            print('Hello World!')
        if self.app.selected_dupes:
            dupe = self.app.selected_dupes[0]
            group = self.app.results.get_group_of_duplicate(dupe)
        else:
            dupe = None
            group = None
        data1 = self.app.get_display_info(dupe, group, False)
        ref = group.ref if group is not None and group.ref is not dupe else None
        data2 = self.app.get_display_info(ref, group, False)
        columns = self.app.result_table.COLUMNS[1:]
        self._table = [(c.display, data1[c.name], data2[c.name]) for c in columns]

    def row_count(self):
        if False:
            print('Hello World!')
        return len(self._table)

    def row(self, row_index):
        if False:
            i = 10
            return i + 15
        return self._table[row_index]

    def dupes_selected(self):
        if False:
            i = 10
            return i + 15
        self._view_updated()