from hscommon.gui.base import GUIObject
from hscommon.gui.selectable_list import GUISelectableList

class CriterionCategoryList(GUISelectableList):

    def __init__(self, dialog):
        if False:
            while True:
                i = 10
        self.dialog = dialog
        GUISelectableList.__init__(self, [c.NAME for c in dialog.categories])

    def _update_selection(self):
        if False:
            print('Hello World!')
        self.dialog.select_category(self.dialog.categories[self.selected_index])
        GUISelectableList._update_selection(self)

class PrioritizationList(GUISelectableList):

    def __init__(self, dialog):
        if False:
            return 10
        self.dialog = dialog
        GUISelectableList.__init__(self)

    def _refresh_contents(self):
        if False:
            print('Hello World!')
        self[:] = [crit.display for crit in self.dialog.prioritizations]

    def move_indexes(self, indexes, dest_index):
        if False:
            for i in range(10):
                print('nop')
        indexes.sort()
        prilist = self.dialog.prioritizations
        selected = [prilist[i] for i in indexes]
        for i in reversed(indexes):
            del prilist[i]
        prilist[dest_index:dest_index] = selected
        self._refresh_contents()

    def remove_selected(self):
        if False:
            for i in range(10):
                print('nop')
        prilist = self.dialog.prioritizations
        for i in sorted(self.selected_indexes, reverse=True):
            del prilist[i]
        self._refresh_contents()

class PrioritizeDialog(GUIObject):

    def __init__(self, app):
        if False:
            return 10
        GUIObject.__init__(self)
        self.app = app
        self.categories = [cat(app.results) for cat in app._prioritization_categories()]
        self.category_list = CriterionCategoryList(self)
        self.criteria = []
        self.criteria_list = GUISelectableList()
        self.prioritizations = []
        self.prioritization_list = PrioritizationList(self)

    def _view_updated(self):
        if False:
            while True:
                i = 10
        self.category_list.select(0)

    def _sort_key(self, dupe):
        if False:
            for i in range(10):
                print('nop')
        return tuple((crit.sort_key(dupe) for crit in self.prioritizations))

    def select_category(self, category):
        if False:
            print('Hello World!')
        self.criteria = category.criteria_list()
        self.criteria_list[:] = [c.display_value for c in self.criteria]

    def add_selected(self):
        if False:
            print('Hello World!')
        if self.criteria_list.selected_index is None:
            return
        for i in self.criteria_list.selected_indexes:
            crit = self.criteria[i]
            self.prioritizations.append(crit)
            del crit
        self.prioritization_list[:] = [crit.display for crit in self.prioritizations]

    def remove_selected(self):
        if False:
            return 10
        self.prioritization_list.remove_selected()
        self.prioritization_list.select([])

    def perform_reprioritization(self):
        if False:
            print('Hello World!')
        self.app.reprioritize_groups(self._sort_key)