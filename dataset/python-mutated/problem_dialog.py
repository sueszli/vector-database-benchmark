from hscommon import desktop
from core.gui.problem_table import ProblemTable

class ProblemDialog:

    def __init__(self, app):
        if False:
            while True:
                i = 10
        self.app = app
        self._selected_dupe = None
        self.problem_table = ProblemTable(self)

    def refresh(self):
        if False:
            for i in range(10):
                print('nop')
        self._selected_dupe = None
        self.problem_table.refresh()

    def reveal_selected_dupe(self):
        if False:
            return 10
        if self._selected_dupe is not None:
            desktop.reveal_path(self._selected_dupe.path)

    def select_dupe(self, dupe):
        if False:
            i = 10
            return i + 15
        self._selected_dupe = dupe