from hscommon.gui.table import GUITable, Row
from hscommon.gui.column import Column, Columns
from hscommon.trans import trget
coltr = trget('columns')

class ProblemTable(GUITable):
    COLUMNS = [Column('path', coltr('File Path')), Column('msg', coltr('Error Message'))]

    def __init__(self, problem_dialog):
        if False:
            i = 10
            return i + 15
        GUITable.__init__(self)
        self._columns = Columns(self)
        self.dialog = problem_dialog

    def _update_selection(self):
        if False:
            print('Hello World!')
        row = self.selected_row
        dupe = row.dupe if row is not None else None
        self.dialog.select_dupe(dupe)

    def _fill(self):
        if False:
            print('Hello World!')
        problems = self.dialog.app.results.problems
        for (dupe, msg) in problems:
            self.append(ProblemRow(self, dupe, msg))

class ProblemRow(Row):

    def __init__(self, table, dupe, msg):
        if False:
            print('Hello World!')
        Row.__init__(self, table)
        self.dupe = dupe
        self.msg = msg
        self.path = str(dupe.path)