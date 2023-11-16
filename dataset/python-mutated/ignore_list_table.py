from hscommon.gui.table import GUITable, Row
from hscommon.gui.column import Column, Columns
from hscommon.trans import trget
coltr = trget('columns')

class IgnoreListTable(GUITable):
    COLUMNS = [Column('path1', coltr('File Path') + ' 1'), Column('path2', coltr('File Path') + ' 2')]

    def __init__(self, ignore_list_dialog):
        if False:
            for i in range(10):
                print('nop')
        GUITable.__init__(self)
        self._columns = Columns(self)
        self.view = None
        self.dialog = ignore_list_dialog

    def _fill(self):
        if False:
            print('Hello World!')
        for (path1, path2) in self.dialog.ignore_list:
            self.append(IgnoreListRow(self, path1, path2))

class IgnoreListRow(Row):

    def __init__(self, table, path1, path2):
        if False:
            while True:
                i = 10
        Row.__init__(self, table)
        self.path1_original = path1
        self.path2_original = path2
        self.path1 = str(path1)
        self.path2 = str(path2)