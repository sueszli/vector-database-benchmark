from tribler_apptester.action import Action

class TableViewSelectAction(Action):
    """
    This action selects a specific row in a table view.
    """

    def __init__(self, table_view_obj_name, row_index):
        if False:
            i = 10
            return i + 15
        super(TableViewSelectAction, self).__init__()
        self.table_view_obj_name = table_view_obj_name
        self.row_index = row_index

    def action_code(self):
        if False:
            return 10
        code = 'table_view = %s\nx = table_view.columnViewportPosition(0)\ny = table_view.rowViewportPosition(%d)\nindex = table_view.indexAt(QPoint(x, y))\ntable_view.setCurrentIndex(index)\n        ' % (self.table_view_obj_name, self.row_index)
        return code

    def required_imports(self):
        if False:
            while True:
                i = 10
        return ['from PyQt5.QtCore import QPoint']

class TableViewRandomSelectAction(Action):
    """
    This action selects a random row in a table view.
    """

    def __init__(self, table_view_obj_name):
        if False:
            while True:
                i = 10
        super(TableViewRandomSelectAction, self).__init__()
        self.table_view_obj_name = table_view_obj_name

    def action_code(self):
        if False:
            while True:
                i = 10
        code = 'table_view = %s\nrandom_row = randint(0, table_view.model().rowCount() - 1)\nx = table_view.columnViewportPosition(0)\ny = table_view.rowViewportPosition(random_row)\nindex = table_view.indexAt(QPoint(x, y))\ntable_view.setCurrentIndex(index)\n        ' % self.table_view_obj_name
        return code

    def required_imports(self):
        if False:
            print('Hello World!')
        return ['from random import randint', 'from PyQt5.QtCore import QPoint']