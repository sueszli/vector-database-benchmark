from hscommon.trans import tr
from core.gui.ignore_list_table import IgnoreListTable

class IgnoreListDialog:

    def __init__(self, app):
        if False:
            i = 10
            return i + 15
        self.app = app
        self.ignore_list = self.app.ignore_list
        self.ignore_list_table = IgnoreListTable(self)

    def clear(self):
        if False:
            while True:
                i = 10
        if not self.ignore_list:
            return
        msg = tr('Do you really want to remove all %d items from the ignore list?') % len(self.ignore_list)
        if self.app.view.ask_yes_no(msg):
            self.ignore_list.clear()
            self.refresh()

    def refresh(self):
        if False:
            return 10
        self.ignore_list_table.refresh()

    def remove_selected(self):
        if False:
            return 10
        for row in self.ignore_list_table.selected_rows:
            self.ignore_list.remove(row.path1_original, row.path2_original)
        self.refresh()

    def show(self):
        if False:
            return 10
        self.view.show()