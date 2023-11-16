from core.gui.exclude_list_table import ExcludeListTable
from core.exclude import has_sep
from os import sep
import logging

class ExcludeListDialogCore:

    def __init__(self, app):
        if False:
            print('Hello World!')
        self.app = app
        self.exclude_list = self.app.exclude_list
        self.exclude_list_table = ExcludeListTable(self, app)

    def restore_defaults(self):
        if False:
            return 10
        self.exclude_list.restore_defaults()
        self.refresh()

    def refresh(self):
        if False:
            i = 10
            return i + 15
        self.exclude_list_table.refresh()

    def remove_selected(self):
        if False:
            print('Hello World!')
        for row in self.exclude_list_table.selected_rows:
            self.exclude_list_table.remove(row)
            self.exclude_list.remove(row.regex)
        self.refresh()

    def rename_selected(self, newregex):
        if False:
            return 10
        "Rename the selected regex to ``newregex``.\n        If there is more than one selected row, the first one is used.\n        :param str newregex: The regex to rename the row's regex to.\n        :return bool: true if success, false if error.\n        "
        try:
            r = self.exclude_list_table.selected_rows[0]
            self.exclude_list.rename(r.regex, newregex)
            self.refresh()
            return True
        except Exception as e:
            logging.warning(f'Error while renaming regex to {newregex}: {e}')
        return False

    def add(self, regex):
        if False:
            return 10
        self.exclude_list.add(regex)
        self.exclude_list.mark(regex)
        self.exclude_list_table.add(regex)

    def test_string(self, test_string):
        if False:
            while True:
                i = 10
        'Set the highlight property on each row when its regex matches the\n        test_string supplied. Return True if any row matched.'
        matched = False
        for row in self.exclude_list_table.rows:
            compiled_regex = self.exclude_list.get_compiled(row.regex)
            if self.is_match(test_string, compiled_regex):
                row.highlight = True
                matched = True
            else:
                row.highlight = False
        return matched

    def is_match(self, test_string, compiled_regex):
        if False:
            print('Hello World!')
        if not compiled_regex:
            return False
        matched = False
        if not has_sep(compiled_regex.pattern) and sep in test_string:
            filename = test_string.rsplit(sep, 1)[1]
            if compiled_regex.fullmatch(filename):
                matched = True
            return matched
        if compiled_regex.fullmatch(test_string):
            matched = True
        return matched

    def reset_rows_highlight(self):
        if False:
            for i in range(10):
                print('nop')
        for row in self.exclude_list_table.rows:
            row.highlight = False

    def show(self):
        if False:
            for i in range(10):
                print('nop')
        self.view.show()