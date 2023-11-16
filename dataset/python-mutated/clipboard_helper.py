"""
Clipboard helper module.
"""
from qtpy.QtWidgets import QApplication
from spyder.py3compat import to_text_string

class ClipboardHelper:
    metadata_hash = None
    metadata_indent = None
    metadata_tab_stop_width_spaces = None

    def get_current_hash(self):
        if False:
            while True:
                i = 10
        clipboard = QApplication.clipboard()
        return hash(to_text_string(clipboard.text()))

    def get_line_indentation(self, text, tab_stop_width_spaces=None):
        if False:
            i = 10
            return i + 15
        'Get indentation for given line.'
        if tab_stop_width_spaces:
            text = text.replace('\t', ' ' * tab_stop_width_spaces)
        return len(text) - len(text.lstrip())

    def save_indentation(self, preceding_text, tab_stop_width_spaces=None):
        if False:
            i = 10
            return i + 15
        '\n        Save the indentation corresponding to the clipboard data.\n\n        Must be called right after copying.\n        '
        self.metadata_hash = self.get_current_hash()
        self.metadata_indent = self.get_line_indentation(preceding_text, tab_stop_width_spaces)
        self.metadata_tab_stop_width_spaces = tab_stop_width_spaces

    def remaining_lines_adjustment(self, preceding_text):
        if False:
            while True:
                i = 10
        '\n        Get remaining lines adjustments needed to keep multiline\n        pasted text consistant.\n        '
        if self.get_current_hash() == self.metadata_hash:
            return self.get_line_indentation(preceding_text, self.metadata_tab_stop_width_spaces) - self.metadata_indent
        return 0
CLIPBOARD_HELPER = ClipboardHelper()