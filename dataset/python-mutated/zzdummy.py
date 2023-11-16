"""Example extension, also used for testing.

See extend.txt for more details on creating an extension.
See config-extension.def for configuring an extension.
"""
from idlelib.config import idleConf
from functools import wraps

def format_selection(format_line):
    if False:
        while True:
            i = 10
    'Apply a formatting function to all of the selected lines.'

    @wraps(format_line)
    def apply(self, event=None):
        if False:
            print('Hello World!')
        (head, tail, chars, lines) = self.formatter.get_region()
        for pos in range(len(lines) - 1):
            line = lines[pos]
            lines[pos] = format_line(self, line)
        self.formatter.set_region(head, tail, chars, lines)
        return 'break'
    return apply

class ZzDummy:
    """Prepend or remove initial text from selected lines."""
    menudefs = [('format', [('Z in', '<<z-in>>'), ('Z out', '<<z-out>>')])]

    def __init__(self, editwin):
        if False:
            return 10
        'Initialize the settings for this extension.'
        self.editwin = editwin
        self.text = editwin.text
        self.formatter = editwin.fregion

    @classmethod
    def reload(cls):
        if False:
            for i in range(10):
                print('nop')
        'Load class variables from config.'
        cls.ztext = idleConf.GetOption('extensions', 'ZzDummy', 'z-text')

    @format_selection
    def z_in_event(self, line):
        if False:
            while True:
                i = 10
        'Insert text at the beginning of each selected line.\n\n        This is bound to the <<z-in>> virtual event when the extensions\n        are loaded.\n        '
        return f'{self.ztext}{line}'

    @format_selection
    def z_out_event(self, line):
        if False:
            while True:
                i = 10
        'Remove specific text from the beginning of each selected line.\n\n        This is bound to the <<z-out>> virtual event when the extensions\n        are loaded.\n        '
        zlength = 0 if not line.startswith(self.ztext) else len(self.ztext)
        return line[zlength:]
ZzDummy.reload()
if __name__ == '__main__':
    import unittest
    unittest.main('idlelib.idle_test.test_zzdummy', verbosity=2, exit=False)