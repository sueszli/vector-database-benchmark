"""Run some automations to test things"""
from __future__ import unicode_literals
from __future__ import print_function
try:
    from pywinauto import application
except ImportError:
    import os.path
    pywinauto_path = os.path.abspath(__file__)
    pywinauto_path = os.path.split(os.path.split(pywinauto_path)[0])[0]
    import sys
    sys.path.append(pywinauto_path)
    from pywinauto import application
from pywinauto.timings import Timings
Timings.window_find_timeout = 10

def test_exceptions():
    if False:
        i = 10
        return i + 15
    'Test some things that should raise exceptions'
    try:
        app = application.Application()
        app.connect(path='No process with this please')
        assert False
    except application.ProcessNotFoundError:
        print('ProcessNotFoundError has been raised. OK.')
    try:
        app = application.Application()
        app.start(cmd_line='No process with this please')
        assert False
    except application.AppStartError:
        print('AppStartError has been raised. OK.')

def get_info():
    if False:
        print('Hello World!')
    'Run Notepad, print some identifiers and exit'
    app = application.Application()
    app.start('notepad.exe')
    app.Notepad.menu_select('File->PageSetup')
    print('==' * 20)
    print('Windows of this application:', app.windows())
    print('The list of identifiers for the Page Setup dialog in Notepad')
    print('==' * 20)
    app.PageSetup.print_control_identifiers()
    print('==' * 20)
    print('The list of identifiers for the 2nd Edit control in the dialog')
    app.PageSetup.Edit2.print_control_identifiers()
    print('==' * 20)
    app.PageSetup.OK.close_click()
    app.Notepad.menu_select('File->Exit')
if __name__ == '__main__':
    test_exceptions()
    get_info()