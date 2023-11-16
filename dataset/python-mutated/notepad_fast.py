"""Run some automations to test things"""
from __future__ import unicode_literals
from __future__ import print_function
import sys
import os.path
import time
try:
    from pywinauto import application
except ImportError:
    pywinauto_path = os.path.abspath(__file__)
    pywinauto_path = os.path.split(os.path.split(pywinauto_path)[0])[0]
    sys.path.append(pywinauto_path)
    from pywinauto import application
import pywinauto
from pywinauto import tests
from pywinauto.timings import Timings

def run_notepad():
    if False:
        i = 10
        return i + 15
    'Run notepad and do some small stuff with it'
    print("Run with option 'language' e.g. notepad_fast.py language to use")
    print('application data. This should work on any language Windows/Notepad')
    print()
    print("Trying fast timing settings - it's  possible these won't work")
    print('if pywinauto tries to access a window that is not accessible yet')
    Timings.fast()
    Timings.window_find_timeout = 10
    start = time.time()
    run_with_appdata = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'language':
        run_with_appdata = True
    scriptdir = os.path.split(os.path.abspath(__file__))[0]
    if run_with_appdata:
        print('\nRunning this script so it will load application data and run')
        print('against any lanuguage version of Notepad/Windows')
        app = application.Application(os.path.join(scriptdir, 'Notepad_fast.pkl'))
    else:
        app = application.Application()
    app.start('notepad.exe')
    app.Notepad.menu_select('File->PageSetup')
    app.PageSetupDlg.SizeComboBox.select(4)
    try:
        app.PageSetupDlg.SizeComboBox.select('Letter')
    except ValueError:
        app.PageSetupDlg.SizeComboBox.select('Letter (8.5" x 11")')
    app.PageSetupDlg.SizeComboBox.select(2)
    bugs = app.PageSetupDlg.run_tests('RepeatedHotkey Truncation')
    tests.print_bugs(bugs)
    app.PageSetupDlg.Printer.click()
    app.PageSetupDlg.Network.click()
    app.ConnectToPrinter.ExpandByDefault.check()
    app.ConnectToPrinter.ExpandByDefault.uncheck()
    app.ConnectToPrinter.ExpandByDefault.click()
    app.ConnectToPrinter.ExpandByDefault.click()
    app.ConnectToPrinter.Cancel.close_click()
    app.PageSetupDlg.Properties.click()
    doc_props = app.window(name_re='.*Properties$')
    doc_props.wait('exists', timeout=40)
    doc_props.Cancel.close_click()
    if doc_props.Cancel.exists():
        doc_props.OK.close_click()
    app.PageSetupDlg.OK.close_click()
    app.PageSetupDlg.Ok.close_click()
    app.Notepad.Edit.set_edit_text(u'I am typing säme text to Notepad\r\n\r\nAnd then I am going to quit')
    app.Notepad.Edit.right_click()
    app.Popup.menu_item('Right To Left Reading Order').click()
    app.Notepad.Edit.type_keys(u'{END}{ENTER}SendText döés süppôrt àcceñted characters!!!', with_spaces=True)
    app.Notepad.menu_select('File->SaveAs')
    app.SaveAs.EncodingComboBox.select('UTF-8')
    app.SaveAs.FileNameEdit.set_edit_text('Example-utf8.txt')
    app.SaveAs.Save.close_click()
    app.SaveAsDialog2.Cancel.wait_not('enabled')
    try:
        app.SaveAs.Yes.wait('exists').close_click()
    except pywinauto.MatchError:
        print('Skip overwriting...')
    app.Notepad.menu_select('File->Exit')
    if not run_with_appdata:
        app.WriteAppData(os.path.join(scriptdir, 'Notepad_fast.pkl'))
    print('That took %.3f to run' % (time.time() - start))
if __name__ == '__main__':
    run_notepad()