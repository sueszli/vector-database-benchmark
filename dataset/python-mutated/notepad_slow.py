"""Run some automations to test things"""
from __future__ import unicode_literals
from __future__ import print_function
import os.path
import sys
import time
try:
    from pywinauto import application
except ImportError:
    pywinauto_path = os.path.abspath(__file__)
    pywinauto_path = os.path.split(os.path.split(pywinauto_path)[0])[0]
    sys.path.append(pywinauto_path)
    from pywinauto import application
from pywinauto import tests
from pywinauto.findbestmatch import MatchError
from pywinauto.timings import Timings
print('Setting timings to slow settings, may be necessary for')
print('slow applications or slow machines.')
Timings.slow()

def run_notepad():
    if False:
        return 10
    'Run notepad and do some small stuff with it'
    start = time.time()
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
    doc_props.TabCtrl.select(0)
    doc_props.TabCtrl.select(1)
    try:
        doc_props.TabCtrl.select(2)
    except IndexError:
        print('Skip 3rd tab selection...')
    doc_props.TabCtrl.select('PaperQuality')
    try:
        doc_props.TabCtrl.select('JobRetention')
    except MatchError:
        print('Skip "Job Retention" tab...')
    doc_props.Cancel.close_click()
    if doc_props.Cancel.exists():
        doc_props.OK.close_click()
    app.PageSetupDlg.OK.close_click()
    app.PageSetupDlg.Ok.close_click()
    app.Notepad.Edit.set_edit_text('I am typing säme text to Notepad\r\n\r\nAnd then I am going to quit')
    app.Notepad.Edit.right_click()
    app.Popup.menu_item('Right To Left Reading Order').click()
    app.Notepad.Edit.type_keys('{END}{ENTER}SendText döés  süppôrt àcceñted characters!!!', with_spaces=True)
    app.Notepad.menu_select('File->SaveAs')
    app.SaveAs.EncodingComboBox.select('UTF-8')
    app.SaveAs.FileNameEdit.set_edit_text('Example-utf8.txt')
    app.SaveAs.Save.close_click()
    app.SaveAsDialog2.Cancel.wait_not('enabled')
    try:
        app.SaveAs.Yes.wait('exists').close_click()
    except MatchError:
        print('Skip overwriting...')
    app.Notepad.menu_select('File->Exit')
    print('That took %.3f to run' % (time.time() - start))
if __name__ == '__main__':
    run_notepad()