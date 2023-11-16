import sys
import win32api
import win32con
import win32ui
NotScriptMsg = 'This demo program is not designed to be run as a Script, but is\nprobably used by some other test program.  Please try another demo.\n'
NeedGUIMsg = "This demo program can only be run from inside of Pythonwin\n\nYou must start Pythonwin, and select 'Run' from the toolbar or File menu\n"
NeedAppMsg = 'This demo program is a \'Pythonwin Application\'.\n\nIt is more demo code than an example of Pythonwin\'s capabilities.\n\nTo run it, you must execute the command:\npythonwin.exe /app "%s"\n\nWould you like to execute it now?\n'

def NotAScript():
    if False:
        return 10
    import win32ui
    win32ui.MessageBox(NotScriptMsg, 'Demos')

def NeedGoodGUI():
    if False:
        for i in range(10):
            print('nop')
    from pywin.framework.app import HaveGoodGUI
    rc = HaveGoodGUI()
    if not rc:
        win32ui.MessageBox(NeedGUIMsg, 'Demos')
    return rc

def NeedApp():
    if False:
        for i in range(10):
            print('nop')
    import win32ui
    rc = win32ui.MessageBox(NeedAppMsg % sys.argv[0], 'Demos', win32con.MB_YESNO)
    if rc == win32con.IDYES:
        try:
            parent = win32ui.GetMainFrame().GetSafeHwnd()
            win32api.ShellExecute(parent, None, 'pythonwin.exe', '/app "%s"' % sys.argv[0], None, 1)
        except win32api.error as details:
            win32ui.MessageBox('Error executing command - %s' % details, 'Demos')
if __name__ == '__main__':
    import demoutils
    demoutils.NotAScript()