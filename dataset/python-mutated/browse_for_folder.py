import os
import win32gui
from win32com.shell import shell, shellcon

def BrowseCallbackProc(hwnd, msg, lp, data):
    if False:
        i = 10
        return i + 15
    if msg == shellcon.BFFM_INITIALIZED:
        win32gui.SendMessage(hwnd, shellcon.BFFM_SETSELECTION, 1, data)
    elif msg == shellcon.BFFM_SELCHANGED:
        pidl = shell.AddressAsPIDL(lp)
        try:
            path = shell.SHGetPathFromIDList(pidl)
            win32gui.SendMessage(hwnd, shellcon.BFFM_SETSTATUSTEXT, 0, path)
        except shell.error:
            pass
if __name__ == '__main__':
    flags = shellcon.BIF_STATUSTEXT
    shell.SHBrowseForFolder(0, None, 'Default of %s' % os.getcwd(), flags, BrowseCallbackProc, os.getcwd())
    desktop = shell.SHGetDesktopFolder()
    (cb, pidl, extra) = desktop.ParseDisplayName(0, None, os.getcwd())
    shell.SHBrowseForFolder(0, pidl, 'From %s down only' % os.getcwd())