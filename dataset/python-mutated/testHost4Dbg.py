import os
import sys
import traceback
import pythoncom
import win32ui
from win32com.axscript import axscript
from win32com.axscript.server import axsite
from win32com.axscript.server.error import Exception
from win32com.server import util
version = '0.0.1'

class MySite(axsite.AXSite):

    def OnScriptError(self, error):
        if False:
            return 10
        print('An error occurred in the Script Code')
        exc = error.GetExceptionInfo()
        try:
            text = error.GetSourceLineText()
        except:
            text = '<unknown>'
        (context, line, char) = error.GetSourcePosition()
        print('Exception: %s (line %d)\n%s\n%s^\n%s' % (exc[1], line, text, ' ' * (char - 1), exc[2]))

class ObjectModel:
    _public_methods_ = ['echo', 'msgbox']

    def echo(self, *args):
        if False:
            return 10
        print(''.join(map(str, args)))

    def msgbox(self, *args):
        if False:
            return 10
        msg = ''.join(map(str, args))
        win32ui.MessageBox(msg)

def TestEngine():
    if False:
        while True:
            i = 10
    model = {'Test': util.wrap(ObjectModel())}
    scriptDir = '.'
    site = MySite(model)
    pyEngine = site._AddEngine('Python')
    vbEngine = site._AddEngine('VBScript')
    try:
        code = open(os.path.join(scriptDir, 'debugTest.pys'), 'rb').read()
        pyEngine.AddCode(code)
        code = open(os.path.join(scriptDir, 'debugTest.vbs'), 'rb').read()
        vbEngine.AddCode(code)
        input('Press enter to continue')
        pyEngine.Start()
        vbEngine.Start()
    except pythoncom.com_error as details:
        print(f'Script failed: {details[1]} (0x{details[0]:x})')
    site._Close()
if __name__ == '__main__':
    import win32com.axdebug.util
    try:
        TestEngine()
    except:
        traceback.print_exc()
    sys.exc_type = sys.exc_value = sys.exc_traceback = None
    print(pythoncom._GetInterfaceCount(), 'com objects still alive')