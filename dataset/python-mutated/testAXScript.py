import os
import sys
import win32api
import win32com.axscript
import win32com.axscript.client
import win32com.test.util
verbose = '-v' in sys.argv

class AXScript(win32com.test.util.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        file = win32api.GetFullPathName(os.path.join(win32com.axscript.client.__path__[0], 'pyscript.py'))
        from win32com.test.util import RegisterPythonServer
        self.verbose = verbose
        RegisterPythonServer(file, 'python', verbose=self.verbose)

    def testHost(self):
        if False:
            while True:
                i = 10
        file = win32api.GetFullPathName(os.path.join(win32com.axscript.__path__[0], 'test\\testHost.py'))
        cmd = f'{win32api.GetModuleFileName(0)} "{file}"'
        if verbose:
            print('Testing Python Scripting host')
        win32com.test.util.ExecuteShellCommand(cmd, self)

    def testCScript(self):
        if False:
            print('Hello World!')
        file = win32api.GetFullPathName(os.path.join(win32com.axscript.__path__[0], 'Demos\\Client\\wsh\\test.pys'))
        cmd = 'cscript.exe "%s"' % file
        if verbose:
            print('Testing Windows Scripting host with Python script')
        win32com.test.util.ExecuteShellCommand(cmd, self)
if __name__ == '__main__':
    win32com.test.util.testmain()