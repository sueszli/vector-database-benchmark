import unittest
import pythoncom
import win32com.client.dynamic
import win32com.test.util
import winerror

def TestConnections():
    if False:
        for i in range(10):
            print('nop')
    import win32com.demos.connect
    win32com.demos.connect.test()

class InterpCase(win32com.test.util.TestCase):

    def setUp(self):
        if False:
            return 10
        from win32com.servers import interp
        from win32com.test.util import RegisterPythonServer
        RegisterPythonServer(interp.__file__, 'Python.Interpreter')

    def _testInterp(self, interp):
        if False:
            while True:
                i = 10
        self.assertEqual(interp.Eval('1+1'), 2)
        win32com.test.util.assertRaisesCOM_HRESULT(self, winerror.DISP_E_TYPEMISMATCH, interp.Eval, 2)

    def testInproc(self):
        if False:
            while True:
                i = 10
        interp = win32com.client.dynamic.Dispatch('Python.Interpreter', clsctx=pythoncom.CLSCTX_INPROC)
        self._testInterp(interp)

    def testLocalServer(self):
        if False:
            return 10
        interp = win32com.client.dynamic.Dispatch('Python.Interpreter', clsctx=pythoncom.CLSCTX_LOCAL_SERVER)
        self._testInterp(interp)

    def testAny(self):
        if False:
            while True:
                i = 10
        interp = win32com.client.dynamic.Dispatch('Python.Interpreter')
        self._testInterp(interp)

class ConnectionsTestCase(win32com.test.util.TestCase):

    def testConnections(self):
        if False:
            return 10
        TestConnections()
if __name__ == '__main__':
    unittest.main('testServers')