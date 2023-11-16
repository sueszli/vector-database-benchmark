import sys
import unittest
import pythoncom
import win32com.server.policy
import win32com.test.util
from win32com.axscript import axscript
from win32com.axscript.server import axsite
from win32com.axscript.server.error import Exception
from win32com.client.dynamic import Dispatch
from win32com.server import connect, util
from win32com.server.exception import COMException
verbose = '-v' in sys.argv

class MySite(axsite.AXSite):

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self.exception_seen = None
        axsite.AXSite.__init__(self, *args)

    def OnScriptError(self, error):
        if False:
            while True:
                i = 10
        self.exception_seen = exc = error.GetExceptionInfo()
        (context, line, char) = error.GetSourcePosition()
        if not verbose:
            return
        print(' >Exception:', exc[1])
        try:
            st = error.GetSourceLineText()
        except pythoncom.com_error:
            st = None
        if st is None:
            st = ''
        text = st + '\n' + ' ' * (char - 1) + '^' + '\n' + exc[2]
        for line in text.splitlines():
            print('  >' + line)

class MyCollection(util.Collection):

    def _NewEnum(self):
        if False:
            return 10
        return util.Collection._NewEnum(self)

class Test:
    _public_methods_ = ['echo', 'fail']
    _public_attrs_ = ['collection']

    def __init__(self):
        if False:
            return 10
        self.verbose = verbose
        self.collection = util.wrap(MyCollection([1, 'Two', 3]))
        self.last = ''
        self.fail_called = 0

    def echo(self, *args):
        if False:
            i = 10
            return i + 15
        self.last = ''.join([str(s) for s in args])
        if self.verbose:
            for arg in args:
                print(arg, end=' ')
            print()

    def fail(self, *args):
        if False:
            for i in range(10):
                print('nop')
        print('**** fail() called ***')
        for arg in args:
            print(arg, end=' ')
        print()
        self.fail_called = 1
IID_ITestEvents = pythoncom.MakeIID('{8EB72F90-0D44-11d1-9C4B-00AA00125A98}')

class TestConnectServer(connect.ConnectableServer):
    _connect_interfaces_ = [IID_ITestEvents]

    def __init__(self, object):
        if False:
            for i in range(10):
                print('nop')
        self.object = object

    def Broadcast(self, arg):
        if False:
            return 10
        self._BroadcastNotify(self.NotifyDoneIt, (arg,))

    def NotifyDoneIt(self, interface, arg):
        if False:
            return 10
        interface.Invoke(1000, 0, pythoncom.DISPATCH_METHOD, 1, arg)
VBScript = 'prop = "Property Value"\n\nsub hello(arg1)\n   test.echo arg1\nend sub\n  \nsub testcollection\n   if test.collection.Item(0) <> 1 then\n     test.fail("Index 0 was wrong")\n   end if\n   if test.collection.Item(1) <> "Two" then\n     test.fail("Index 1 was wrong")\n   end if\n   if test.collection.Item(2) <> 3 then\n     test.fail("Index 2 was wrong")\n   end if\n   num = 0\n   for each item in test.collection\n     num = num + 1\n   next\n   if num <> 3 then\n     test.fail("Collection didn\'t have 3 items")\n   end if\nend sub\n'
PyScript = '# A unicode ©omment.\nprop = "Property Value"\ndef hello(arg1):\n   test.echo(arg1)\n   \ndef testcollection():\n#   test.collection[1] = "New one"\n   got = []\n   for item in test.collection:\n     got.append(item)\n   if got != [1, "Two", 3]:\n     test.fail("Didn\'t get the collection")\n   pass\n'
ErrScript = 'bad code for everyone!\n'
state_map = {axscript.SCRIPTSTATE_UNINITIALIZED: 'SCRIPTSTATE_UNINITIALIZED', axscript.SCRIPTSTATE_INITIALIZED: 'SCRIPTSTATE_INITIALIZED', axscript.SCRIPTSTATE_STARTED: 'SCRIPTSTATE_STARTED', axscript.SCRIPTSTATE_CONNECTED: 'SCRIPTSTATE_CONNECTED', axscript.SCRIPTSTATE_DISCONNECTED: 'SCRIPTSTATE_DISCONNECTED', axscript.SCRIPTSTATE_CLOSED: 'SCRIPTSTATE_CLOSED'}

def _CheckEngineState(engine, name, state):
    if False:
        print('Hello World!')
    got = engine.engine.eScript.GetScriptState()
    if got != state:
        got_name = state_map.get(got, str(got))
        state_name = state_map.get(state, str(state))
        raise RuntimeError(f'Warning - engine {name} has state {got_name}, but expected {state_name}')

class EngineTester(win32com.test.util.TestCase):

    def _TestEngine(self, engineName, code, expected_exc=None):
        if False:
            return 10
        echoer = Test()
        model = {'test': util.wrap(echoer)}
        site = MySite(model)
        engine = site._AddEngine(engineName)
        try:
            _CheckEngineState(site, engineName, axscript.SCRIPTSTATE_INITIALIZED)
            engine.AddCode(code)
            engine.Start()
            _CheckEngineState(site, engineName, axscript.SCRIPTSTATE_STARTED)
            self.assertTrue(not echoer.fail_called, 'Fail should not have been called')
            ob = Dispatch(engine.GetScriptDispatch())
            try:
                ob.hello('Goober')
                self.assertTrue(expected_exc is None, f'Expected {expected_exc!r}, but no exception seen')
            except pythoncom.com_error:
                if expected_exc is None:
                    self.fail(f'Unexpected failure from script code: {site.exception_seen}')
                if expected_exc not in site.exception_seen[2]:
                    self.fail(f'Could not find {expected_exc!r} in {site.exception_seen[2]!r}')
                return
            self.assertEqual(echoer.last, 'Goober')
            self.assertEqual(str(ob.prop), 'Property Value')
            ob.testcollection()
            self.assertTrue(not echoer.fail_called, 'Fail should not have been called')
            result = engine.eParse.ParseScriptText('1+1', None, None, None, 0, 0, axscript.SCRIPTTEXT_ISEXPRESSION)
            self.assertEqual(result, 2)
            engine.SetScriptState(axscript.SCRIPTSTATE_INITIALIZED)
            _CheckEngineState(site, engineName, axscript.SCRIPTSTATE_INITIALIZED)
            engine.Start()
            _CheckEngineState(site, engineName, axscript.SCRIPTSTATE_STARTED)
            engine.SetScriptState(axscript.SCRIPTSTATE_INITIALIZED)
            _CheckEngineState(site, engineName, axscript.SCRIPTSTATE_INITIALIZED)
            engine.SetScriptState(axscript.SCRIPTSTATE_CONNECTED)
            _CheckEngineState(site, engineName, axscript.SCRIPTSTATE_CONNECTED)
            engine.SetScriptState(axscript.SCRIPTSTATE_INITIALIZED)
            _CheckEngineState(site, engineName, axscript.SCRIPTSTATE_INITIALIZED)
            engine.SetScriptState(axscript.SCRIPTSTATE_CONNECTED)
            _CheckEngineState(site, engineName, axscript.SCRIPTSTATE_CONNECTED)
            engine.SetScriptState(axscript.SCRIPTSTATE_DISCONNECTED)
            _CheckEngineState(site, engineName, axscript.SCRIPTSTATE_DISCONNECTED)
        finally:
            engine.Close()
            engine = None
            site = None

    def testVB(self):
        if False:
            print('Hello World!')
        self._TestEngine('VBScript', VBScript)

    def testPython(self):
        if False:
            return 10
        self._TestEngine('Python', PyScript)

    def testPythonUnicodeError(self):
        if False:
            for i in range(10):
                print('nop')
        self._TestEngine('Python', PyScript)

    def testVBExceptions(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(pythoncom.com_error, self._TestEngine, 'VBScript', ErrScript)
if __name__ == '__main__':
    unittest.main()