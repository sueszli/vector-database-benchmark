import sys
import pythoncom
import win32com.server.policy
from win32com.axscript import axscript
from win32com.axscript.server import axsite
from win32com.axscript.server.error import Exception
from win32com.server import connect, util

class MySite(axsite.AXSite):

    def OnScriptError(self, error):
        if False:
            return 10
        exc = error.GetExceptionInfo()
        (context, line, char) = error.GetSourcePosition()
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
            print('Hello World!')
        print('Making new Enumerator')
        return util.Collection._NewEnum(self)

class Test:
    _public_methods_ = ['echo']
    _public_attrs_ = ['collection', 'verbose']

    def __init__(self):
        if False:
            print('Hello World!')
        self.verbose = 0
        self.collection = util.wrap(MyCollection([1, 'Two', 3]))
        self.last = ''

    def echo(self, *args):
        if False:
            i = 10
            return i + 15
        self.last = ''.join(map(str, args))
        if self.verbose:
            for arg in args:
                print(arg, end=' ')
            print()
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
            while True:
                i = 10
        self._BroadcastNotify(self.NotifyDoneIt, (arg,))

    def NotifyDoneIt(self, interface, arg):
        if False:
            while True:
                i = 10
        interface.Invoke(1000, 0, pythoncom.DISPATCH_METHOD, 1, arg)
VBScript = 'prop = "Property Value"\n\nsub hello(arg1)\n   test.echo arg1\nend sub\n\nsub testcollection\n   test.verbose = 1\n   for each item in test.collection\n     test.echo "Collection item is", item\n   next\nend sub\n'
PyScript = 'print("PyScript is being parsed...")\n\n\nprop = "Property Value"\ndef hello(arg1):\n   test.echo(arg1)\n   pass\n\ndef testcollection():\n   test.verbose = 1\n#   test.collection[1] = "New one"\n   for item in test.collection:\n     test.echo("Collection item is", item)\n   pass\n'
ErrScript = 'bad code for everyone!\n'

def TestEngine(engineName, code, bShouldWork=1):
    if False:
        print('Hello World!')
    echoer = Test()
    model = {'test': util.wrap(echoer)}
    site = MySite(model)
    engine = site._AddEngine(engineName)
    engine.AddCode(code, axscript.SCRIPTTEXT_ISPERSISTENT)
    try:
        engine.Start()
    finally:
        if not bShouldWork:
            engine.Close()
            return
    doTestEngine(engine, echoer)
    engine.eScript.SetScriptState(axscript.SCRIPTSTATE_UNINITIALIZED)
    engine.eScript.SetScriptSite(util.wrap(site))
    print('restarting')
    engine.Start()
    engine.Close()

def doTestEngine(engine, echoer):
    if False:
        print('Hello World!')
    from win32com.client.dynamic import Dispatch
    ob = Dispatch(engine.GetScriptDispatch())
    try:
        ob.hello('Goober')
    except pythoncom.com_error as exc:
        print("***** Calling 'hello' failed", exc)
        return
    if echoer.last != 'Goober':
        print('***** Function call didnt set value correctly', repr(echoer.last))
    if str(ob.prop) != 'Property Value':
        print('***** Property Value not correct - ', repr(ob.prop))
    ob.testcollection()
    result = engine.eParse.ParseScriptText('1+1', None, None, None, 0, 0, axscript.SCRIPTTEXT_ISEXPRESSION)
    if result != 2:
        print("Engine could not evaluate '1+1' - said the result was", result)

def dotestall():
    if False:
        print('Hello World!')
    for i in range(10):
        TestEngine('Python', PyScript)
        print(sys.gettotalrefcount())

def testall():
    if False:
        for i in range(10):
            print('nop')
    dotestall()
    pythoncom.CoUninitialize()
    print('AXScript Host worked correctly - %d/%d COM objects left alive.' % (pythoncom._GetInterfaceCount(), pythoncom._GetGatewayCount()))
if __name__ == '__main__':
    testall()