from win32com.axscript import axscript
from . import pyscript
from .pyscript import SCRIPTTEXT_FORCEEXECUTION, Exception, RaiseAssert, trace
PyDump_CLSID = '{ac527e60-c693-11d0-9c25-00aa00125a98}'

class AXScriptAttribute(pyscript.AXScriptAttribute):
    pass

class NamedScriptAttribute(pyscript.NamedScriptAttribute):
    pass

class PyScript(pyscript.PyScript):
    pass

def Register():
    if False:
        for i in range(10):
            print('nop')
    import sys
    if '-d' in sys.argv:
        dispatcher = 'DispatcherWin32trace'
        debug_desc = ' (' + dispatcher + ')'
        debug_option = 'Yes'
    else:
        dispatcher = None
        debug_desc = ''
        debug_option = ''
    categories = [axscript.CATID_ActiveScript, axscript.CATID_ActiveScriptParse]
    clsid = PyDump_CLSID
    lcid = 1033
    policy = None
    print('Registering COM server%s...' % debug_desc)
    from win32com.server.register import RegisterServer
    languageName = 'PyDump'
    verProgId = 'Python.Dumper.1'
    RegisterServer(clsid=clsid, pythonInstString='win32com.axscript.client.pyscript.PyDumper', className='Python Debugging/Dumping ActiveX Scripting Engine', progID=languageName, verProgID=verProgId, catids=categories, policy=policy, dispatcher=dispatcher)
    CreateRegKey(languageName + '\\OLEScript')
    win32com.server.register._set_string('.pysDump', 'pysDumpFile')
    win32com.server.register._set_string('pysDumpFile\\ScriptEngine', languageName)
    print('Dumping Server registered.')
if __name__ == '__main__':
    Register()