import pythoncom
from win32com.axscript import axscript
from . import pyscript
INTERFACE_USES_DISPEX = 4
INTERFACE_USES_SECURITY_MANAGER = 8

class PyScriptRExec(pyscript.PyScript):
    _reg_verprogid_ = 'Python.AXScript-rexec.2'
    _reg_progid_ = 'Python'
    _reg_catids_ = [axscript.CATID_ActiveScript, axscript.CATID_ActiveScriptParse]
    _reg_desc_ = 'Python ActiveX Scripting Engine (with rexec support)'
    _reg_clsid_ = '{69c2454b-efa2-455b-988c-c3651c4a2f69}'
    _reg_class_spec_ = 'win32com.axscript.client.pyscript_rexec.PyScriptRExec'
    _reg_remove_keys_ = [('.pys',), ('pysFile',)]
    _reg_threading_ = 'Apartment'

    def _GetSupportedInterfaceSafetyOptions(self):
        if False:
            i = 10
            return i + 15
        return INTERFACE_USES_DISPEX | INTERFACE_USES_SECURITY_MANAGER | axscript.INTERFACESAFE_FOR_UNTRUSTED_DATA | axscript.INTERFACESAFE_FOR_UNTRUSTED_CALLER
if __name__ == '__main__':
    print('WARNING: By registering this engine, you are giving remote HTML code')
    print('the ability to execute *any* code on your system.')
    print()
    print('You almost certainly do NOT want to do this.')
    print('You have been warned, and are doing this at your own (significant) risk')
    pyscript.Register(PyScriptRExec)