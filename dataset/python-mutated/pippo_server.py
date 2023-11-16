import os
import sys
import pythoncom
import win32com
import winerror
from win32com.server.util import wrap

class CPippo:
    _reg_clsid_ = '{1F0F75D6-BD63-41B9-9F88-2D9D2E1AA5C3}'
    _reg_desc_ = 'Pippo Python test object'
    _reg_progid_ = 'Python.Test.Pippo'
    _typelib_guid_ = '{7783054E-9A20-4584-8C62-6ED2A08F6AC6}'
    _typelib_version_ = (1, 0)
    _com_interfaces_ = ['IPippo']

    def __init__(self):
        if False:
            print('Hello World!')
        self.MyProp1 = 10

    def Method1(self):
        if False:
            for i in range(10):
                print('nop')
        return wrap(CPippo())

    def Method2(self, in1, inout1):
        if False:
            print('Hello World!')
        return (in1, inout1 * 2)

    def Method3(self, in1):
        if False:
            print('Hello World!')
        return list(in1)

def BuildTypelib():
    if False:
        return 10
    from distutils.dep_util import newer
    this_dir = os.path.dirname(__file__)
    idl = os.path.abspath(os.path.join(this_dir, 'pippo.idl'))
    tlb = os.path.splitext(idl)[0] + '.tlb'
    if newer(idl, tlb):
        print(f'Compiling {idl}')
        rc = os.system(f'midl "{idl}"')
        if rc:
            raise RuntimeError('Compiling MIDL failed!')
        for fname in 'dlldata.c pippo_i.c pippo_p.c pippo.h'.split():
            os.remove(os.path.join(this_dir, fname))
    print(f'Registering {tlb}')
    tli = pythoncom.LoadTypeLib(tlb)
    pythoncom.RegisterTypeLib(tli, tlb)

def UnregisterTypelib():
    if False:
        for i in range(10):
            print('nop')
    k = CPippo
    try:
        pythoncom.UnRegisterTypeLib(k._typelib_guid_, k._typelib_version_[0], k._typelib_version_[1], 0, pythoncom.SYS_WIN32)
        print('Unregistered typelib')
    except pythoncom.error as details:
        if details[0] == winerror.TYPE_E_REGISTRYACCESS:
            pass
        else:
            raise

def main(argv=None):
    if False:
        print('Hello World!')
    if argv is None:
        argv = sys.argv[1:]
    if '--unregister' in argv:
        UnregisterTypelib()
    else:
        BuildTypelib()
    import win32com.server.register
    win32com.server.register.UseCommandLine(CPippo)
if __name__ == '__main__':
    main(sys.argv)