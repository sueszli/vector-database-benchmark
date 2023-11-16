import pythoncom
from win32com.server.util import wrap
from .util import CheckClean
numErrors = 0

def CheckSameCOMObject(ob1, ob2):
    if False:
        return 10
    addr1 = repr(ob1).split()[6][:-1]
    addr2 = repr(ob2).split()[6][:-1]
    return addr1 == addr2

def CheckObjectIdentity(ob1, ob2):
    if False:
        for i in range(10):
            print('nop')
    u1 = ob1.QueryInterface(pythoncom.IID_IUnknown)
    u2 = ob2.QueryInterface(pythoncom.IID_IUnknown)
    return CheckSameCOMObject(u1, u2)

def FailObjectIdentity(ob1, ob2, when):
    if False:
        for i in range(10):
            print('nop')
    if not CheckObjectIdentity(ob1, ob2):
        global numErrors
        numErrors = numErrors + 1
        print(when, f'are not identical ({repr(ob1)}, {repr(ob2)})')

class Dummy:
    _public_methods_ = []
    _com_interfaces_ = [pythoncom.IID_IPersistStorage]

class Dummy2:
    _public_methods_ = []
    _com_interfaces_ = [pythoncom.IID_IPersistStorage, pythoncom.IID_IExternalConnection]

class DeletgatedDummy:
    _public_methods_ = []

class Dummy3:
    _public_methods_ = []
    _com_interfaces_ = [pythoncom.IID_IPersistStorage]

    def _query_interface_(self, iid):
        if False:
            for i in range(10):
                print('nop')
        if iid == pythoncom.IID_IExternalConnection:
            return wrap(DelegatedDummy())

def TestGatewayInheritance():
    if False:
        while True:
            i = 10
    o = wrap(Dummy(), pythoncom.IID_IPersistStorage)
    o2 = o.QueryInterface(pythoncom.IID_IUnknown)
    FailObjectIdentity(o, o2, 'IID_IPersistStorage->IID_IUnknown')
    o3 = o2.QueryInterface(pythoncom.IID_IDispatch)
    FailObjectIdentity(o2, o3, 'IID_IUnknown->IID_IDispatch')
    FailObjectIdentity(o, o3, 'IID_IPersistStorage->IID_IDispatch')
    o4 = o3.QueryInterface(pythoncom.IID_IPersistStorage)
    FailObjectIdentity(o, o4, 'IID_IPersistStorage->IID_IPersistStorage(2)')
    FailObjectIdentity(o2, o4, 'IID_IUnknown->IID_IPersistStorage(2)')
    FailObjectIdentity(o3, o4, 'IID_IDispatch->IID_IPersistStorage(2)')
    o5 = o4.QueryInterface(pythoncom.IID_IPersist)
    FailObjectIdentity(o, o5, 'IID_IPersistStorage->IID_IPersist')
    FailObjectIdentity(o2, o5, 'IID_IUnknown->IID_IPersist')
    FailObjectIdentity(o3, o5, 'IID_IDispatch->IID_IPersist')
    FailObjectIdentity(o4, o5, 'IID_IPersistStorage(2)->IID_IPersist')

def TestMultiInterface():
    if False:
        return 10
    o = wrap(Dummy2(), pythoncom.IID_IPersistStorage)
    o2 = o.QueryInterface(pythoncom.IID_IExternalConnection)
    FailObjectIdentity(o, o2, 'IID_IPersistStorage->IID_IExternalConnection')
    o22 = o.QueryInterface(pythoncom.IID_IExternalConnection)
    FailObjectIdentity(o, o22, 'IID_IPersistStorage->IID_IExternalConnection')
    FailObjectIdentity(o2, o22, 'IID_IPersistStorage->IID_IExternalConnection (stability)')
    o3 = o2.QueryInterface(pythoncom.IID_IPersistStorage)
    FailObjectIdentity(o2, o3, 'IID_IExternalConnection->IID_IPersistStorage')
    FailObjectIdentity(o, o3, 'IID_IPersistStorage->IID_IExternalConnection->IID_IPersistStorage')

def test():
    if False:
        return 10
    TestGatewayInheritance()
    TestMultiInterface()
    if numErrors == 0:
        print('Worked ok')
    else:
        print('There were', numErrors, 'errors.')
if __name__ == '__main__':
    test()
    CheckClean()