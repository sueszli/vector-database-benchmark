import pythoncom
import winerror
from win32com import universal
from win32com.client import constants, gencache
from win32com.server.exception import COMException
from win32com.server.util import wrap
pythoncom.__future_currency__ = True
gencache.EnsureModule('{6BCDCB60-5605-11D0-AE5F-CADD4C000000}', 0, 1, 1)

class PyCOMTest:
    _typelib_guid_ = '{6BCDCB60-5605-11D0-AE5F-CADD4C000000}'
    _typelib_version = (1, 0)
    _com_interfaces_ = ['IPyCOMTest']
    _reg_clsid_ = '{e743d9cd-cb03-4b04-b516-11d3a81c1597}'
    _reg_progid_ = 'Python.Test.PyCOMTest'

    def DoubleString(self, str):
        if False:
            return 10
        return str * 2

    def DoubleInOutString(self, str):
        if False:
            for i in range(10):
                print('nop')
        return str * 2

    def Fire(self, nID):
        if False:
            for i in range(10):
                print('nop')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetLastVarArgs(self):
        if False:
            i = 10
            return i + 15
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetMultipleInterfaces(self, outinterface1, outinterface2):
        if False:
            return 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetSafeArrays(self, attrs, attrs2, ints):
        if False:
            while True:
                i = 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetSetDispatch(self, indisp):
        if False:
            return 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetSetInterface(self, ininterface):
        if False:
            while True:
                i = 10
        return wrap(self)

    def GetSetVariant(self, indisp):
        if False:
            for i in range(10):
                print('nop')
        return indisp

    def TestByRefVariant(self, v):
        if False:
            for i in range(10):
                print('nop')
        return v * 2

    def TestByRefString(self, v):
        if False:
            for i in range(10):
                print('nop')
        return v * 2

    def GetSetInterfaceArray(self, ininterface):
        if False:
            return 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetSetUnknown(self, inunk):
        if False:
            return 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetSimpleCounter(self):
        if False:
            while True:
                i = 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetSimpleSafeArray(self, ints):
        if False:
            while True:
                i = 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetStruct(self):
        if False:
            i = 10
            return i + 15
        raise COMException(hresult=winerror.E_NOTIMPL)

    def SetIntSafeArray(self, ints):
        if False:
            return 10
        return len(ints)

    def SetLongLongSafeArray(self, ints):
        if False:
            print('Hello World!')
        return len(ints)

    def SetULongLongSafeArray(self, ints):
        if False:
            for i in range(10):
                print('nop')
        return len(ints)

    def SetBinSafeArray(self, buf):
        if False:
            for i in range(10):
                print('nop')
        return len(buf)

    def SetVarArgs(self, *args):
        if False:
            while True:
                i = 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def SetVariantSafeArray(self, vars):
        if False:
            i = 10
            return i + 15
        raise COMException(hresult=winerror.E_NOTIMPL)

    def Start(self):
        if False:
            for i in range(10):
                print('nop')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def Stop(self, nID):
        if False:
            return 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def StopAll(self):
        if False:
            i = 10
            return i + 15
        raise COMException(hresult=winerror.E_NOTIMPL)

    def TakeByRefDispatch(self, inout):
        if False:
            i = 10
            return i + 15
        raise COMException(hresult=winerror.E_NOTIMPL)

    def TakeByRefTypedDispatch(self, inout):
        if False:
            print('Hello World!')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def Test(self, key, inval):
        if False:
            print('Hello World!')
        return not inval

    def Test2(self, inval):
        if False:
            return 10
        return inval

    def Test3(self, inval):
        if False:
            print('Hello World!')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def Test4(self, inval):
        if False:
            for i in range(10):
                print('nop')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def Test5(self, inout):
        if False:
            print('Hello World!')
        if inout == constants.TestAttr1:
            return constants.TestAttr1_1
        elif inout == constants.TestAttr1_1:
            return constants.TestAttr1
        else:
            return -1

    def Test6(self, inval):
        if False:
            i = 10
            return i + 15
        return inval

    def TestInOut(self, fval, bval, lval):
        if False:
            while True:
                i = 10
        return (winerror.S_OK, fval * 2, not bval, lval * 2)

    def TestOptionals(self, strArg='def', sval=0, lval=1, dval=3.140000104904175):
        if False:
            i = 10
            return i + 15
        raise COMException(hresult=winerror.E_NOTIMPL)

    def TestOptionals2(self, dval, strval='', sval=1):
        if False:
            return 10
        raise COMException(hresult=winerror.E_NOTIMPL)

    def CheckVariantSafeArray(self, data):
        if False:
            print('Hello World!')
        return 1

    def LongProp(self):
        if False:
            while True:
                i = 10
        return self.longval

    def SetLongProp(self, val):
        if False:
            print('Hello World!')
        self.longval = val

    def ULongProp(self):
        if False:
            i = 10
            return i + 15
        return self.ulongval

    def SetULongProp(self, val):
        if False:
            return 10
        self.ulongval = val

    def IntProp(self):
        if False:
            print('Hello World!')
        return self.intval

    def SetIntProp(self, val):
        if False:
            i = 10
            return i + 15
        self.intval = val

class PyCOMTestMI(PyCOMTest):
    _typelib_guid_ = '{6BCDCB60-5605-11D0-AE5F-CADD4C000000}'
    _typelib_version = (1, 0)
    _com_interfaces_ = ['IPyCOMTest', pythoncom.IID_IStream, str(pythoncom.IID_IStorage)]
    _reg_clsid_ = '{F506E9A1-FB46-4238-A597-FA4EB69787CA}'
    _reg_progid_ = 'Python.Test.PyCOMTestMI'
if __name__ == '__main__':
    import win32com.server.register
    win32com.server.register.UseCommandLine(PyCOMTest)
    win32com.server.register.UseCommandLine(PyCOMTestMI)