import unittest
import pythoncom
import win32clipboard
import win32con
import winerror
from win32com.server.exception import COMException
from win32com.server.util import NewEnum, wrap
IDataObject_Methods = 'GetData GetDataHere QueryGetData\n                         GetCanonicalFormatEtc SetData EnumFormatEtc\n                         DAdvise DUnadvise EnumDAdvise'.split()
num_do_objects = 0

def WrapCOMObject(ob, iid=None):
    if False:
        for i in range(10):
            print('nop')
    return wrap(ob, iid=iid, useDispatcher=0)

class TestDataObject:
    _com_interfaces_ = [pythoncom.IID_IDataObject]
    _public_methods_ = IDataObject_Methods

    def __init__(self, bytesval):
        if False:
            i = 10
            return i + 15
        global num_do_objects
        num_do_objects += 1
        self.bytesval = bytesval
        self.supported_fe = []
        for cf in (win32con.CF_TEXT, win32con.CF_UNICODETEXT):
            fe = (cf, None, pythoncom.DVASPECT_CONTENT, -1, pythoncom.TYMED_HGLOBAL)
            self.supported_fe.append(fe)

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        global num_do_objects
        num_do_objects -= 1

    def _query_interface_(self, iid):
        if False:
            i = 10
            return i + 15
        if iid == pythoncom.IID_IEnumFORMATETC:
            return NewEnum(self.supported_fe, iid=iid)

    def GetData(self, fe):
        if False:
            while True:
                i = 10
        ret_stg = None
        (cf, target, aspect, index, tymed) = fe
        if aspect & pythoncom.DVASPECT_CONTENT and tymed == pythoncom.TYMED_HGLOBAL:
            if cf == win32con.CF_TEXT:
                ret_stg = pythoncom.STGMEDIUM()
                ret_stg.set(pythoncom.TYMED_HGLOBAL, self.bytesval)
            elif cf == win32con.CF_UNICODETEXT:
                ret_stg = pythoncom.STGMEDIUM()
                ret_stg.set(pythoncom.TYMED_HGLOBAL, self.bytesval.decode('latin1'))
        if ret_stg is None:
            raise COMException(hresult=winerror.E_NOTIMPL)
        return ret_stg

    def GetDataHere(self, fe):
        if False:
            for i in range(10):
                print('nop')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def QueryGetData(self, fe):
        if False:
            while True:
                i = 10
        (cf, target, aspect, index, tymed) = fe
        if aspect & pythoncom.DVASPECT_CONTENT == 0:
            raise COMException(hresult=winerror.DV_E_DVASPECT)
        if tymed != pythoncom.TYMED_HGLOBAL:
            raise COMException(hresult=winerror.DV_E_TYMED)
        return None

    def GetCanonicalFormatEtc(self, fe):
        if False:
            i = 10
            return i + 15
        RaiseCOMException(winerror.DATA_S_SAMEFORMATETC)

    def SetData(self, fe, medium):
        if False:
            print('Hello World!')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def EnumFormatEtc(self, direction):
        if False:
            i = 10
            return i + 15
        if direction != pythoncom.DATADIR_GET:
            raise COMException(hresult=winerror.E_NOTIMPL)
        return NewEnum(self.supported_fe, iid=pythoncom.IID_IEnumFORMATETC)

    def DAdvise(self, fe, flags, sink):
        if False:
            for i in range(10):
                print('nop')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def DUnadvise(self, connection):
        if False:
            print('Hello World!')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def EnumDAdvise(self):
        if False:
            i = 10
            return i + 15
        raise COMException(hresult=winerror.E_NOTIMPL)

class ClipboardTester(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        pythoncom.OleInitialize()

    def tearDown(self):
        if False:
            print('Hello World!')
        try:
            pythoncom.OleFlushClipboard()
        except pythoncom.com_error:
            pass

    def testIsCurrentClipboard(self):
        if False:
            for i in range(10):
                print('nop')
        do = TestDataObject(b'Hello from Python')
        do = WrapCOMObject(do, iid=pythoncom.IID_IDataObject)
        pythoncom.OleSetClipboard(do)
        self.assertTrue(pythoncom.OleIsCurrentClipboard(do))

    def testComToWin32(self):
        if False:
            print('Hello World!')
        do = TestDataObject(b'Hello from Python')
        do = WrapCOMObject(do, iid=pythoncom.IID_IDataObject)
        pythoncom.OleSetClipboard(do)
        win32clipboard.OpenClipboard()
        got = win32clipboard.GetClipboardData(win32con.CF_TEXT)
        expected = b'Hello from Python'
        self.assertEqual(got, expected)
        got = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
        self.assertEqual(got, 'Hello from Python')
        win32clipboard.CloseClipboard()

    def testWin32ToCom(self):
        if False:
            while True:
                i = 10
        val = b'Hello again!'
        win32clipboard.OpenClipboard()
        win32clipboard.SetClipboardData(win32con.CF_TEXT, val)
        win32clipboard.CloseClipboard()
        do = pythoncom.OleGetClipboard()
        cf = (win32con.CF_TEXT, None, pythoncom.DVASPECT_CONTENT, -1, pythoncom.TYMED_HGLOBAL)
        stg = do.GetData(cf)
        got = stg.data
        self.assertTrue(got, b'Hello again!\x00')

    def testDataObjectFlush(self):
        if False:
            print('Hello World!')
        do = TestDataObject(b'Hello from Python')
        do = WrapCOMObject(do, iid=pythoncom.IID_IDataObject)
        pythoncom.OleSetClipboard(do)
        self.assertEqual(num_do_objects, 1)
        do = None
        pythoncom.OleFlushClipboard()
        self.assertEqual(num_do_objects, 0)

    def testDataObjectReset(self):
        if False:
            while True:
                i = 10
        do = TestDataObject(b'Hello from Python')
        do = WrapCOMObject(do)
        pythoncom.OleSetClipboard(do)
        do = None
        self.assertEqual(num_do_objects, 1)
        pythoncom.OleSetClipboard(None)
        self.assertEqual(num_do_objects, 0)
if __name__ == '__main__':
    from win32com.test import util
    util.testmain()