import os
import pythoncom
import pywintypes
import win32api
import win32com
import win32com.client
import win32com.client.dynamic
import win32com.server.util
import win32timezone
import win32ui
from win32com import storagecon
from win32com.axcontrol import axcontrol
from win32com.test.util import CheckClean
S_OK = 0
now = win32timezone.now()

class LockBytes:
    _public_methods_ = ['ReadAt', 'WriteAt', 'Flush', 'SetSize', 'LockRegion', 'UnlockRegion', 'Stat']
    _com_interfaces_ = [pythoncom.IID_ILockBytes]

    def __init__(self, data=b''):
        if False:
            for i in range(10):
                print('nop')
        self.data = data
        self.ctime = now
        self.mtime = now
        self.atime = now

    def ReadAt(self, offset, cb):
        if False:
            i = 10
            return i + 15
        print('ReadAt')
        result = self.data[offset:offset + cb]
        return result

    def WriteAt(self, offset, data):
        if False:
            for i in range(10):
                print('nop')
        print('WriteAt ' + str(offset))
        print('len ' + str(len(data)))
        print('data:')
        if len(self.data) >= offset:
            newdata = self.data[0:offset] + data
        print(len(newdata))
        if len(self.data) >= offset + len(data):
            newdata = newdata + self.data[offset + len(data):]
        print(len(newdata))
        self.data = newdata
        return len(data)

    def Flush(self, whatsthis=0):
        if False:
            while True:
                i = 10
        print('Flush' + str(whatsthis))
        fname = os.path.join(win32api.GetTempPath(), 'persist.doc')
        open(fname, 'wb').write(self.data)
        return S_OK

    def SetSize(self, size):
        if False:
            return 10
        print('Set Size' + str(size))
        if size > len(self.data):
            self.data = self.data + b'\x00' * (size - len(self.data))
        else:
            self.data = self.data[0:size]
        return S_OK

    def LockRegion(self, offset, size, locktype):
        if False:
            i = 10
            return i + 15
        print('LockRegion')

    def UnlockRegion(self, offset, size, locktype):
        if False:
            i = 10
            return i + 15
        print('UnlockRegion')

    def Stat(self, statflag):
        if False:
            while True:
                i = 10
        print('returning Stat ' + str(statflag))
        return ('PyMemBytes', storagecon.STGTY_LOCKBYTES, len(self.data), self.mtime, self.ctime, self.atime, storagecon.STGM_DIRECT | storagecon.STGM_READWRITE | storagecon.STGM_CREATE, storagecon.STGM_SHARE_EXCLUSIVE, '{00020905-0000-0000-C000-000000000046}', 0, 0)

class OleClientSite:
    _public_methods_ = ['SaveObject', 'GetMoniker', 'GetContainer', 'ShowObject', 'OnShowWindow', 'RequestNewObjectLayout']
    _com_interfaces_ = [axcontrol.IID_IOleClientSite]

    def __init__(self, data=''):
        if False:
            return 10
        self.IPersistStorage = None
        self.IStorage = None

    def SetIPersistStorage(self, IPersistStorage):
        if False:
            print('Hello World!')
        self.IPersistStorage = IPersistStorage

    def SetIStorage(self, IStorage):
        if False:
            return 10
        self.IStorage = IStorage

    def SaveObject(self):
        if False:
            return 10
        print('SaveObject')
        if self.IPersistStorage is not None and self.IStorage is not None:
            self.IPersistStorage.Save(self.IStorage, 1)
            self.IStorage.Commit(0)
        return S_OK

    def GetMoniker(self, dwAssign, dwWhichMoniker):
        if False:
            for i in range(10):
                print('nop')
        print('GetMoniker ' + str(dwAssign) + ' ' + str(dwWhichMoniker))

    def GetContainer(self):
        if False:
            return 10
        print('GetContainer')

    def ShowObject(self):
        if False:
            while True:
                i = 10
        print('ShowObject')

    def OnShowWindow(self, fShow):
        if False:
            print('Hello World!')
        print('ShowObject' + str(fShow))

    def RequestNewObjectLayout(self):
        if False:
            i = 10
            return i + 15
        print('RequestNewObjectLayout')

def test():
    if False:
        while True:
            i = 10
    lbcom = win32com.server.util.wrap(LockBytes(), pythoncom.IID_ILockBytes)
    stcom = pythoncom.StgCreateDocfileOnILockBytes(lbcom, storagecon.STGM_DIRECT | storagecon.STGM_CREATE | storagecon.STGM_READWRITE | storagecon.STGM_SHARE_EXCLUSIVE, 0)
    ocs = OleClientSite()
    ocscom = win32com.server.util.wrap(ocs, axcontrol.IID_IOleClientSite)
    oocom = axcontrol.OleCreate('{00020906-0000-0000-C000-000000000046}', axcontrol.IID_IOleObject, 0, (0,), ocscom, stcom)
    mf = win32ui.GetMainFrame()
    hwnd = mf.GetSafeHwnd()
    oocom.SetHostNames('OTPython', 'This is Cool')
    oocom.DoVerb(-1, ocscom, 0, hwnd, mf.GetWindowRect())
    oocom.SetHostNames('OTPython2', 'ThisisCool2')
    doc = win32com.client.Dispatch(oocom.QueryInterface(pythoncom.IID_IDispatch))
    dpcom = oocom.QueryInterface(pythoncom.IID_IPersistStorage)
    ocs.SetIPersistStorage(dpcom)
    ocs.SetIStorage(stcom)
    wrange = doc.Range()
    for i in range(10):
        wrange.InsertAfter('Hello from Python %d\n' % i)
    paras = doc.Paragraphs
    for i in range(len(paras)):
        paras[i]().Font.ColorIndex = i + 1
        paras[i]().Font.Size = 12 + 4 * i
    dpcom.Save(stcom, 0)
    dpcom.HandsOffStorage()
    lbcom.Flush()
    doc.Application.Quit()
if __name__ == '__main__':
    test()
    pythoncom.CoUninitialize()
    CheckClean()