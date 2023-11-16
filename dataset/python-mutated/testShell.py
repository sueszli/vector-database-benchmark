import datetime
import os
import struct
import sys
import pythoncom
import pywintypes
import win32com.test.util
import win32con
import win32timezone
from win32com.shell import shell
from win32com.shell.shellcon import *
from win32com.storagecon import *

class ShellTester(win32com.test.util.TestCase):

    def testShellLink(self):
        if False:
            for i in range(10):
                print('nop')
        desktop = str(shell.SHGetSpecialFolderPath(0, CSIDL_DESKTOP))
        num = 0
        shellLink = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink)
        persistFile = shellLink.QueryInterface(pythoncom.IID_IPersistFile)
        names = [os.path.join(desktop, n) for n in os.listdir(desktop)]
        programs = str(shell.SHGetSpecialFolderPath(0, CSIDL_PROGRAMS))
        names.extend([os.path.join(programs, n) for n in os.listdir(programs)])
        for name in names:
            try:
                persistFile.Load(name, STGM_READ)
            except pythoncom.com_error:
                continue
            (fname, findData) = shellLink.GetPath(0)
            unc = shellLink.GetPath(shell.SLGP_UNCPRIORITY)[0]
            num += 1
        if num == 0:
            print('Could not find any links on your desktop or programs dir, which is unusual')

    def testShellFolder(self):
        if False:
            return 10
        sf = shell.SHGetDesktopFolder()
        names_1 = []
        for i in sf:
            name = sf.GetDisplayNameOf(i, SHGDN_NORMAL)
            names_1.append(name)
        enum = sf.EnumObjects(0, SHCONTF_FOLDERS | SHCONTF_NONFOLDERS | SHCONTF_INCLUDEHIDDEN)
        names_2 = []
        for i in enum:
            name = sf.GetDisplayNameOf(i, SHGDN_NORMAL)
            names_2.append(name)
        names_1.sort()
        names_2.sort()
        self.assertEqual(names_1, names_2)

class PIDLTester(win32com.test.util.TestCase):

    def _rtPIDL(self, pidl):
        if False:
            return 10
        pidl_str = shell.PIDLAsString(pidl)
        pidl_rt = shell.StringAsPIDL(pidl_str)
        self.assertEqual(pidl_rt, pidl)
        pidl_str_rt = shell.PIDLAsString(pidl_rt)
        self.assertEqual(pidl_str_rt, pidl_str)

    def _rtCIDA(self, parent, kids):
        if False:
            print('Hello World!')
        cida = (parent, kids)
        cida_str = shell.CIDAAsString(cida)
        cida_rt = shell.StringAsCIDA(cida_str)
        self.assertEqual(cida, cida_rt)
        cida_str_rt = shell.CIDAAsString(cida_rt)
        self.assertEqual(cida_str_rt, cida_str)

    def testPIDL(self):
        if False:
            for i in range(10):
                print('nop')
        expect = b'\x03\x00' + b'\x01' + b'\x00\x00'
        self.assertEqual(shell.PIDLAsString([b'\x01']), expect)
        self._rtPIDL([b'\x00'])
        self._rtPIDL([b'\x01', b'\x02', b'\x03'])
        self._rtPIDL([b'\x00' * 2048] * 2048)
        self.assertRaises(TypeError, shell.PIDLAsString, 'foo')

    def testCIDA(self):
        if False:
            return 10
        self._rtCIDA([b'\x00'], [[b'\x00']])
        self._rtCIDA([b'\x01'], [[b'\x02']])
        self._rtCIDA([b'\x00'], [[b'\x00'], [b'\x01'], [b'\x02']])

    def testBadShortPIDL(self):
        if False:
            i = 10
            return i + 15
        pidl = b'\x01\x00' + b'\x01'
        self.assertRaises(ValueError, shell.StringAsPIDL, pidl)

class FILEGROUPDESCRIPTORTester(win32com.test.util.TestCase):

    def _getTestTimes(self):
        if False:
            for i in range(10):
                print('nop')
        if issubclass(pywintypes.TimeType, datetime.datetime):
            ctime = win32timezone.now()
            ctime = ctime.replace(microsecond=ctime.microsecond // 1000 * 1000)
            atime = ctime + datetime.timedelta(seconds=1)
            wtime = atime + datetime.timedelta(seconds=1)
        else:
            ctime = pywintypes.Time(11)
            atime = pywintypes.Time(12)
            wtime = pywintypes.Time(13)
        return (ctime, atime, wtime)

    def _testRT(self, fd):
        if False:
            i = 10
            return i + 15
        fgd_string = shell.FILEGROUPDESCRIPTORAsString([fd])
        fd2 = shell.StringAsFILEGROUPDESCRIPTOR(fgd_string)[0]
        fd = fd.copy()
        fd2 = fd2.copy()
        if 'dwFlags' not in fd:
            del fd2['dwFlags']
        if 'cFileName' not in fd:
            self.assertEqual(fd2['cFileName'], '')
            del fd2['cFileName']
        self.assertEqual(fd, fd2)

    def _testSimple(self, make_unicode):
        if False:
            return 10
        fgd = shell.FILEGROUPDESCRIPTORAsString([], make_unicode)
        header = struct.pack('i', 0)
        self.assertEqual(header, fgd[:len(header)])
        self._testRT({})
        d = {}
        fgd = shell.FILEGROUPDESCRIPTORAsString([d], make_unicode)
        header = struct.pack('i', 1)
        self.assertEqual(header, fgd[:len(header)])
        self._testRT(d)

    def testSimpleBytes(self):
        if False:
            for i in range(10):
                print('nop')
        self._testSimple(False)

    def testSimpleUnicode(self):
        if False:
            for i in range(10):
                print('nop')
        self._testSimple(True)

    def testComplex(self):
        if False:
            i = 10
            return i + 15
        clsid = pythoncom.MakeIID('{CD637886-DB8B-4b04-98B5-25731E1495BE}')
        (ctime, atime, wtime) = self._getTestTimes()
        d = {'cFileName': 'foo.txt', 'clsid': clsid, 'sizel': (1, 2), 'pointl': (3, 4), 'dwFileAttributes': win32con.FILE_ATTRIBUTE_NORMAL, 'ftCreationTime': ctime, 'ftLastAccessTime': atime, 'ftLastWriteTime': wtime, 'nFileSize': sys.maxsize + 1}
        self._testRT(d)

    def testUnicode(self):
        if False:
            for i in range(10):
                print('nop')
        (ctime, atime, wtime) = self._getTestTimes()
        d = [{'cFileName': 'foo.txt', 'sizel': (1, 2), 'pointl': (3, 4), 'dwFileAttributes': win32con.FILE_ATTRIBUTE_NORMAL, 'ftCreationTime': ctime, 'ftLastAccessTime': atime, 'ftLastWriteTime': wtime, 'nFileSize': sys.maxsize + 1}, {'cFileName': 'foo2.txt', 'sizel': (1, 2), 'pointl': (3, 4), 'dwFileAttributes': win32con.FILE_ATTRIBUTE_NORMAL, 'ftCreationTime': ctime, 'ftLastAccessTime': atime, 'ftLastWriteTime': wtime, 'nFileSize': sys.maxsize + 1}, {'cFileName': 'foo©.txt', 'sizel': (1, 2), 'pointl': (3, 4), 'dwFileAttributes': win32con.FILE_ATTRIBUTE_NORMAL, 'ftCreationTime': ctime, 'ftLastAccessTime': atime, 'ftLastWriteTime': wtime, 'nFileSize': sys.maxsize + 1}]
        s = shell.FILEGROUPDESCRIPTORAsString(d, 1)
        d2 = shell.StringAsFILEGROUPDESCRIPTOR(s)
        for t in d2:
            del t['dwFlags']
        self.assertEqual(d, d2)

class FileOperationTester(win32com.test.util.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        import tempfile
        self.src_name = os.path.join(tempfile.gettempdir(), 'pywin32_testshell')
        self.dest_name = os.path.join(tempfile.gettempdir(), 'pywin32_testshell_dest')
        self.test_data = b'Hello from\x00Python'
        f = open(self.src_name, 'wb')
        f.write(self.test_data)
        f.close()
        try:
            os.unlink(self.dest_name)
        except OSError:
            pass

    def tearDown(self):
        if False:
            print('Hello World!')
        for fname in (self.src_name, self.dest_name):
            if os.path.isfile(fname):
                os.unlink(fname)

    def testCopy(self):
        if False:
            return 10
        s = (0, FO_COPY, self.src_name, self.dest_name)
        (rc, aborted) = shell.SHFileOperation(s)
        self.assertTrue(not aborted)
        self.assertEqual(0, rc)
        self.assertTrue(os.path.isfile(self.src_name))
        self.assertTrue(os.path.isfile(self.dest_name))

    def testRename(self):
        if False:
            print('Hello World!')
        s = (0, FO_RENAME, self.src_name, self.dest_name)
        (rc, aborted) = shell.SHFileOperation(s)
        self.assertTrue(not aborted)
        self.assertEqual(0, rc)
        self.assertTrue(os.path.isfile(self.dest_name))
        self.assertTrue(not os.path.isfile(self.src_name))

    def testMove(self):
        if False:
            while True:
                i = 10
        s = (0, FO_MOVE, self.src_name, self.dest_name)
        (rc, aborted) = shell.SHFileOperation(s)
        self.assertTrue(not aborted)
        self.assertEqual(0, rc)
        self.assertTrue(os.path.isfile(self.dest_name))
        self.assertTrue(not os.path.isfile(self.src_name))

    def testDelete(self):
        if False:
            i = 10
            return i + 15
        s = (0, FO_DELETE, self.src_name, None, FOF_NOCONFIRMATION)
        (rc, aborted) = shell.SHFileOperation(s)
        self.assertTrue(not aborted)
        self.assertEqual(0, rc)
        self.assertTrue(not os.path.isfile(self.src_name))
if __name__ == '__main__':
    win32com.test.util.testmain()