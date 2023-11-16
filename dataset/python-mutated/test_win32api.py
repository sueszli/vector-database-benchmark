import datetime
import os
import sys
import tempfile
import unittest
import win32api
import win32con
import win32event
import winerror
from pywin32_testutil import TestSkipped

class CurrentUserTestCase(unittest.TestCase):

    def testGetCurrentUser(self):
        if False:
            return 10
        domain = win32api.GetDomainName()
        if domain == 'NT AUTHORITY':
            raise TestSkipped('running as service account')
        name = f'{domain}\\{win32api.GetUserName()}'
        self.assertEqual(name, win32api.GetUserNameEx(win32api.NameSamCompatible))

class TestTime(unittest.TestCase):

    def testTimezone(self):
        if False:
            print('Hello World!')
        (rc, tzinfo) = win32api.GetTimeZoneInformation()
        if rc == win32con.TIME_ZONE_ID_DAYLIGHT:
            tz_str = tzinfo[4]
            tz_time = tzinfo[5]
        else:
            tz_str = tzinfo[1]
            tz_time = tzinfo[2]
        tz_str.encode()
        if not isinstance(tz_time, datetime.datetime) and (not isinstance(tz_time, tuple)):
            tz_time.Format()

    def TestDateFormat(self):
        if False:
            print('Hello World!')
        DATE_LONGDATE = 2
        date_flags = DATE_LONGDATE
        win32api.GetDateFormat(0, date_flags, None)
        win32api.GetDateFormat(0, date_flags, 0)
        win32api.GetDateFormat(0, date_flags, datetime.datetime.now())
        win32api.GetDateFormat(0, date_flags, time.time())

    def TestTimeFormat(self):
        if False:
            return 10
        win32api.GetTimeFormat(0, 0, None)
        win32api.GetTimeFormat(0, 0, 0)
        win32api.GetTimeFormat(0, 0, datetime.datetime.now())
        win32api.GetTimeFormat(0, 0, time.time())

class Registry(unittest.TestCase):
    key_name = 'PythonTestHarness\\Whatever'

    def test1(self):
        if False:
            print('Hello World!')

        def reg_operation():
            if False:
                i = 10
                return i + 15
            hkey = win32api.RegCreateKey(win32con.HKEY_CURRENT_USER, self.key_name)
            x = 3 / 0
        try:
            try:
                try:
                    reg_operation()
                except:
                    1 / 0
            finally:
                win32api.RegDeleteKey(win32con.HKEY_CURRENT_USER, self.key_name)
        except ZeroDivisionError:
            pass

    def testValues(self):
        if False:
            i = 10
            return i + 15
        key_name = 'PythonTestHarness\\win32api'
        values = ((None, win32con.REG_SZ, 'This is default unnamed value'), ('REG_SZ', win32con.REG_SZ, 'REG_SZ text data'), ('REG_EXPAND_SZ', win32con.REG_EXPAND_SZ, '%systemdir%'), ('REG_MULTI_SZ', win32con.REG_MULTI_SZ, ['string 1', 'string 2', 'string 3', 'string 4']), ('REG_MULTI_SZ_empty', win32con.REG_MULTI_SZ, []), ('REG_DWORD', win32con.REG_DWORD, 666), ('REG_QWORD_INT', win32con.REG_QWORD, 99), ('REG_QWORD', win32con.REG_QWORD, 2 ** 33), ('REG_BINARY', win32con.REG_BINARY, b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x01\x00'))
        hkey = win32api.RegCreateKey(win32con.HKEY_CURRENT_USER, key_name)
        for (value_name, reg_type, data) in values:
            win32api.RegSetValueEx(hkey, value_name, None, reg_type, data)
        for (value_name, orig_type, orig_data) in values:
            (data, typ) = win32api.RegQueryValueEx(hkey, value_name)
            self.assertEqual(typ, orig_type)
            self.assertEqual(data, orig_data)

    def testNotifyChange(self):
        if False:
            print('Hello World!')

        def change():
            if False:
                for i in range(10):
                    print('nop')
            hkey = win32api.RegCreateKey(win32con.HKEY_CURRENT_USER, self.key_name)
            try:
                win32api.RegSetValue(hkey, None, win32con.REG_SZ, 'foo')
            finally:
                win32api.RegDeleteKey(win32con.HKEY_CURRENT_USER, self.key_name)
        evt = win32event.CreateEvent(None, 0, 0, None)
        win32api.RegNotifyChangeKeyValue(win32con.HKEY_CURRENT_USER, 1, win32api.REG_NOTIFY_CHANGE_LAST_SET, evt, True)
        ret_code = win32event.WaitForSingleObject(evt, 0)
        self.assertTrue(ret_code == win32con.WAIT_TIMEOUT)
        change()
        ret_code = win32event.WaitForSingleObject(evt, 0)
        self.assertTrue(ret_code == win32con.WAIT_OBJECT_0)

class FileNames(unittest.TestCase):

    def testShortLongPathNames(self):
        if False:
            print('Hello World!')
        try:
            me = __file__
        except NameError:
            me = sys.argv[0]
        fname = os.path.abspath(me).lower()
        short_name = win32api.GetShortPathName(fname).lower()
        long_name = win32api.GetLongPathName(short_name).lower()
        self.assertTrue(long_name == fname, f"Expected long name ('{long_name}') to be original name ('{fname}')")
        self.assertEqual(long_name, win32api.GetLongPathNameW(short_name).lower())
        long_name = win32api.GetLongPathNameW(short_name).lower()
        self.assertTrue(isinstance(long_name, str), f"GetLongPathNameW returned type '{type(long_name)}'")
        self.assertTrue(long_name == fname, f"Expected long name ('{long_name}') to be original name ('{fname}')")

    def testShortUnicodeNames(self):
        if False:
            while True:
                i = 10
        try:
            me = __file__
        except NameError:
            me = sys.argv[0]
        fname = os.path.abspath(me).lower()
        short_name = win32api.GetShortPathName(str(fname)).lower()
        self.assertTrue(isinstance(short_name, str))
        long_name = win32api.GetLongPathName(short_name).lower()
        self.assertTrue(long_name == fname, f"Expected long name ('{long_name}') to be original name ('{fname}')")
        self.assertEqual(long_name, win32api.GetLongPathNameW(short_name).lower())
        long_name = win32api.GetLongPathNameW(short_name).lower()
        self.assertTrue(isinstance(long_name, str), f"GetLongPathNameW returned type '{type(long_name)}'")
        self.assertTrue(long_name == fname, f"Expected long name ('{long_name}') to be original name ('{fname}')")

    def testLongLongPathNames(self):
        if False:
            return 10
        import win32file
        basename = 'a' * 250
        long_temp_dir = win32api.GetLongPathNameW(tempfile.gettempdir())
        fname = '\\\\?\\' + os.path.join(long_temp_dir, basename)
        try:
            win32file.CreateDirectoryW(fname, None)
        except win32api.error as details:
            if details.winerror != winerror.ERROR_ALREADY_EXISTS:
                raise
        try:
            try:
                attr = win32api.GetFileAttributes(fname)
            except win32api.error as details:
                if details.winerror != winerror.ERROR_FILENAME_EXCED_RANGE:
                    raise
            attr = win32api.GetFileAttributes(str(fname))
            self.assertTrue(attr & win32con.FILE_ATTRIBUTE_DIRECTORY, attr)
            long_name = win32api.GetLongPathNameW(fname)
            self.assertEqual(long_name.lower(), fname.lower())
        finally:
            win32file.RemoveDirectory(fname)

class FormatMessage(unittest.TestCase):

    def test_FromString(self):
        if False:
            i = 10
            return i + 15
        msg = 'Hello %1, how are you %2?'
        inserts = ['Mark', 'today']
        result = win32api.FormatMessage(win32con.FORMAT_MESSAGE_FROM_STRING, msg, 0, 0, inserts)
        self.assertEqual(result, 'Hello Mark, how are you today?')

class Misc(unittest.TestCase):

    def test_last_error(self):
        if False:
            for i in range(10):
                print('nop')
        for x in (0, 1, -1, winerror.TRUST_E_PROVIDER_UNKNOWN):
            win32api.SetLastError(x)
            self.assertEqual(x, win32api.GetLastError())

    def testVkKeyScan(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(win32api.VkKeyScan(' '), 32)

    def testVkKeyScanEx(self):
        if False:
            while True:
                i = 10
        self.assertEqual(win32api.VkKeyScanEx(' ', 0), 32)

    def testGetSystemPowerStatus(self):
        if False:
            for i in range(10):
                print('nop')
        sps = win32api.GetSystemPowerStatus()
        self.assertIsInstance(sps, dict)
        test_keys = ('ACLineStatus', 'BatteryFlag', 'BatteryLifePercent', 'SystemStatusFlag', 'BatteryLifeTime', 'BatteryFullLifeTime')
        self.assertEqual(set(test_keys), set(sps.keys()))
if __name__ == '__main__':
    unittest.main()