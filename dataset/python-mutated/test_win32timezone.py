import doctest
import unittest
import win32timezone

class Win32TimeZoneTest(unittest.TestCase):

    def testWin32TZ(self):
        if False:
            return 10
        (failed, total) = doctest.testmod(win32timezone, verbose=False)
        self.assertFalse(failed)
if __name__ == '__main__':
    unittest.main()