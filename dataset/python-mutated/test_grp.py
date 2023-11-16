"""Test script for the grp module."""
import unittest
from test.support import import_helper
grp = import_helper.import_module('grp')

class GroupDatabaseTestCase(unittest.TestCase):

    def check_value(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(value), 4)
        self.assertEqual(value[0], value.gr_name)
        self.assertIsInstance(value.gr_name, str)
        self.assertEqual(value[1], value.gr_passwd)
        self.assertIsInstance(value.gr_passwd, str)
        self.assertEqual(value[2], value.gr_gid)
        self.assertIsInstance(value.gr_gid, int)
        self.assertEqual(value[3], value.gr_mem)
        self.assertIsInstance(value.gr_mem, list)

    def test_values(self):
        if False:
            return 10
        entries = grp.getgrall()
        for e in entries:
            self.check_value(e)

    def test_values_extended(self):
        if False:
            return 10
        entries = grp.getgrall()
        if len(entries) > 1000:
            self.skipTest('huge group file, extended test skipped')
        for e in entries:
            e2 = grp.getgrgid(e.gr_gid)
            self.check_value(e2)
            self.assertEqual(e2.gr_gid, e.gr_gid)
            name = e.gr_name
            if name.startswith('+') or name.startswith('-'):
                continue
            e2 = grp.getgrnam(name)
            self.check_value(e2)
            self.assertEqual(e2.gr_name.lower(), name.lower())

    def test_errors(self):
        if False:
            return 10
        self.assertRaises(TypeError, grp.getgrgid)
        self.assertRaises(TypeError, grp.getgrnam)
        self.assertRaises(TypeError, grp.getgrall, 42)
        self.assertRaises(ValueError, grp.getgrnam, 'a\x00b')
        bynames = {}
        bygids = {}
        for (n, p, g, mem) in grp.getgrall():
            if not n or n == '+':
                continue
            bynames[n] = g
            bygids[g] = n
        allnames = list(bynames.keys())
        namei = 0
        fakename = allnames[namei]
        while fakename in bynames:
            chars = list(fakename)
            for i in range(len(chars)):
                if chars[i] == 'z':
                    chars[i] = 'A'
                    break
                elif chars[i] == 'Z':
                    continue
                else:
                    chars[i] = chr(ord(chars[i]) + 1)
                    break
            else:
                namei = namei + 1
                try:
                    fakename = allnames[namei]
                except IndexError:
                    break
            fakename = ''.join(chars)
        self.assertRaises(KeyError, grp.getgrnam, fakename)

    def test_noninteger_gid(self):
        if False:
            return 10
        entries = grp.getgrall()
        if not entries:
            self.skipTest('no groups')
        gid = entries[0][2]
        self.assertRaises(TypeError, grp.getgrgid, float(gid))
        self.assertRaises(TypeError, grp.getgrgid, str(gid))
if __name__ == '__main__':
    unittest.main()