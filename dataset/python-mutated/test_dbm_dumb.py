"""Test script for the dumbdbm module
   Original by Roger E. Masse
"""
import contextlib
import io
import operator
import os
import stat
import unittest
import dbm.dumb as dumbdbm
from test import support
from test.support import os_helper
from functools import partial
_fname = os_helper.TESTFN

def _delete_files():
    if False:
        return 10
    for ext in ['.dir', '.dat', '.bak']:
        try:
            os.unlink(_fname + ext)
        except OSError:
            pass

class DumbDBMTestCase(unittest.TestCase):
    _dict = {b'0': b'', b'a': b'Python:', b'b': b'Programming', b'c': b'the', b'd': b'way', b'f': b'Guido', b'g': b'intended', 'ü'.encode('utf-8'): b'!'}

    def test_dumbdbm_creation(self):
        if False:
            return 10
        with contextlib.closing(dumbdbm.open(_fname, 'c')) as f:
            self.assertEqual(list(f.keys()), [])
            for key in self._dict:
                f[key] = self._dict[key]
            self.read_helper(f)

    @unittest.skipUnless(hasattr(os, 'umask'), 'test needs os.umask()')
    def test_dumbdbm_creation_mode(self):
        if False:
            while True:
                i = 10
        try:
            old_umask = os.umask(2)
            f = dumbdbm.open(_fname, 'c', 415)
            f.close()
        finally:
            os.umask(old_umask)
        expected_mode = 413
        if os.name != 'posix':
            expected_mode = 438
        import stat
        st = os.stat(_fname + '.dat')
        self.assertEqual(stat.S_IMODE(st.st_mode), expected_mode)
        st = os.stat(_fname + '.dir')
        self.assertEqual(stat.S_IMODE(st.st_mode), expected_mode)

    def test_close_twice(self):
        if False:
            for i in range(10):
                print('nop')
        f = dumbdbm.open(_fname)
        f[b'a'] = b'b'
        self.assertEqual(f[b'a'], b'b')
        f.close()
        f.close()

    def test_dumbdbm_modification(self):
        if False:
            i = 10
            return i + 15
        self.init_db()
        with contextlib.closing(dumbdbm.open(_fname, 'w')) as f:
            self._dict[b'g'] = f[b'g'] = b'indented'
            self.read_helper(f)
            self.assertEqual(f.setdefault(b'xxx', b'foo'), b'foo')
            self.assertEqual(f[b'xxx'], b'foo')

    def test_dumbdbm_read(self):
        if False:
            i = 10
            return i + 15
        self.init_db()
        with contextlib.closing(dumbdbm.open(_fname, 'r')) as f:
            self.read_helper(f)
            with self.assertRaisesRegex(dumbdbm.error, 'The database is opened for reading only'):
                f[b'g'] = b'x'
            with self.assertRaisesRegex(dumbdbm.error, 'The database is opened for reading only'):
                del f[b'a']
            self.assertEqual(f.get(b'a'), self._dict[b'a'])
            self.assertEqual(f.get(b'xxx', b'foo'), b'foo')
            self.assertIsNone(f.get(b'xxx'))
            with self.assertRaises(KeyError):
                f[b'xxx']

    def test_dumbdbm_keys(self):
        if False:
            print('Hello World!')
        self.init_db()
        with contextlib.closing(dumbdbm.open(_fname)) as f:
            keys = self.keys_helper(f)

    def test_write_contains(self):
        if False:
            i = 10
            return i + 15
        with contextlib.closing(dumbdbm.open(_fname)) as f:
            f[b'1'] = b'hello'
            self.assertIn(b'1', f)

    def test_write_write_read(self):
        if False:
            while True:
                i = 10
        with contextlib.closing(dumbdbm.open(_fname)) as f:
            f[b'1'] = b'hello'
            f[b'1'] = b'hello2'
        with contextlib.closing(dumbdbm.open(_fname)) as f:
            self.assertEqual(f[b'1'], b'hello2')

    def test_str_read(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_db()
        with contextlib.closing(dumbdbm.open(_fname, 'r')) as f:
            self.assertEqual(f['ü'], self._dict['ü'.encode('utf-8')])

    def test_str_write_contains(self):
        if False:
            i = 10
            return i + 15
        self.init_db()
        with contextlib.closing(dumbdbm.open(_fname)) as f:
            f['ü'] = b'!'
            f['1'] = 'a'
        with contextlib.closing(dumbdbm.open(_fname, 'r')) as f:
            self.assertIn('ü', f)
            self.assertEqual(f['ü'.encode('utf-8')], self._dict['ü'.encode('utf-8')])
            self.assertEqual(f[b'1'], b'a')

    def test_line_endings(self):
        if False:
            return 10
        with contextlib.closing(dumbdbm.open(_fname)) as f:
            f[b'1'] = b'hello'
            f[b'2'] = b'hello2'
        with io.open(_fname + '.dir', 'rb') as file:
            data = file.read()
        if os.linesep == '\n':
            data = data.replace(b'\n', b'\r\n')
        else:
            data = data.replace(b'\r\n', b'\n')
        with io.open(_fname + '.dir', 'wb') as file:
            file.write(data)
        f = dumbdbm.open(_fname)
        self.assertEqual(f[b'1'], b'hello')
        self.assertEqual(f[b'2'], b'hello2')

    def read_helper(self, f):
        if False:
            while True:
                i = 10
        keys = self.keys_helper(f)
        for key in self._dict:
            self.assertEqual(self._dict[key], f[key])

    def init_db(self):
        if False:
            while True:
                i = 10
        with contextlib.closing(dumbdbm.open(_fname, 'n')) as f:
            for k in self._dict:
                f[k] = self._dict[k]

    def keys_helper(self, f):
        if False:
            i = 10
            return i + 15
        keys = sorted(f.keys())
        dkeys = sorted(self._dict.keys())
        self.assertEqual(keys, dkeys)
        return keys

    def test_random(self):
        if False:
            for i in range(10):
                print('nop')
        import random
        d = {}
        for dummy in range(5):
            with contextlib.closing(dumbdbm.open(_fname)) as f:
                for dummy in range(100):
                    k = random.choice('abcdefghijklm')
                    if random.random() < 0.2:
                        if k in d:
                            del d[k]
                            del f[k]
                    else:
                        v = random.choice((b'a', b'b', b'c')) * random.randrange(10000)
                        d[k] = v
                        f[k] = v
                        self.assertEqual(f[k], v)
            with contextlib.closing(dumbdbm.open(_fname)) as f:
                expected = sorted(((k.encode('latin-1'), v) for (k, v) in d.items()))
                got = sorted(f.items())
                self.assertEqual(expected, got)

    def test_context_manager(self):
        if False:
            return 10
        with dumbdbm.open(_fname, 'c') as db:
            db['dumbdbm context manager'] = 'context manager'
        with dumbdbm.open(_fname, 'r') as db:
            self.assertEqual(list(db.keys()), [b'dumbdbm context manager'])
        with self.assertRaises(dumbdbm.error):
            db.keys()

    def test_check_closed(self):
        if False:
            print('Hello World!')
        f = dumbdbm.open(_fname, 'c')
        f.close()
        for meth in (partial(operator.delitem, f), partial(operator.setitem, f, 'b'), partial(operator.getitem, f), partial(operator.contains, f)):
            with self.assertRaises(dumbdbm.error) as cm:
                meth('test')
            self.assertEqual(str(cm.exception), 'DBM object has already been closed')
        for meth in (operator.methodcaller('keys'), operator.methodcaller('iterkeys'), operator.methodcaller('items'), len):
            with self.assertRaises(dumbdbm.error) as cm:
                meth(f)
            self.assertEqual(str(cm.exception), 'DBM object has already been closed')

    def test_create_new(self):
        if False:
            i = 10
            return i + 15
        with dumbdbm.open(_fname, 'n') as f:
            for k in self._dict:
                f[k] = self._dict[k]
        with dumbdbm.open(_fname, 'n') as f:
            self.assertEqual(f.keys(), [])

    def test_eval(self):
        if False:
            i = 10
            return i + 15
        with open(_fname + '.dir', 'w', encoding='utf-8') as stream:
            stream.write("str(print('Hacked!')), 0\n")
        with support.captured_stdout() as stdout:
            with self.assertRaises(ValueError):
                with dumbdbm.open(_fname) as f:
                    pass
            self.assertEqual(stdout.getvalue(), '')

    def test_missing_data(self):
        if False:
            i = 10
            return i + 15
        for value in ('r', 'w'):
            _delete_files()
            with self.assertRaises(FileNotFoundError):
                dumbdbm.open(_fname, value)
            self.assertFalse(os.path.exists(_fname + '.dir'))
            self.assertFalse(os.path.exists(_fname + '.bak'))

    def test_missing_index(self):
        if False:
            i = 10
            return i + 15
        with dumbdbm.open(_fname, 'n') as f:
            pass
        os.unlink(_fname + '.dir')
        for value in ('r', 'w'):
            with self.assertRaises(FileNotFoundError):
                dumbdbm.open(_fname, value)
            self.assertFalse(os.path.exists(_fname + '.dir'))
            self.assertFalse(os.path.exists(_fname + '.bak'))

    def test_invalid_flag(self):
        if False:
            i = 10
            return i + 15
        for flag in ('x', 'rf', None):
            with self.assertRaisesRegex(ValueError, "Flag must be one of 'r', 'w', 'c', or 'n'"):
                dumbdbm.open(_fname, flag)

    def test_readonly_files(self):
        if False:
            return 10
        with os_helper.temp_dir() as dir:
            fname = os.path.join(dir, 'db')
            with dumbdbm.open(fname, 'n') as f:
                self.assertEqual(list(f.keys()), [])
                for key in self._dict:
                    f[key] = self._dict[key]
            os.chmod(fname + '.dir', stat.S_IRUSR)
            os.chmod(fname + '.dat', stat.S_IRUSR)
            os.chmod(dir, stat.S_IRUSR | stat.S_IXUSR)
            with dumbdbm.open(fname, 'r') as f:
                self.assertEqual(sorted(f.keys()), sorted(self._dict))
                f.close()

    @unittest.skipUnless(os_helper.TESTFN_NONASCII, 'requires OS support of non-ASCII encodings')
    def test_nonascii_filename(self):
        if False:
            return 10
        filename = os_helper.TESTFN_NONASCII
        for suffix in ['.dir', '.dat', '.bak']:
            self.addCleanup(os_helper.unlink, filename + suffix)
        with dumbdbm.open(filename, 'c') as db:
            db[b'key'] = b'value'
        self.assertTrue(os.path.exists(filename + '.dat'))
        self.assertTrue(os.path.exists(filename + '.dir'))
        with dumbdbm.open(filename, 'r') as db:
            self.assertEqual(list(db.keys()), [b'key'])
            self.assertTrue(b'key' in db)
            self.assertEqual(db[b'key'], b'value')

    def tearDown(self):
        if False:
            while True:
                i = 10
        _delete_files()

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        _delete_files()
if __name__ == '__main__':
    unittest.main()