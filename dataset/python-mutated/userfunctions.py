import unittest
import unittest.mock
import sqlite3 as sqlite
from test.support import gc_collect

def func_returntext():
    if False:
        return 10
    return 'foo'

def func_returntextwithnull():
    if False:
        print('Hello World!')
    return '1\x002'

def func_returnunicode():
    if False:
        print('Hello World!')
    return 'bar'

def func_returnint():
    if False:
        while True:
            i = 10
    return 42

def func_returnfloat():
    if False:
        print('Hello World!')
    return 3.14

def func_returnnull():
    if False:
        i = 10
        return i + 15
    return None

def func_returnblob():
    if False:
        i = 10
        return i + 15
    return b'blob'

def func_returnlonglong():
    if False:
        i = 10
        return i + 15
    return 1 << 31

def func_raiseexception():
    if False:
        i = 10
        return i + 15
    5 / 0

class AggrNoStep:

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def finalize(self):
        if False:
            i = 10
            return i + 15
        return 1

class AggrNoFinalize:

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def step(self, x):
        if False:
            for i in range(10):
                print('nop')
        pass

class AggrExceptionInInit:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        5 / 0

    def step(self, x):
        if False:
            while True:
                i = 10
        pass

    def finalize(self):
        if False:
            i = 10
            return i + 15
        pass

class AggrExceptionInStep:

    def __init__(self):
        if False:
            return 10
        pass

    def step(self, x):
        if False:
            i = 10
            return i + 15
        5 / 0

    def finalize(self):
        if False:
            i = 10
            return i + 15
        return 42

class AggrExceptionInFinalize:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def step(self, x):
        if False:
            print('Hello World!')
        pass

    def finalize(self):
        if False:
            i = 10
            return i + 15
        5 / 0

class AggrCheckType:

    def __init__(self):
        if False:
            print('Hello World!')
        self.val = None

    def step(self, whichType, val):
        if False:
            print('Hello World!')
        theType = {'str': str, 'int': int, 'float': float, 'None': type(None), 'blob': bytes}
        self.val = int(theType[whichType] is type(val))

    def finalize(self):
        if False:
            i = 10
            return i + 15
        return self.val

class AggrCheckTypes:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.val = 0

    def step(self, whichType, *vals):
        if False:
            return 10
        theType = {'str': str, 'int': int, 'float': float, 'None': type(None), 'blob': bytes}
        for val in vals:
            self.val += int(theType[whichType] is type(val))

    def finalize(self):
        if False:
            print('Hello World!')
        return self.val

class AggrSum:

    def __init__(self):
        if False:
            print('Hello World!')
        self.val = 0.0

    def step(self, val):
        if False:
            while True:
                i = 10
        self.val += val

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        return self.val

class AggrText:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.txt = ''

    def step(self, txt):
        if False:
            print('Hello World!')
        self.txt = self.txt + txt

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        return self.txt

class FunctionTests(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.con = sqlite.connect(':memory:')
        self.con.create_function('returntext', 0, func_returntext)
        self.con.create_function('returntextwithnull', 0, func_returntextwithnull)
        self.con.create_function('returnunicode', 0, func_returnunicode)
        self.con.create_function('returnint', 0, func_returnint)
        self.con.create_function('returnfloat', 0, func_returnfloat)
        self.con.create_function('returnnull', 0, func_returnnull)
        self.con.create_function('returnblob', 0, func_returnblob)
        self.con.create_function('returnlonglong', 0, func_returnlonglong)
        self.con.create_function('returnnan', 0, lambda : float('nan'))
        self.con.create_function('returntoolargeint', 0, lambda : 1 << 65)
        self.con.create_function('raiseexception', 0, func_raiseexception)
        self.con.create_function('isblob', 1, lambda x: isinstance(x, bytes))
        self.con.create_function('isnone', 1, lambda x: x is None)
        self.con.create_function('spam', -1, lambda *x: len(x))
        self.con.execute('create table test(t text)')

    def tearDown(self):
        if False:
            print('Hello World!')
        self.con.close()

    def test_func_error_on_create(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(sqlite.OperationalError):
            self.con.create_function('bla', -100, lambda x: 2 * x)

    def test_func_ref_count(self):
        if False:
            print('Hello World!')

        def getfunc():
            if False:
                print('Hello World!')

            def f():
                if False:
                    return 10
                return 1
            return f
        f = getfunc()
        globals()['foo'] = f
        self.con.create_function('reftest', 0, f)
        cur = self.con.cursor()
        cur.execute('select reftest()')

    def test_func_return_text(self):
        if False:
            for i in range(10):
                print('nop')
        cur = self.con.cursor()
        cur.execute('select returntext()')
        val = cur.fetchone()[0]
        self.assertEqual(type(val), str)
        self.assertEqual(val, 'foo')

    def test_func_return_text_with_null_char(self):
        if False:
            for i in range(10):
                print('nop')
        cur = self.con.cursor()
        res = cur.execute('select returntextwithnull()').fetchone()[0]
        self.assertEqual(type(res), str)
        self.assertEqual(res, '1\x002')

    def test_func_return_unicode(self):
        if False:
            while True:
                i = 10
        cur = self.con.cursor()
        cur.execute('select returnunicode()')
        val = cur.fetchone()[0]
        self.assertEqual(type(val), str)
        self.assertEqual(val, 'bar')

    def test_func_return_int(self):
        if False:
            return 10
        cur = self.con.cursor()
        cur.execute('select returnint()')
        val = cur.fetchone()[0]
        self.assertEqual(type(val), int)
        self.assertEqual(val, 42)

    def test_func_return_float(self):
        if False:
            while True:
                i = 10
        cur = self.con.cursor()
        cur.execute('select returnfloat()')
        val = cur.fetchone()[0]
        self.assertEqual(type(val), float)
        if val < 3.139 or val > 3.141:
            self.fail('wrong value')

    def test_func_return_null(self):
        if False:
            return 10
        cur = self.con.cursor()
        cur.execute('select returnnull()')
        val = cur.fetchone()[0]
        self.assertEqual(type(val), type(None))
        self.assertEqual(val, None)

    def test_func_return_blob(self):
        if False:
            print('Hello World!')
        cur = self.con.cursor()
        cur.execute('select returnblob()')
        val = cur.fetchone()[0]
        self.assertEqual(type(val), bytes)
        self.assertEqual(val, b'blob')

    def test_func_return_long_long(self):
        if False:
            i = 10
            return i + 15
        cur = self.con.cursor()
        cur.execute('select returnlonglong()')
        val = cur.fetchone()[0]
        self.assertEqual(val, 1 << 31)

    def test_func_return_nan(self):
        if False:
            while True:
                i = 10
        cur = self.con.cursor()
        cur.execute('select returnnan()')
        self.assertIsNone(cur.fetchone()[0])

    def test_func_return_too_large_int(self):
        if False:
            while True:
                i = 10
        cur = self.con.cursor()
        with self.assertRaises(sqlite.OperationalError):
            self.con.execute('select returntoolargeint()')

    def test_func_exception(self):
        if False:
            print('Hello World!')
        cur = self.con.cursor()
        with self.assertRaises(sqlite.OperationalError) as cm:
            cur.execute('select raiseexception()')
            cur.fetchone()
        self.assertEqual(str(cm.exception), 'user-defined function raised exception')

    def test_any_arguments(self):
        if False:
            print('Hello World!')
        cur = self.con.cursor()
        cur.execute('select spam(?, ?)', (1, 2))
        val = cur.fetchone()[0]
        self.assertEqual(val, 2)

    def test_empty_blob(self):
        if False:
            print('Hello World!')
        cur = self.con.execute("select isblob(x'')")
        self.assertTrue(cur.fetchone()[0])

    def test_nan_float(self):
        if False:
            i = 10
            return i + 15
        cur = self.con.execute('select isnone(?)', (float('nan'),))
        self.assertTrue(cur.fetchone()[0])

    def test_too_large_int(self):
        if False:
            print('Hello World!')
        err = 'Python int too large to convert to SQLite INTEGER'
        self.assertRaisesRegex(OverflowError, err, self.con.execute, 'select spam(?)', (1 << 65,))

    def test_non_contiguous_blob(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaisesRegex(ValueError, 'could not convert BLOB to buffer', self.con.execute, 'select spam(?)', (memoryview(b'blob')[::2],))

    def test_param_surrogates(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaisesRegex(UnicodeEncodeError, 'surrogates not allowed', self.con.execute, 'select spam(?)', ('\ud803\ude6d',))

    def test_func_params(self):
        if False:
            print('Hello World!')
        results = []

        def append_result(arg):
            if False:
                return 10
            results.append((arg, type(arg)))
        self.con.create_function('test_params', 1, append_result)
        dataset = [(42, int), (-1, int), (1234567890123456789, int), (4611686018427387905, int), (3.14, float), (float('inf'), float), ('text', str), ('1\x002', str), ('ˢqˡⁱᵗᵉ', str), (b'blob', bytes), (bytearray(range(2)), bytes), (memoryview(b'blob'), bytes), (None, type(None))]
        for (val, _) in dataset:
            cur = self.con.execute('select test_params(?)', (val,))
            cur.fetchone()
        self.assertEqual(dataset, results)

    @unittest.skipIf(sqlite.sqlite_version_info < (3, 8, 3), 'Requires SQLite 3.8.3 or higher')
    def test_func_non_deterministic(self):
        if False:
            for i in range(10):
                print('nop')
        mock = unittest.mock.Mock(return_value=None)
        self.con.create_function('nondeterministic', 0, mock, deterministic=False)
        if sqlite.sqlite_version_info < (3, 15, 0):
            self.con.execute('select nondeterministic() = nondeterministic()')
            self.assertEqual(mock.call_count, 2)
        else:
            with self.assertRaises(sqlite.OperationalError):
                self.con.execute('create index t on test(t) where nondeterministic() is not null')

    @unittest.skipIf(sqlite.sqlite_version_info < (3, 8, 3), 'Requires SQLite 3.8.3 or higher')
    def test_func_deterministic(self):
        if False:
            print('Hello World!')
        mock = unittest.mock.Mock(return_value=None)
        self.con.create_function('deterministic', 0, mock, deterministic=True)
        if sqlite.sqlite_version_info < (3, 15, 0):
            self.con.execute('select deterministic() = deterministic()')
            self.assertEqual(mock.call_count, 1)
        else:
            try:
                self.con.execute('create index t on test(t) where deterministic() is not null')
            except sqlite.OperationalError:
                self.fail('Unexpected failure while creating partial index')

    @unittest.skipIf(sqlite.sqlite_version_info >= (3, 8, 3), 'SQLite < 3.8.3 needed')
    def test_func_deterministic_not_supported(self):
        if False:
            return 10
        with self.assertRaises(sqlite.NotSupportedError):
            self.con.create_function('deterministic', 0, int, deterministic=True)

    def test_func_deterministic_keyword_only(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            self.con.create_function('deterministic', 0, int, True)

    def test_function_destructor_via_gc(self):
        if False:
            while True:
                i = 10
        dest = sqlite.connect(':memory:')

        def md5sum(t):
            if False:
                return 10
            return
        dest.create_function('md5', 1, md5sum)
        x = dest('create table lang (name, first_appeared)')
        del md5sum, dest
        y = [x]
        y.append(y)
        del x, y
        gc_collect()

class AggregateTests(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.con = sqlite.connect(':memory:')
        cur = self.con.cursor()
        cur.execute('\n            create table test(\n                t text,\n                i integer,\n                f float,\n                n,\n                b blob\n                )\n            ')
        cur.execute('insert into test(t, i, f, n, b) values (?, ?, ?, ?, ?)', ('foo', 5, 3.14, None, memoryview(b'blob')))
        self.con.create_aggregate('nostep', 1, AggrNoStep)
        self.con.create_aggregate('nofinalize', 1, AggrNoFinalize)
        self.con.create_aggregate('excInit', 1, AggrExceptionInInit)
        self.con.create_aggregate('excStep', 1, AggrExceptionInStep)
        self.con.create_aggregate('excFinalize', 1, AggrExceptionInFinalize)
        self.con.create_aggregate('checkType', 2, AggrCheckType)
        self.con.create_aggregate('checkTypes', -1, AggrCheckTypes)
        self.con.create_aggregate('mysum', 1, AggrSum)
        self.con.create_aggregate('aggtxt', 1, AggrText)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_aggr_error_on_create(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(sqlite.OperationalError):
            self.con.create_function('bla', -100, AggrSum)

    def test_aggr_no_step(self):
        if False:
            i = 10
            return i + 15
        cur = self.con.cursor()
        with self.assertRaises(AttributeError) as cm:
            cur.execute('select nostep(t) from test')
        self.assertEqual(str(cm.exception), "'AggrNoStep' object has no attribute 'step'")

    def test_aggr_no_finalize(self):
        if False:
            return 10
        cur = self.con.cursor()
        with self.assertRaises(sqlite.OperationalError) as cm:
            cur.execute('select nofinalize(t) from test')
            val = cur.fetchone()[0]
        self.assertEqual(str(cm.exception), "user-defined aggregate's 'finalize' method raised error")

    def test_aggr_exception_in_init(self):
        if False:
            while True:
                i = 10
        cur = self.con.cursor()
        with self.assertRaises(sqlite.OperationalError) as cm:
            cur.execute('select excInit(t) from test')
            val = cur.fetchone()[0]
        self.assertEqual(str(cm.exception), "user-defined aggregate's '__init__' method raised error")

    def test_aggr_exception_in_step(self):
        if False:
            print('Hello World!')
        cur = self.con.cursor()
        with self.assertRaises(sqlite.OperationalError) as cm:
            cur.execute('select excStep(t) from test')
            val = cur.fetchone()[0]
        self.assertEqual(str(cm.exception), "user-defined aggregate's 'step' method raised error")

    def test_aggr_exception_in_finalize(self):
        if False:
            while True:
                i = 10
        cur = self.con.cursor()
        with self.assertRaises(sqlite.OperationalError) as cm:
            cur.execute('select excFinalize(t) from test')
            val = cur.fetchone()[0]
        self.assertEqual(str(cm.exception), "user-defined aggregate's 'finalize' method raised error")

    def test_aggr_check_param_str(self):
        if False:
            return 10
        cur = self.con.cursor()
        cur.execute("select checkTypes('str', ?, ?)", ('foo', str()))
        val = cur.fetchone()[0]
        self.assertEqual(val, 2)

    def test_aggr_check_param_int(self):
        if False:
            while True:
                i = 10
        cur = self.con.cursor()
        cur.execute("select checkType('int', ?)", (42,))
        val = cur.fetchone()[0]
        self.assertEqual(val, 1)

    def test_aggr_check_params_int(self):
        if False:
            while True:
                i = 10
        cur = self.con.cursor()
        cur.execute("select checkTypes('int', ?, ?)", (42, 24))
        val = cur.fetchone()[0]
        self.assertEqual(val, 2)

    def test_aggr_check_param_float(self):
        if False:
            i = 10
            return i + 15
        cur = self.con.cursor()
        cur.execute("select checkType('float', ?)", (3.14,))
        val = cur.fetchone()[0]
        self.assertEqual(val, 1)

    def test_aggr_check_param_none(self):
        if False:
            print('Hello World!')
        cur = self.con.cursor()
        cur.execute("select checkType('None', ?)", (None,))
        val = cur.fetchone()[0]
        self.assertEqual(val, 1)

    def test_aggr_check_param_blob(self):
        if False:
            while True:
                i = 10
        cur = self.con.cursor()
        cur.execute("select checkType('blob', ?)", (memoryview(b'blob'),))
        val = cur.fetchone()[0]
        self.assertEqual(val, 1)

    def test_aggr_check_aggr_sum(self):
        if False:
            return 10
        cur = self.con.cursor()
        cur.execute('delete from test')
        cur.executemany('insert into test(i) values (?)', [(10,), (20,), (30,)])
        cur.execute('select mysum(i) from test')
        val = cur.fetchone()[0]
        self.assertEqual(val, 60)

    def test_aggr_no_match(self):
        if False:
            print('Hello World!')
        cur = self.con.execute('select mysum(i) from (select 1 as i) where i == 0')
        val = cur.fetchone()[0]
        self.assertIsNone(val)

    def test_aggr_text(self):
        if False:
            print('Hello World!')
        cur = self.con.cursor()
        for txt in ['foo', '1\x002']:
            with self.subTest(txt=txt):
                cur.execute('select aggtxt(?) from test', (txt,))
                val = cur.fetchone()[0]
                self.assertEqual(val, txt)

class AuthorizerTests(unittest.TestCase):

    @staticmethod
    def authorizer_cb(action, arg1, arg2, dbname, source):
        if False:
            i = 10
            return i + 15
        if action != sqlite.SQLITE_SELECT:
            return sqlite.SQLITE_DENY
        if arg2 == 'c2' or arg1 == 't2':
            return sqlite.SQLITE_DENY
        return sqlite.SQLITE_OK

    def setUp(self):
        if False:
            print('Hello World!')
        self.con = sqlite.connect(':memory:')
        self.con.executescript('\n            create table t1 (c1, c2);\n            create table t2 (c1, c2);\n            insert into t1 (c1, c2) values (1, 2);\n            insert into t2 (c1, c2) values (4, 5);\n            ')
        self.con.execute('select c2 from t2')
        self.con.set_authorizer(self.authorizer_cb)

    def tearDown(self):
        if False:
            return 10
        pass

    def test_table_access(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(sqlite.DatabaseError) as cm:
            self.con.execute('select * from t2')
        self.assertIn('prohibited', str(cm.exception))

    def test_column_access(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(sqlite.DatabaseError) as cm:
            self.con.execute('select c2 from t1')
        self.assertIn('prohibited', str(cm.exception))

class AuthorizerRaiseExceptionTests(AuthorizerTests):

    @staticmethod
    def authorizer_cb(action, arg1, arg2, dbname, source):
        if False:
            i = 10
            return i + 15
        if action != sqlite.SQLITE_SELECT:
            raise ValueError
        if arg2 == 'c2' or arg1 == 't2':
            raise ValueError
        return sqlite.SQLITE_OK

class AuthorizerIllegalTypeTests(AuthorizerTests):

    @staticmethod
    def authorizer_cb(action, arg1, arg2, dbname, source):
        if False:
            for i in range(10):
                print('nop')
        if action != sqlite.SQLITE_SELECT:
            return 0.0
        if arg2 == 'c2' or arg1 == 't2':
            return 0.0
        return sqlite.SQLITE_OK

class AuthorizerLargeIntegerTests(AuthorizerTests):

    @staticmethod
    def authorizer_cb(action, arg1, arg2, dbname, source):
        if False:
            return 10
        if action != sqlite.SQLITE_SELECT:
            return 2 ** 32
        if arg2 == 'c2' or arg1 == 't2':
            return 2 ** 32
        return sqlite.SQLITE_OK

def suite():
    if False:
        print('Hello World!')
    tests = [AggregateTests, AuthorizerIllegalTypeTests, AuthorizerLargeIntegerTests, AuthorizerRaiseExceptionTests, AuthorizerTests, FunctionTests]
    return unittest.TestSuite([unittest.TestLoader().loadTestsFromTestCase(t) for t in tests])

def test():
    if False:
        for i in range(10):
            print('nop')
    runner = unittest.TextTestRunner()
    runner.run(suite())
if __name__ == '__main__':
    test()