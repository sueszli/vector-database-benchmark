from collections import namedtuple
import contextlib
import itertools
import os
import pickle
import sys
from textwrap import dedent
import threading
import time
import unittest
from test import support
from test.support import import_helper
from test.support import script_helper
interpreters = import_helper.import_module('_xxsubinterpreters')

def _captured_script(script):
    if False:
        for i in range(10):
            print('nop')
    (r, w) = os.pipe()
    indented = script.replace('\n', '\n                ')
    wrapped = dedent(f"""\n        import contextlib\n        with open({w}, 'w', encoding="utf-8") as spipe:\n            with contextlib.redirect_stdout(spipe):\n                {indented}\n        """)
    return (wrapped, open(r, encoding='utf-8'))

def _run_output(interp, request, shared=None):
    if False:
        i = 10
        return i + 15
    (script, rpipe) = _captured_script(request)
    with rpipe:
        interpreters.run_string(interp, script, shared)
        return rpipe.read()

def _wait_for_interp_to_run(interp, timeout=None):
    if False:
        return 10
    if timeout is None:
        timeout = support.SHORT_TIMEOUT
    start_time = time.monotonic()
    deadline = start_time + timeout
    while not interpreters.is_running(interp):
        if time.monotonic() > deadline:
            raise RuntimeError('interp is not running')
        time.sleep(0.01)

@contextlib.contextmanager
def _running(interp):
    if False:
        print('Hello World!')
    (r, w) = os.pipe()

    def run():
        if False:
            return 10
        interpreters.run_string(interp, dedent(f'\n            # wait for "signal"\n            with open({r}, encoding="utf-8") as rpipe:\n                rpipe.read()\n            '))
    t = threading.Thread(target=run)
    t.start()
    _wait_for_interp_to_run(interp)
    yield
    with open(w, 'w', encoding='utf-8') as spipe:
        spipe.write('done')
    t.join()

def run_interp(id, source, **shared):
    if False:
        while True:
            i = 10
    _run_interp(id, source, shared)

def _run_interp(id, source, shared, _mainns={}):
    if False:
        for i in range(10):
            print('nop')
    source = dedent(source)
    main = interpreters.get_main()
    if main == id:
        if interpreters.get_current() != main:
            raise RuntimeError
        exec(source, _mainns)
    else:
        interpreters.run_string(id, source, shared)

class Interpreter(namedtuple('Interpreter', 'name id')):

    @classmethod
    def from_raw(cls, raw):
        if False:
            print('Hello World!')
        if isinstance(raw, cls):
            return raw
        elif isinstance(raw, str):
            return cls(raw)
        else:
            raise NotImplementedError

    def __new__(cls, name=None, id=None):
        if False:
            while True:
                i = 10
        main = interpreters.get_main()
        if id == main:
            if not name:
                name = 'main'
            elif name != 'main':
                raise ValueError('name mismatch (expected "main", got "{}")'.format(name))
            id = main
        elif id is not None:
            if not name:
                name = 'interp'
            elif name == 'main':
                raise ValueError('name mismatch (unexpected "main")')
            if not isinstance(id, interpreters.InterpreterID):
                id = interpreters.InterpreterID(id)
        elif not name or name == 'main':
            name = 'main'
            id = main
        else:
            id = interpreters.create()
        self = super().__new__(cls, name, id)
        return self

@contextlib.contextmanager
def expect_channel_closed():
    if False:
        return 10
    try:
        yield
    except interpreters.ChannelClosedError:
        pass
    else:
        assert False, 'channel not closed'

class ChannelAction(namedtuple('ChannelAction', 'action end interp')):

    def __new__(cls, action, end=None, interp=None):
        if False:
            for i in range(10):
                print('nop')
        if not end:
            end = 'both'
        if not interp:
            interp = 'main'
        self = super().__new__(cls, action, end, interp)
        return self

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        if self.action == 'use':
            if self.end not in ('same', 'opposite', 'send', 'recv'):
                raise ValueError(self.end)
        elif self.action in ('close', 'force-close'):
            if self.end not in ('both', 'same', 'opposite', 'send', 'recv'):
                raise ValueError(self.end)
        else:
            raise ValueError(self.action)
        if self.interp not in ('main', 'same', 'other', 'extra'):
            raise ValueError(self.interp)

    def resolve_end(self, end):
        if False:
            print('Hello World!')
        if self.end == 'same':
            return end
        elif self.end == 'opposite':
            return 'recv' if end == 'send' else 'send'
        else:
            return self.end

    def resolve_interp(self, interp, other, extra):
        if False:
            i = 10
            return i + 15
        if self.interp == 'same':
            return interp
        elif self.interp == 'other':
            if other is None:
                raise RuntimeError
            return other
        elif self.interp == 'extra':
            if extra is None:
                raise RuntimeError
            return extra
        elif self.interp == 'main':
            if interp.name == 'main':
                return interp
            elif other and other.name == 'main':
                return other
            else:
                raise RuntimeError

class ChannelState(namedtuple('ChannelState', 'pending closed')):

    def __new__(cls, pending=0, *, closed=False):
        if False:
            for i in range(10):
                print('nop')
        self = super().__new__(cls, pending, closed)
        return self

    def incr(self):
        if False:
            print('Hello World!')
        return type(self)(self.pending + 1, closed=self.closed)

    def decr(self):
        if False:
            return 10
        return type(self)(self.pending - 1, closed=self.closed)

    def close(self, *, force=True):
        if False:
            return 10
        if self.closed:
            if not force or self.pending == 0:
                return self
        return type(self)(0 if force else self.pending, closed=True)

def run_action(cid, action, end, state, *, hideclosed=True):
    if False:
        while True:
            i = 10
    if state.closed:
        if action == 'use' and end == 'recv' and state.pending:
            expectfail = False
        else:
            expectfail = True
    else:
        expectfail = False
    try:
        result = _run_action(cid, action, end, state)
    except interpreters.ChannelClosedError:
        if not hideclosed and (not expectfail):
            raise
        result = state.close()
    else:
        if expectfail:
            raise ...
    return result

def _run_action(cid, action, end, state):
    if False:
        print('Hello World!')
    if action == 'use':
        if end == 'send':
            interpreters.channel_send(cid, b'spam')
            return state.incr()
        elif end == 'recv':
            if not state.pending:
                try:
                    interpreters.channel_recv(cid)
                except interpreters.ChannelEmptyError:
                    return state
                else:
                    raise Exception('expected ChannelEmptyError')
            else:
                interpreters.channel_recv(cid)
                return state.decr()
        else:
            raise ValueError(end)
    elif action == 'close':
        kwargs = {}
        if end in ('recv', 'send'):
            kwargs[end] = True
        interpreters.channel_close(cid, **kwargs)
        return state.close()
    elif action == 'force-close':
        kwargs = {'force': True}
        if end in ('recv', 'send'):
            kwargs[end] = True
        interpreters.channel_close(cid, **kwargs)
        return state.close(force=True)
    else:
        raise ValueError(action)

def clean_up_interpreters():
    if False:
        while True:
            i = 10
    for id in interpreters.list_all():
        if id == 0:
            continue
        try:
            interpreters.destroy(id)
        except RuntimeError:
            pass

def clean_up_channels():
    if False:
        return 10
    for cid in interpreters.channel_list_all():
        try:
            interpreters.channel_destroy(cid)
        except interpreters.ChannelNotFoundError:
            pass

class TestBase(unittest.TestCase):

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        clean_up_interpreters()
        clean_up_channels()

class IsShareableTests(unittest.TestCase):

    def test_default_shareables(self):
        if False:
            while True:
                i = 10
        shareables = [None, b'spam', 'spam', 10, -10]
        for obj in shareables:
            with self.subTest(obj):
                self.assertTrue(interpreters.is_shareable(obj))

    def test_not_shareable(self):
        if False:
            print('Hello World!')

        class Cheese:

            def __init__(self, name):
                if False:
                    while True:
                        i = 10
                self.name = name

            def __str__(self):
                if False:
                    while True:
                        i = 10
                return self.name

        class SubBytes(bytes):
            """A subclass of a shareable type."""
        not_shareables = [True, False, NotImplemented, ..., type, object, object(), Exception(), 100.0, Cheese, Cheese('Wensleydale'), SubBytes(b'spam')]
        for obj in not_shareables:
            with self.subTest(repr(obj)):
                self.assertFalse(interpreters.is_shareable(obj))

class ShareableTypeTests(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.cid = interpreters.channel_create()

    def tearDown(self):
        if False:
            print('Hello World!')
        interpreters.channel_destroy(self.cid)
        super().tearDown()

    def _assert_values(self, values):
        if False:
            i = 10
            return i + 15
        for obj in values:
            with self.subTest(obj):
                interpreters.channel_send(self.cid, obj)
                got = interpreters.channel_recv(self.cid)
                self.assertEqual(got, obj)
                self.assertIs(type(got), type(obj))

    def test_singletons(self):
        if False:
            i = 10
            return i + 15
        for obj in [None]:
            with self.subTest(obj):
                interpreters.channel_send(self.cid, obj)
                got = interpreters.channel_recv(self.cid)
                self.assertIs(got, obj)

    def test_types(self):
        if False:
            print('Hello World!')
        self._assert_values([b'spam', 9999, self.cid])

    def test_bytes(self):
        if False:
            i = 10
            return i + 15
        self._assert_values((i.to_bytes(2, 'little', signed=True) for i in range(-1, 258)))

    def test_strs(self):
        if False:
            return 10
        self._assert_values(['hello world', '你好世界', ''])

    def test_int(self):
        if False:
            return 10
        self._assert_values(itertools.chain(range(-1, 258), [sys.maxsize, -sys.maxsize - 1]))

    def test_non_shareable_int(self):
        if False:
            while True:
                i = 10
        ints = [sys.maxsize + 1, -sys.maxsize - 2, 2 ** 1000]
        for i in ints:
            with self.subTest(i):
                with self.assertRaises(OverflowError):
                    interpreters.channel_send(self.cid, i)

class ListAllTests(TestBase):

    def test_initial(self):
        if False:
            print('Hello World!')
        main = interpreters.get_main()
        ids = interpreters.list_all()
        self.assertEqual(ids, [main])

    def test_after_creating(self):
        if False:
            i = 10
            return i + 15
        main = interpreters.get_main()
        first = interpreters.create()
        second = interpreters.create()
        ids = interpreters.list_all()
        self.assertEqual(ids, [main, first, second])

    def test_after_destroying(self):
        if False:
            print('Hello World!')
        main = interpreters.get_main()
        first = interpreters.create()
        second = interpreters.create()
        interpreters.destroy(first)
        ids = interpreters.list_all()
        self.assertEqual(ids, [main, second])

class GetCurrentTests(TestBase):

    def test_main(self):
        if False:
            for i in range(10):
                print('nop')
        main = interpreters.get_main()
        cur = interpreters.get_current()
        self.assertEqual(cur, main)
        self.assertIsInstance(cur, interpreters.InterpreterID)

    def test_subinterpreter(self):
        if False:
            while True:
                i = 10
        main = interpreters.get_main()
        interp = interpreters.create()
        out = _run_output(interp, dedent('\n            import _xxsubinterpreters as _interpreters\n            cur = _interpreters.get_current()\n            print(cur)\n            assert isinstance(cur, _interpreters.InterpreterID)\n            '))
        cur = int(out.strip())
        (_, expected) = interpreters.list_all()
        self.assertEqual(cur, expected)
        self.assertNotEqual(cur, main)

class GetMainTests(TestBase):

    def test_from_main(self):
        if False:
            print('Hello World!')
        [expected] = interpreters.list_all()
        main = interpreters.get_main()
        self.assertEqual(main, expected)
        self.assertIsInstance(main, interpreters.InterpreterID)

    def test_from_subinterpreter(self):
        if False:
            for i in range(10):
                print('nop')
        [expected] = interpreters.list_all()
        interp = interpreters.create()
        out = _run_output(interp, dedent('\n            import _xxsubinterpreters as _interpreters\n            main = _interpreters.get_main()\n            print(main)\n            assert isinstance(main, _interpreters.InterpreterID)\n            '))
        main = int(out.strip())
        self.assertEqual(main, expected)

class IsRunningTests(TestBase):

    def test_main(self):
        if False:
            while True:
                i = 10
        main = interpreters.get_main()
        self.assertTrue(interpreters.is_running(main))

    @unittest.skip('Fails on FreeBSD')
    def test_subinterpreter(self):
        if False:
            return 10
        interp = interpreters.create()
        self.assertFalse(interpreters.is_running(interp))
        with _running(interp):
            self.assertTrue(interpreters.is_running(interp))
        self.assertFalse(interpreters.is_running(interp))

    def test_from_subinterpreter(self):
        if False:
            while True:
                i = 10
        interp = interpreters.create()
        out = _run_output(interp, dedent(f'\n            import _xxsubinterpreters as _interpreters\n            if _interpreters.is_running({interp}):\n                print(True)\n            else:\n                print(False)\n            '))
        self.assertEqual(out.strip(), 'True')

    def test_already_destroyed(self):
        if False:
            return 10
        interp = interpreters.create()
        interpreters.destroy(interp)
        with self.assertRaises(RuntimeError):
            interpreters.is_running(interp)

    def test_does_not_exist(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(RuntimeError):
            interpreters.is_running(1000000)

    def test_bad_id(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            interpreters.is_running(-1)

class InterpreterIDTests(TestBase):

    def test_with_int(self):
        if False:
            print('Hello World!')
        id = interpreters.InterpreterID(10, force=True)
        self.assertEqual(int(id), 10)

    def test_coerce_id(self):
        if False:
            return 10

        class Int(str):

            def __index__(self):
                if False:
                    while True:
                        i = 10
                return 10
        id = interpreters.InterpreterID(Int(), force=True)
        self.assertEqual(int(id), 10)

    def test_bad_id(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, interpreters.InterpreterID, object())
        self.assertRaises(TypeError, interpreters.InterpreterID, 10.0)
        self.assertRaises(TypeError, interpreters.InterpreterID, '10')
        self.assertRaises(TypeError, interpreters.InterpreterID, b'10')
        self.assertRaises(ValueError, interpreters.InterpreterID, -1)
        self.assertRaises(OverflowError, interpreters.InterpreterID, 2 ** 64)

    def test_does_not_exist(self):
        if False:
            while True:
                i = 10
        id = interpreters.channel_create()
        with self.assertRaises(RuntimeError):
            interpreters.InterpreterID(int(id) + 1)

    def test_str(self):
        if False:
            print('Hello World!')
        id = interpreters.InterpreterID(10, force=True)
        self.assertEqual(str(id), '10')

    def test_repr(self):
        if False:
            print('Hello World!')
        id = interpreters.InterpreterID(10, force=True)
        self.assertEqual(repr(id), 'InterpreterID(10)')

    def test_equality(self):
        if False:
            i = 10
            return i + 15
        id1 = interpreters.create()
        id2 = interpreters.InterpreterID(int(id1))
        id3 = interpreters.create()
        self.assertTrue(id1 == id1)
        self.assertTrue(id1 == id2)
        self.assertTrue(id1 == int(id1))
        self.assertTrue(int(id1) == id1)
        self.assertTrue(id1 == float(int(id1)))
        self.assertTrue(float(int(id1)) == id1)
        self.assertFalse(id1 == float(int(id1)) + 0.1)
        self.assertFalse(id1 == str(int(id1)))
        self.assertFalse(id1 == 2 ** 1000)
        self.assertFalse(id1 == float('inf'))
        self.assertFalse(id1 == 'spam')
        self.assertFalse(id1 == id3)
        self.assertFalse(id1 != id1)
        self.assertFalse(id1 != id2)
        self.assertTrue(id1 != id3)

class CreateTests(TestBase):

    def test_in_main(self):
        if False:
            return 10
        id = interpreters.create()
        self.assertIsInstance(id, interpreters.InterpreterID)
        self.assertIn(id, interpreters.list_all())

    @unittest.skip('enable this test when working on pystate.c')
    def test_unique_id(self):
        if False:
            return 10
        seen = set()
        for _ in range(100):
            id = interpreters.create()
            interpreters.destroy(id)
            seen.add(id)
        self.assertEqual(len(seen), 100)

    def test_in_thread(self):
        if False:
            for i in range(10):
                print('nop')
        lock = threading.Lock()
        id = None

        def f():
            if False:
                return 10
            nonlocal id
            id = interpreters.create()
            lock.acquire()
            lock.release()
        t = threading.Thread(target=f)
        with lock:
            t.start()
        t.join()
        self.assertIn(id, interpreters.list_all())

    def test_in_subinterpreter(self):
        if False:
            for i in range(10):
                print('nop')
        (main,) = interpreters.list_all()
        id1 = interpreters.create()
        out = _run_output(id1, dedent('\n            import _xxsubinterpreters as _interpreters\n            id = _interpreters.create()\n            print(id)\n            assert isinstance(id, _interpreters.InterpreterID)\n            '))
        id2 = int(out.strip())
        self.assertEqual(set(interpreters.list_all()), {main, id1, id2})

    def test_in_threaded_subinterpreter(self):
        if False:
            print('Hello World!')
        (main,) = interpreters.list_all()
        id1 = interpreters.create()
        id2 = None

        def f():
            if False:
                while True:
                    i = 10
            nonlocal id2
            out = _run_output(id1, dedent('\n                import _xxsubinterpreters as _interpreters\n                id = _interpreters.create()\n                print(id)\n                '))
            id2 = int(out.strip())
        t = threading.Thread(target=f)
        t.start()
        t.join()
        self.assertEqual(set(interpreters.list_all()), {main, id1, id2})

    def test_after_destroy_all(self):
        if False:
            while True:
                i = 10
        before = set(interpreters.list_all())
        ids = []
        for _ in range(3):
            id = interpreters.create()
            ids.append(id)
        for id in ids:
            interpreters.destroy(id)
        id = interpreters.create()
        self.assertEqual(set(interpreters.list_all()), before | {id})

    def test_after_destroy_some(self):
        if False:
            print('Hello World!')
        before = set(interpreters.list_all())
        id1 = interpreters.create()
        id2 = interpreters.create()
        id3 = interpreters.create()
        interpreters.destroy(id1)
        interpreters.destroy(id3)
        id = interpreters.create()
        self.assertEqual(set(interpreters.list_all()), before | {id, id2})

class DestroyTests(TestBase):

    def test_one(self):
        if False:
            print('Hello World!')
        id1 = interpreters.create()
        id2 = interpreters.create()
        id3 = interpreters.create()
        self.assertIn(id2, interpreters.list_all())
        interpreters.destroy(id2)
        self.assertNotIn(id2, interpreters.list_all())
        self.assertIn(id1, interpreters.list_all())
        self.assertIn(id3, interpreters.list_all())

    def test_all(self):
        if False:
            i = 10
            return i + 15
        before = set(interpreters.list_all())
        ids = set()
        for _ in range(3):
            id = interpreters.create()
            ids.add(id)
        self.assertEqual(set(interpreters.list_all()), before | ids)
        for id in ids:
            interpreters.destroy(id)
        self.assertEqual(set(interpreters.list_all()), before)

    def test_main(self):
        if False:
            print('Hello World!')
        (main,) = interpreters.list_all()
        with self.assertRaises(RuntimeError):
            interpreters.destroy(main)

        def f():
            if False:
                return 10
            with self.assertRaises(RuntimeError):
                interpreters.destroy(main)
        t = threading.Thread(target=f)
        t.start()
        t.join()

    def test_already_destroyed(self):
        if False:
            while True:
                i = 10
        id = interpreters.create()
        interpreters.destroy(id)
        with self.assertRaises(RuntimeError):
            interpreters.destroy(id)

    def test_does_not_exist(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(RuntimeError):
            interpreters.destroy(1000000)

    def test_bad_id(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            interpreters.destroy(-1)

    def test_from_current(self):
        if False:
            i = 10
            return i + 15
        (main,) = interpreters.list_all()
        id = interpreters.create()
        script = dedent(f'\n            import _xxsubinterpreters as _interpreters\n            try:\n                _interpreters.destroy({id})\n            except RuntimeError:\n                pass\n            ')
        interpreters.run_string(id, script)
        self.assertEqual(set(interpreters.list_all()), {main, id})

    def test_from_sibling(self):
        if False:
            for i in range(10):
                print('nop')
        (main,) = interpreters.list_all()
        id1 = interpreters.create()
        id2 = interpreters.create()
        script = dedent(f'\n            import _xxsubinterpreters as _interpreters\n            _interpreters.destroy({id2})\n            ')
        interpreters.run_string(id1, script)
        self.assertEqual(set(interpreters.list_all()), {main, id1})

    def test_from_other_thread(self):
        if False:
            i = 10
            return i + 15
        id = interpreters.create()

        def f():
            if False:
                for i in range(10):
                    print('nop')
            interpreters.destroy(id)
        t = threading.Thread(target=f)
        t.start()
        t.join()

    def test_still_running(self):
        if False:
            for i in range(10):
                print('nop')
        (main,) = interpreters.list_all()
        interp = interpreters.create()
        with _running(interp):
            self.assertTrue(interpreters.is_running(interp), msg=f'Interp {interp} should be running before destruction.')
            with self.assertRaises(RuntimeError, msg=f"Should not be able to destroy interp {interp} while it's still running."):
                interpreters.destroy(interp)
            self.assertTrue(interpreters.is_running(interp))

class RunStringTests(TestBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.id = interpreters.create()

    def test_success(self):
        if False:
            print('Hello World!')
        (script, file) = _captured_script('print("it worked!", end="")')
        with file:
            interpreters.run_string(self.id, script)
            out = file.read()
        self.assertEqual(out, 'it worked!')

    def test_in_thread(self):
        if False:
            for i in range(10):
                print('nop')
        (script, file) = _captured_script('print("it worked!", end="")')
        with file:

            def f():
                if False:
                    return 10
                interpreters.run_string(self.id, script)
            t = threading.Thread(target=f)
            t.start()
            t.join()
            out = file.read()
        self.assertEqual(out, 'it worked!')

    def test_create_thread(self):
        if False:
            while True:
                i = 10
        subinterp = interpreters.create(isolated=False)
        (script, file) = _captured_script("\n            import threading\n            def f():\n                print('it worked!', end='')\n\n            t = threading.Thread(target=f)\n            t.start()\n            t.join()\n            ")
        with file:
            interpreters.run_string(subinterp, script)
            out = file.read()
        self.assertEqual(out, 'it worked!')

    @unittest.skipUnless(hasattr(os, 'fork'), 'test needs os.fork()')
    def test_fork(self):
        if False:
            i = 10
            return i + 15
        import tempfile
        with tempfile.NamedTemporaryFile('w+', encoding='utf-8') as file:
            file.write('')
            file.flush()
            expected = 'spam spam spam spam spam'
            script = dedent(f"\n                import os\n                try:\n                    os.fork()\n                except RuntimeError:\n                    with open('{file.name}', 'w', encoding='utf-8') as out:\n                        out.write('{expected}')\n                ")
            interpreters.run_string(self.id, script)
            file.seek(0)
            content = file.read()
            self.assertEqual(content, expected)

    def test_already_running(self):
        if False:
            return 10
        with _running(self.id):
            with self.assertRaises(RuntimeError):
                interpreters.run_string(self.id, 'print("spam")')

    def test_does_not_exist(self):
        if False:
            for i in range(10):
                print('nop')
        id = 0
        while id in interpreters.list_all():
            id += 1
        with self.assertRaises(RuntimeError):
            interpreters.run_string(id, 'print("spam")')

    def test_error_id(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            interpreters.run_string(-1, 'print("spam")')

    def test_bad_id(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            interpreters.run_string('spam', 'print("spam")')

    def test_bad_script(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            interpreters.run_string(self.id, 10)

    def test_bytes_for_script(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError):
            interpreters.run_string(self.id, b'print("spam")')

    @contextlib.contextmanager
    def assert_run_failed(self, exctype, msg=None):
        if False:
            print('Hello World!')
        with self.assertRaises(interpreters.RunFailedError) as caught:
            yield
        if msg is None:
            self.assertEqual(str(caught.exception).split(':')[0], str(exctype))
        else:
            self.assertEqual(str(caught.exception), '{}: {}'.format(exctype, msg))

    def test_invalid_syntax(self):
        if False:
            while True:
                i = 10
        with self.assert_run_failed(SyntaxError):
            interpreters.run_string(self.id, 'print("spam"')

    def test_failure(self):
        if False:
            while True:
                i = 10
        with self.assert_run_failed(Exception, 'spam'):
            interpreters.run_string(self.id, 'raise Exception("spam")')

    def test_SystemExit(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assert_run_failed(SystemExit, '42'):
            interpreters.run_string(self.id, 'raise SystemExit(42)')

    def test_sys_exit(self):
        if False:
            while True:
                i = 10
        with self.assert_run_failed(SystemExit):
            interpreters.run_string(self.id, dedent('\n                import sys\n                sys.exit()\n                '))
        with self.assert_run_failed(SystemExit, '42'):
            interpreters.run_string(self.id, dedent('\n                import sys\n                sys.exit(42)\n                '))

    def test_with_shared(self):
        if False:
            print('Hello World!')
        (r, w) = os.pipe()
        shared = {'spam': b'ham', 'eggs': b'-1', 'cheddar': None}
        script = dedent(f"\n            eggs = int(eggs)\n            spam = 42\n            result = spam + eggs\n\n            ns = dict(vars())\n            del ns['__builtins__']\n            import pickle\n            with open({w}, 'wb') as chan:\n                pickle.dump(ns, chan)\n            ")
        interpreters.run_string(self.id, script, shared)
        with open(r, 'rb') as chan:
            ns = pickle.load(chan)
        self.assertEqual(ns['spam'], 42)
        self.assertEqual(ns['eggs'], -1)
        self.assertEqual(ns['result'], 41)
        self.assertIsNone(ns['cheddar'])

    def test_shared_overwrites(self):
        if False:
            for i in range(10):
                print('nop')
        interpreters.run_string(self.id, dedent("\n            spam = 'eggs'\n            ns1 = dict(vars())\n            del ns1['__builtins__']\n            "))
        shared = {'spam': b'ham'}
        script = dedent(f"\n            ns2 = dict(vars())\n            del ns2['__builtins__']\n        ")
        interpreters.run_string(self.id, script, shared)
        (r, w) = os.pipe()
        script = dedent(f"\n            ns = dict(vars())\n            del ns['__builtins__']\n            import pickle\n            with open({w}, 'wb') as chan:\n                pickle.dump(ns, chan)\n            ")
        interpreters.run_string(self.id, script)
        with open(r, 'rb') as chan:
            ns = pickle.load(chan)
        self.assertEqual(ns['ns1']['spam'], 'eggs')
        self.assertEqual(ns['ns2']['spam'], b'ham')
        self.assertEqual(ns['spam'], b'ham')

    def test_shared_overwrites_default_vars(self):
        if False:
            while True:
                i = 10
        (r, w) = os.pipe()
        shared = {'__name__': b'not __main__'}
        script = dedent(f"\n            spam = 42\n\n            ns = dict(vars())\n            del ns['__builtins__']\n            import pickle\n            with open({w}, 'wb') as chan:\n                pickle.dump(ns, chan)\n            ")
        interpreters.run_string(self.id, script, shared)
        with open(r, 'rb') as chan:
            ns = pickle.load(chan)
        self.assertEqual(ns['__name__'], b'not __main__')

    def test_main_reused(self):
        if False:
            while True:
                i = 10
        (r, w) = os.pipe()
        interpreters.run_string(self.id, dedent(f"\n            spam = True\n\n            ns = dict(vars())\n            del ns['__builtins__']\n            import pickle\n            with open({w}, 'wb') as chan:\n                pickle.dump(ns, chan)\n            del ns, pickle, chan\n            "))
        with open(r, 'rb') as chan:
            ns1 = pickle.load(chan)
        (r, w) = os.pipe()
        interpreters.run_string(self.id, dedent(f"\n            eggs = False\n\n            ns = dict(vars())\n            del ns['__builtins__']\n            import pickle\n            with open({w}, 'wb') as chan:\n                pickle.dump(ns, chan)\n            "))
        with open(r, 'rb') as chan:
            ns2 = pickle.load(chan)
        self.assertIn('spam', ns1)
        self.assertNotIn('eggs', ns1)
        self.assertIn('eggs', ns2)
        self.assertIn('spam', ns2)

    def test_execution_namespace_is_main(self):
        if False:
            for i in range(10):
                print('nop')
        (r, w) = os.pipe()
        script = dedent(f"\n            spam = 42\n\n            ns = dict(vars())\n            ns['__builtins__'] = str(ns['__builtins__'])\n            import pickle\n            with open({w}, 'wb') as chan:\n                pickle.dump(ns, chan)\n            ")
        interpreters.run_string(self.id, script)
        with open(r, 'rb') as chan:
            ns = pickle.load(chan)
        ns.pop('__builtins__')
        ns.pop('__loader__')
        self.assertEqual(ns, {'__name__': '__main__', '__annotations__': {}, '__doc__': None, '__package__': None, '__spec__': None, 'spam': 42})

    @unittest.skip('blocking forever')
    def test_still_running_at_exit(self):
        if False:
            i = 10
            return i + 15
        script = dedent(f"\n        from textwrap import dedent\n        import threading\n        import _xxsubinterpreters as _interpreters\n        id = _interpreters.create()\n        def f():\n            _interpreters.run_string(id, dedent('''\n                import time\n                # Give plenty of time for the main interpreter to finish.\n                time.sleep(1_000_000)\n                '''))\n\n        t = threading.Thread(target=f)\n        t.start()\n        ")
        with support.temp_dir() as dirname:
            filename = script_helper.make_script(dirname, 'interp', script)
            with script_helper.spawn_python(filename) as proc:
                retcode = proc.wait()
        self.assertEqual(retcode, 0)

class ChannelIDTests(TestBase):

    def test_default_kwargs(self):
        if False:
            while True:
                i = 10
        cid = interpreters._channel_id(10, force=True)
        self.assertEqual(int(cid), 10)
        self.assertEqual(cid.end, 'both')

    def test_with_kwargs(self):
        if False:
            print('Hello World!')
        cid = interpreters._channel_id(10, send=True, force=True)
        self.assertEqual(cid.end, 'send')
        cid = interpreters._channel_id(10, send=True, recv=False, force=True)
        self.assertEqual(cid.end, 'send')
        cid = interpreters._channel_id(10, recv=True, force=True)
        self.assertEqual(cid.end, 'recv')
        cid = interpreters._channel_id(10, recv=True, send=False, force=True)
        self.assertEqual(cid.end, 'recv')
        cid = interpreters._channel_id(10, send=True, recv=True, force=True)
        self.assertEqual(cid.end, 'both')

    def test_coerce_id(self):
        if False:
            while True:
                i = 10

        class Int(str):

            def __index__(self):
                if False:
                    while True:
                        i = 10
                return 10
        cid = interpreters._channel_id(Int(), force=True)
        self.assertEqual(int(cid), 10)

    def test_bad_id(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, interpreters._channel_id, object())
        self.assertRaises(TypeError, interpreters._channel_id, 10.0)
        self.assertRaises(TypeError, interpreters._channel_id, '10')
        self.assertRaises(TypeError, interpreters._channel_id, b'10')
        self.assertRaises(ValueError, interpreters._channel_id, -1)
        self.assertRaises(OverflowError, interpreters._channel_id, 2 ** 64)

    def test_bad_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            interpreters._channel_id(10, send=False, recv=False)

    def test_does_not_exist(self):
        if False:
            i = 10
            return i + 15
        cid = interpreters.channel_create()
        with self.assertRaises(interpreters.ChannelNotFoundError):
            interpreters._channel_id(int(cid) + 1)

    def test_str(self):
        if False:
            for i in range(10):
                print('nop')
        cid = interpreters._channel_id(10, force=True)
        self.assertEqual(str(cid), '10')

    def test_repr(self):
        if False:
            print('Hello World!')
        cid = interpreters._channel_id(10, force=True)
        self.assertEqual(repr(cid), 'ChannelID(10)')
        cid = interpreters._channel_id(10, send=True, force=True)
        self.assertEqual(repr(cid), 'ChannelID(10, send=True)')
        cid = interpreters._channel_id(10, recv=True, force=True)
        self.assertEqual(repr(cid), 'ChannelID(10, recv=True)')
        cid = interpreters._channel_id(10, send=True, recv=True, force=True)
        self.assertEqual(repr(cid), 'ChannelID(10)')

    def test_equality(self):
        if False:
            return 10
        cid1 = interpreters.channel_create()
        cid2 = interpreters._channel_id(int(cid1))
        cid3 = interpreters.channel_create()
        self.assertTrue(cid1 == cid1)
        self.assertTrue(cid1 == cid2)
        self.assertTrue(cid1 == int(cid1))
        self.assertTrue(int(cid1) == cid1)
        self.assertTrue(cid1 == float(int(cid1)))
        self.assertTrue(float(int(cid1)) == cid1)
        self.assertFalse(cid1 == float(int(cid1)) + 0.1)
        self.assertFalse(cid1 == str(int(cid1)))
        self.assertFalse(cid1 == 2 ** 1000)
        self.assertFalse(cid1 == float('inf'))
        self.assertFalse(cid1 == 'spam')
        self.assertFalse(cid1 == cid3)
        self.assertFalse(cid1 != cid1)
        self.assertFalse(cid1 != cid2)
        self.assertTrue(cid1 != cid3)

class ChannelTests(TestBase):

    def test_create_cid(self):
        if False:
            while True:
                i = 10
        cid = interpreters.channel_create()
        self.assertIsInstance(cid, interpreters.ChannelID)

    def test_sequential_ids(self):
        if False:
            for i in range(10):
                print('nop')
        before = interpreters.channel_list_all()
        id1 = interpreters.channel_create()
        id2 = interpreters.channel_create()
        id3 = interpreters.channel_create()
        after = interpreters.channel_list_all()
        self.assertEqual(id2, int(id1) + 1)
        self.assertEqual(id3, int(id2) + 1)
        self.assertEqual(set(after) - set(before), {id1, id2, id3})

    def test_ids_global(self):
        if False:
            for i in range(10):
                print('nop')
        id1 = interpreters.create()
        out = _run_output(id1, dedent('\n            import _xxsubinterpreters as _interpreters\n            cid = _interpreters.channel_create()\n            print(cid)\n            '))
        cid1 = int(out.strip())
        id2 = interpreters.create()
        out = _run_output(id2, dedent('\n            import _xxsubinterpreters as _interpreters\n            cid = _interpreters.channel_create()\n            print(cid)\n            '))
        cid2 = int(out.strip())
        self.assertEqual(cid2, int(cid1) + 1)

    def test_channel_list_interpreters_none(self):
        if False:
            i = 10
            return i + 15
        'Test listing interpreters for a channel with no associations.'
        cid = interpreters.channel_create()
        send_interps = interpreters.channel_list_interpreters(cid, send=True)
        recv_interps = interpreters.channel_list_interpreters(cid, send=False)
        self.assertEqual(send_interps, [])
        self.assertEqual(recv_interps, [])

    def test_channel_list_interpreters_basic(self):
        if False:
            while True:
                i = 10
        'Test basic listing channel interpreters.'
        interp0 = interpreters.get_main()
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, 'send')
        send_interps = interpreters.channel_list_interpreters(cid, send=True)
        recv_interps = interpreters.channel_list_interpreters(cid, send=False)
        self.assertEqual(send_interps, [interp0])
        self.assertEqual(recv_interps, [])
        interp1 = interpreters.create()
        _run_output(interp1, dedent(f'\n            import _xxsubinterpreters as _interpreters\n            obj = _interpreters.channel_recv({cid})\n            '))
        send_interps = interpreters.channel_list_interpreters(cid, send=True)
        recv_interps = interpreters.channel_list_interpreters(cid, send=False)
        self.assertEqual(send_interps, [interp0])
        self.assertEqual(recv_interps, [interp1])

    def test_channel_list_interpreters_multiple(self):
        if False:
            i = 10
            return i + 15
        'Test listing interpreters for a channel with many associations.'
        interp0 = interpreters.get_main()
        interp1 = interpreters.create()
        interp2 = interpreters.create()
        interp3 = interpreters.create()
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, 'send')
        _run_output(interp1, dedent(f'\n            import _xxsubinterpreters as _interpreters\n            _interpreters.channel_send({cid}, "send")\n            '))
        _run_output(interp2, dedent(f'\n            import _xxsubinterpreters as _interpreters\n            obj = _interpreters.channel_recv({cid})\n            '))
        _run_output(interp3, dedent(f'\n            import _xxsubinterpreters as _interpreters\n            obj = _interpreters.channel_recv({cid})\n            '))
        send_interps = interpreters.channel_list_interpreters(cid, send=True)
        recv_interps = interpreters.channel_list_interpreters(cid, send=False)
        self.assertEqual(set(send_interps), {interp0, interp1})
        self.assertEqual(set(recv_interps), {interp2, interp3})

    def test_channel_list_interpreters_destroyed(self):
        if False:
            while True:
                i = 10
        'Test listing channel interpreters with a destroyed interpreter.'
        interp0 = interpreters.get_main()
        interp1 = interpreters.create()
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, 'send')
        _run_output(interp1, dedent(f'\n            import _xxsubinterpreters as _interpreters\n            obj = _interpreters.channel_recv({cid})\n            '))
        send_interps = interpreters.channel_list_interpreters(cid, send=True)
        recv_interps = interpreters.channel_list_interpreters(cid, send=False)
        self.assertEqual(send_interps, [interp0])
        self.assertEqual(recv_interps, [interp1])
        interpreters.destroy(interp1)
        send_interps = interpreters.channel_list_interpreters(cid, send=True)
        recv_interps = interpreters.channel_list_interpreters(cid, send=False)
        self.assertEqual(send_interps, [interp0])
        self.assertEqual(recv_interps, [])

    def test_channel_list_interpreters_released(self):
        if False:
            print('Hello World!')
        'Test listing channel interpreters with a released channel.'
        interp0 = interpreters.get_main()
        interp1 = interpreters.create()
        interp2 = interpreters.create()
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, 'data')
        _run_output(interp1, dedent(f'\n            import _xxsubinterpreters as _interpreters\n            obj = _interpreters.channel_recv({cid})\n            '))
        interpreters.channel_send(cid, 'data')
        _run_output(interp2, dedent(f'\n            import _xxsubinterpreters as _interpreters\n            obj = _interpreters.channel_recv({cid})\n            '))
        send_interps = interpreters.channel_list_interpreters(cid, send=True)
        recv_interps = interpreters.channel_list_interpreters(cid, send=False)
        self.assertEqual(len(send_interps), 1)
        self.assertEqual(len(recv_interps), 2)
        interpreters.channel_release(cid, send=True)
        send_interps = interpreters.channel_list_interpreters(cid, send=True)
        recv_interps = interpreters.channel_list_interpreters(cid, send=False)
        self.assertEqual(len(send_interps), 0)
        self.assertEqual(len(recv_interps), 2)
        _run_output(interp2, dedent(f'\n            import _xxsubinterpreters as _interpreters\n            _interpreters.channel_release({cid})\n            '))
        send_interps = interpreters.channel_list_interpreters(cid, send=True)
        recv_interps = interpreters.channel_list_interpreters(cid, send=False)
        self.assertEqual(len(send_interps), 0)
        self.assertEqual(recv_interps, [interp1])

    def test_channel_list_interpreters_closed(self):
        if False:
            print('Hello World!')
        'Test listing channel interpreters with a closed channel.'
        interp0 = interpreters.get_main()
        interp1 = interpreters.create()
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, 'send')
        send_interps = interpreters.channel_list_interpreters(cid, send=True)
        recv_interps = interpreters.channel_list_interpreters(cid, send=False)
        self.assertEqual(len(send_interps), 1)
        self.assertEqual(len(recv_interps), 0)
        interpreters.channel_close(cid, force=True)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_list_interpreters(cid, send=True)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_list_interpreters(cid, send=False)

    def test_channel_list_interpreters_closed_send_end(self):
        if False:
            for i in range(10):
                print('nop')
        "Test listing channel interpreters with a channel's send end closed."
        interp0 = interpreters.get_main()
        interp1 = interpreters.create()
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, 'send')
        send_interps = interpreters.channel_list_interpreters(cid, send=True)
        recv_interps = interpreters.channel_list_interpreters(cid, send=False)
        self.assertEqual(len(send_interps), 1)
        self.assertEqual(len(recv_interps), 0)
        interpreters.channel_close(cid, send=True)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_list_interpreters(cid, send=True)
        recv_interps = interpreters.channel_list_interpreters(cid, send=False)
        self.assertEqual(len(recv_interps), 0)
        _run_output(interp1, dedent(f'\n            import _xxsubinterpreters as _interpreters\n            _interpreters.channel_close({cid}, force=True)\n            '))
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_list_interpreters(cid, send=True)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_list_interpreters(cid, send=False)

    def test_send_recv_main(self):
        if False:
            return 10
        cid = interpreters.channel_create()
        orig = b'spam'
        interpreters.channel_send(cid, orig)
        obj = interpreters.channel_recv(cid)
        self.assertEqual(obj, orig)
        self.assertIsNot(obj, orig)

    def test_send_recv_same_interpreter(self):
        if False:
            print('Hello World!')
        id1 = interpreters.create()
        out = _run_output(id1, dedent("\n            import _xxsubinterpreters as _interpreters\n            cid = _interpreters.channel_create()\n            orig = b'spam'\n            _interpreters.channel_send(cid, orig)\n            obj = _interpreters.channel_recv(cid)\n            assert obj is not orig\n            assert obj == orig\n            "))

    def test_send_recv_different_interpreters(self):
        if False:
            while True:
                i = 10
        cid = interpreters.channel_create()
        id1 = interpreters.create()
        out = _run_output(id1, dedent(f"\n            import _xxsubinterpreters as _interpreters\n            _interpreters.channel_send({cid}, b'spam')\n            "))
        obj = interpreters.channel_recv(cid)
        self.assertEqual(obj, b'spam')

    def test_send_recv_different_threads(self):
        if False:
            i = 10
            return i + 15
        cid = interpreters.channel_create()

        def f():
            if False:
                print('Hello World!')
            while True:
                try:
                    obj = interpreters.channel_recv(cid)
                    break
                except interpreters.ChannelEmptyError:
                    time.sleep(0.1)
            interpreters.channel_send(cid, obj)
        t = threading.Thread(target=f)
        t.start()
        interpreters.channel_send(cid, b'spam')
        t.join()
        obj = interpreters.channel_recv(cid)
        self.assertEqual(obj, b'spam')

    def test_send_recv_different_interpreters_and_threads(self):
        if False:
            while True:
                i = 10
        cid = interpreters.channel_create()
        id1 = interpreters.create()
        out = None

        def f():
            if False:
                return 10
            nonlocal out
            out = _run_output(id1, dedent(f"\n                import time\n                import _xxsubinterpreters as _interpreters\n                while True:\n                    try:\n                        obj = _interpreters.channel_recv({cid})\n                        break\n                    except _interpreters.ChannelEmptyError:\n                        time.sleep(0.1)\n                assert(obj == b'spam')\n                _interpreters.channel_send({cid}, b'eggs')\n                "))
        t = threading.Thread(target=f)
        t.start()
        interpreters.channel_send(cid, b'spam')
        t.join()
        obj = interpreters.channel_recv(cid)
        self.assertEqual(obj, b'eggs')

    def test_send_not_found(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(interpreters.ChannelNotFoundError):
            interpreters.channel_send(10, b'spam')

    def test_recv_not_found(self):
        if False:
            print('Hello World!')
        with self.assertRaises(interpreters.ChannelNotFoundError):
            interpreters.channel_recv(10)

    def test_recv_empty(self):
        if False:
            for i in range(10):
                print('nop')
        cid = interpreters.channel_create()
        with self.assertRaises(interpreters.ChannelEmptyError):
            interpreters.channel_recv(cid)

    def test_recv_default(self):
        if False:
            i = 10
            return i + 15
        default = object()
        cid = interpreters.channel_create()
        obj1 = interpreters.channel_recv(cid, default)
        interpreters.channel_send(cid, None)
        interpreters.channel_send(cid, 1)
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_send(cid, b'eggs')
        obj2 = interpreters.channel_recv(cid, default)
        obj3 = interpreters.channel_recv(cid, default)
        obj4 = interpreters.channel_recv(cid)
        obj5 = interpreters.channel_recv(cid, default)
        obj6 = interpreters.channel_recv(cid, default)
        self.assertIs(obj1, default)
        self.assertIs(obj2, None)
        self.assertEqual(obj3, 1)
        self.assertEqual(obj4, b'spam')
        self.assertEqual(obj5, b'eggs')
        self.assertIs(obj6, default)

    def test_run_string_arg_unresolved(self):
        if False:
            i = 10
            return i + 15
        cid = interpreters.channel_create()
        interp = interpreters.create()
        out = _run_output(interp, dedent("\n            import _xxsubinterpreters as _interpreters\n            print(cid.end)\n            _interpreters.channel_send(cid, b'spam')\n            "), dict(cid=cid.send))
        obj = interpreters.channel_recv(cid)
        self.assertEqual(obj, b'spam')
        self.assertEqual(out.strip(), 'send')

    @unittest.skip('disabled until high-level channels exist')
    def test_run_string_arg_resolved(self):
        if False:
            while True:
                i = 10
        cid = interpreters.channel_create()
        cid = interpreters._channel_id(cid, _resolve=True)
        interp = interpreters.create()
        out = _run_output(interp, dedent("\n            import _xxsubinterpreters as _interpreters\n            print(chan.id.end)\n            _interpreters.channel_send(chan.id, b'spam')\n            "), dict(chan=cid.send))
        obj = interpreters.channel_recv(cid)
        self.assertEqual(obj, b'spam')
        self.assertEqual(out.strip(), 'send')

    def test_close_single_user(self):
        if False:
            while True:
                i = 10
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_recv(cid)
        interpreters.channel_close(cid)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_send(cid, b'eggs')
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_recv(cid)

    def test_close_multiple_users(self):
        if False:
            print('Hello World!')
        cid = interpreters.channel_create()
        id1 = interpreters.create()
        id2 = interpreters.create()
        interpreters.run_string(id1, dedent(f"\n            import _xxsubinterpreters as _interpreters\n            _interpreters.channel_send({cid}, b'spam')\n            "))
        interpreters.run_string(id2, dedent(f'\n            import _xxsubinterpreters as _interpreters\n            _interpreters.channel_recv({cid})\n            '))
        interpreters.channel_close(cid)
        with self.assertRaises(interpreters.RunFailedError) as cm:
            interpreters.run_string(id1, dedent(f"\n                _interpreters.channel_send({cid}, b'spam')\n                "))
        self.assertIn('ChannelClosedError', str(cm.exception))
        with self.assertRaises(interpreters.RunFailedError) as cm:
            interpreters.run_string(id2, dedent(f"\n                _interpreters.channel_send({cid}, b'spam')\n                "))
        self.assertIn('ChannelClosedError', str(cm.exception))

    def test_close_multiple_times(self):
        if False:
            for i in range(10):
                print('nop')
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_recv(cid)
        interpreters.channel_close(cid)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_close(cid)

    def test_close_empty(self):
        if False:
            while True:
                i = 10
        tests = [(False, False), (True, False), (False, True), (True, True)]
        for (send, recv) in tests:
            with self.subTest((send, recv)):
                cid = interpreters.channel_create()
                interpreters.channel_send(cid, b'spam')
                interpreters.channel_recv(cid)
                interpreters.channel_close(cid, send=send, recv=recv)
                with self.assertRaises(interpreters.ChannelClosedError):
                    interpreters.channel_send(cid, b'eggs')
                with self.assertRaises(interpreters.ChannelClosedError):
                    interpreters.channel_recv(cid)

    def test_close_defaults_with_unused_items(self):
        if False:
            return 10
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_send(cid, b'ham')
        with self.assertRaises(interpreters.ChannelNotEmptyError):
            interpreters.channel_close(cid)
        interpreters.channel_recv(cid)
        interpreters.channel_send(cid, b'eggs')

    def test_close_recv_with_unused_items_unforced(self):
        if False:
            i = 10
            return i + 15
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_send(cid, b'ham')
        with self.assertRaises(interpreters.ChannelNotEmptyError):
            interpreters.channel_close(cid, recv=True)
        interpreters.channel_recv(cid)
        interpreters.channel_send(cid, b'eggs')
        interpreters.channel_recv(cid)
        interpreters.channel_recv(cid)
        interpreters.channel_close(cid, recv=True)

    def test_close_send_with_unused_items_unforced(self):
        if False:
            print('Hello World!')
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_send(cid, b'ham')
        interpreters.channel_close(cid, send=True)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_send(cid, b'eggs')
        interpreters.channel_recv(cid)
        interpreters.channel_recv(cid)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_recv(cid)

    def test_close_both_with_unused_items_unforced(self):
        if False:
            return 10
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_send(cid, b'ham')
        with self.assertRaises(interpreters.ChannelNotEmptyError):
            interpreters.channel_close(cid, recv=True, send=True)
        interpreters.channel_recv(cid)
        interpreters.channel_send(cid, b'eggs')
        interpreters.channel_recv(cid)
        interpreters.channel_recv(cid)
        interpreters.channel_close(cid, recv=True)

    def test_close_recv_with_unused_items_forced(self):
        if False:
            i = 10
            return i + 15
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_send(cid, b'ham')
        interpreters.channel_close(cid, recv=True, force=True)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_send(cid, b'eggs')
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_recv(cid)

    def test_close_send_with_unused_items_forced(self):
        if False:
            i = 10
            return i + 15
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_send(cid, b'ham')
        interpreters.channel_close(cid, send=True, force=True)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_send(cid, b'eggs')
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_recv(cid)

    def test_close_both_with_unused_items_forced(self):
        if False:
            while True:
                i = 10
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_send(cid, b'ham')
        interpreters.channel_close(cid, send=True, recv=True, force=True)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_send(cid, b'eggs')
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_recv(cid)

    def test_close_never_used(self):
        if False:
            print('Hello World!')
        cid = interpreters.channel_create()
        interpreters.channel_close(cid)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_send(cid, b'spam')
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_recv(cid)

    def test_close_by_unassociated_interp(self):
        if False:
            i = 10
            return i + 15
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interp = interpreters.create()
        interpreters.run_string(interp, dedent(f'\n            import _xxsubinterpreters as _interpreters\n            _interpreters.channel_close({cid}, force=True)\n            '))
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_recv(cid)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_close(cid)

    def test_close_used_multiple_times_by_single_user(self):
        if False:
            i = 10
            return i + 15
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_recv(cid)
        interpreters.channel_close(cid, force=True)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_send(cid, b'eggs')
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_recv(cid)

    def test_channel_list_interpreters_invalid_channel(self):
        if False:
            return 10
        cid = interpreters.channel_create()
        with self.assertRaises(interpreters.ChannelNotFoundError):
            interpreters.channel_list_interpreters(1000, send=True)
        interpreters.channel_close(cid)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_list_interpreters(cid, send=True)

    def test_channel_list_interpreters_invalid_args(self):
        if False:
            return 10
        cid = interpreters.channel_create()
        with self.assertRaises(TypeError):
            interpreters.channel_list_interpreters(cid)

class ChannelReleaseTests(TestBase):
    """
    - main / interp / other
    - run in: current thread / new thread / other thread / different threads
    - end / opposite
    - force / no force
    - used / not used  (associated / not associated)
    - empty / emptied / never emptied / partly emptied
    - closed / not closed
    - released / not released
    - creator (interp) / other
    - associated interpreter not running
    - associated interpreter destroyed
    """
    '\n    use\n    pre-release\n    release\n    after\n    check\n    '
    '\n    release in:         main, interp1\n    creator:            same, other (incl. interp2)\n\n    use:                None,send,recv,send/recv in None,same,other(incl. interp2),same+other(incl. interp2),all\n    pre-release:        None,send,recv,both in None,same,other(incl. interp2),same+other(incl. interp2),all\n    pre-release forced: None,send,recv,both in None,same,other(incl. interp2),same+other(incl. interp2),all\n\n    release:            same\n    release forced:     same\n\n    use after:          None,send,recv,send/recv in None,same,other(incl. interp2),same+other(incl. interp2),all\n    release after:      None,send,recv,send/recv in None,same,other(incl. interp2),same+other(incl. interp2),all\n    check released:     send/recv for same/other(incl. interp2)\n    check closed:       send/recv for same/other(incl. interp2)\n    '

    def test_single_user(self):
        if False:
            while True:
                i = 10
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_recv(cid)
        interpreters.channel_release(cid, send=True, recv=True)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_send(cid, b'eggs')
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_recv(cid)

    def test_multiple_users(self):
        if False:
            print('Hello World!')
        cid = interpreters.channel_create()
        id1 = interpreters.create()
        id2 = interpreters.create()
        interpreters.run_string(id1, dedent(f"\n            import _xxsubinterpreters as _interpreters\n            _interpreters.channel_send({cid}, b'spam')\n            "))
        out = _run_output(id2, dedent(f'\n            import _xxsubinterpreters as _interpreters\n            obj = _interpreters.channel_recv({cid})\n            _interpreters.channel_release({cid})\n            print(repr(obj))\n            '))
        interpreters.run_string(id1, dedent(f'\n            _interpreters.channel_release({cid})\n            '))
        self.assertEqual(out.strip(), "b'spam'")

    def test_no_kwargs(self):
        if False:
            i = 10
            return i + 15
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_recv(cid)
        interpreters.channel_release(cid)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_send(cid, b'eggs')
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_recv(cid)

    def test_multiple_times(self):
        if False:
            i = 10
            return i + 15
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_recv(cid)
        interpreters.channel_release(cid, send=True, recv=True)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_release(cid, send=True, recv=True)

    def test_with_unused_items(self):
        if False:
            for i in range(10):
                print('nop')
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_send(cid, b'ham')
        interpreters.channel_release(cid, send=True, recv=True)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_recv(cid)

    def test_never_used(self):
        if False:
            print('Hello World!')
        cid = interpreters.channel_create()
        interpreters.channel_release(cid)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_send(cid, b'spam')
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_recv(cid)

    def test_by_unassociated_interp(self):
        if False:
            print('Hello World!')
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interp = interpreters.create()
        interpreters.run_string(interp, dedent(f'\n            import _xxsubinterpreters as _interpreters\n            _interpreters.channel_release({cid})\n            '))
        obj = interpreters.channel_recv(cid)
        interpreters.channel_release(cid)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_send(cid, b'eggs')
        self.assertEqual(obj, b'spam')

    def test_close_if_unassociated(self):
        if False:
            print('Hello World!')
        cid = interpreters.channel_create()
        interp = interpreters.create()
        interpreters.run_string(interp, dedent(f"\n            import _xxsubinterpreters as _interpreters\n            obj = _interpreters.channel_send({cid}, b'spam')\n            _interpreters.channel_release({cid})\n            "))
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_recv(cid)

    def test_partially(self):
        if False:
            for i in range(10):
                print('nop')
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, None)
        interpreters.channel_recv(cid)
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_release(cid, send=True)
        obj = interpreters.channel_recv(cid)
        self.assertEqual(obj, b'spam')

    def test_used_multiple_times_by_single_user(self):
        if False:
            return 10
        cid = interpreters.channel_create()
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_send(cid, b'spam')
        interpreters.channel_recv(cid)
        interpreters.channel_release(cid, send=True, recv=True)
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_send(cid, b'eggs')
        with self.assertRaises(interpreters.ChannelClosedError):
            interpreters.channel_recv(cid)

class ChannelCloseFixture(namedtuple('ChannelCloseFixture', 'end interp other extra creator')):
    QUICK = False

    def __new__(cls, end, interp, other, extra, creator):
        if False:
            print('Hello World!')
        assert end in ('send', 'recv')
        if cls.QUICK:
            known = {}
        else:
            interp = Interpreter.from_raw(interp)
            other = Interpreter.from_raw(other)
            extra = Interpreter.from_raw(extra)
            known = {interp.name: interp, other.name: other, extra.name: extra}
        if not creator:
            creator = 'same'
        self = super().__new__(cls, end, interp, other, extra, creator)
        self._prepped = set()
        self._state = ChannelState()
        self._known = known
        return self

    @property
    def state(self):
        if False:
            return 10
        return self._state

    @property
    def cid(self):
        if False:
            while True:
                i = 10
        try:
            return self._cid
        except AttributeError:
            creator = self._get_interpreter(self.creator)
            self._cid = self._new_channel(creator)
            return self._cid

    def get_interpreter(self, interp):
        if False:
            i = 10
            return i + 15
        interp = self._get_interpreter(interp)
        self._prep_interpreter(interp)
        return interp

    def expect_closed_error(self, end=None):
        if False:
            while True:
                i = 10
        if end is None:
            end = self.end
        if end == 'recv' and self.state.closed == 'send':
            return False
        return bool(self.state.closed)

    def prep_interpreter(self, interp):
        if False:
            for i in range(10):
                print('nop')
        self._prep_interpreter(interp)

    def record_action(self, action, result):
        if False:
            i = 10
            return i + 15
        self._state = result

    def clean_up(self):
        if False:
            while True:
                i = 10
        clean_up_interpreters()
        clean_up_channels()

    def _new_channel(self, creator):
        if False:
            print('Hello World!')
        if creator.name == 'main':
            return interpreters.channel_create()
        else:
            ch = interpreters.channel_create()
            run_interp(creator.id, f'\n                import _xxsubinterpreters\n                cid = _xxsubinterpreters.channel_create()\n                # We purposefully send back an int to avoid tying the\n                # channel to the other interpreter.\n                _xxsubinterpreters.channel_send({ch}, int(cid))\n                del _xxsubinterpreters\n                ')
            self._cid = interpreters.channel_recv(ch)
        return self._cid

    def _get_interpreter(self, interp):
        if False:
            while True:
                i = 10
        if interp in ('same', 'interp'):
            return self.interp
        elif interp == 'other':
            return self.other
        elif interp == 'extra':
            return self.extra
        else:
            name = interp
            try:
                interp = self._known[name]
            except KeyError:
                interp = self._known[name] = Interpreter(name)
            return interp

    def _prep_interpreter(self, interp):
        if False:
            for i in range(10):
                print('nop')
        if interp.id in self._prepped:
            return
        self._prepped.add(interp.id)
        if interp.name == 'main':
            return
        run_interp(interp.id, f'\n            import _xxsubinterpreters as interpreters\n            import test.test__xxsubinterpreters as helpers\n            ChannelState = helpers.ChannelState\n            try:\n                cid\n            except NameError:\n                cid = interpreters._channel_id({self.cid})\n            ')

@unittest.skip('these tests take several hours to run')
class ExhaustiveChannelTests(TestBase):
    """
    - main / interp / other
    - run in: current thread / new thread / other thread / different threads
    - end / opposite
    - force / no force
    - used / not used  (associated / not associated)
    - empty / emptied / never emptied / partly emptied
    - closed / not closed
    - released / not released
    - creator (interp) / other
    - associated interpreter not running
    - associated interpreter destroyed

    - close after unbound
    """
    '\n    use\n    pre-close\n    close\n    after\n    check\n    '
    '\n    close in:         main, interp1\n    creator:          same, other, extra\n\n    use:              None,send,recv,send/recv in None,same,other,same+other,all\n    pre-close:        None,send,recv in None,same,other,same+other,all\n    pre-close forced: None,send,recv in None,same,other,same+other,all\n\n    close:            same\n    close forced:     same\n\n    use after:        None,send,recv,send/recv in None,same,other,extra,same+other,all\n    close after:      None,send,recv,send/recv in None,same,other,extra,same+other,all\n    check closed:     send/recv for same/other(incl. interp2)\n    '

    def iter_action_sets(self):
        if False:
            print('Hello World!')
        yield []
        for closeactions in self._iter_close_action_sets('same', 'other'):
            yield closeactions
            for postactions in self._iter_post_close_action_sets():
                yield (closeactions + postactions)
        for closeactions in self._iter_close_action_sets('other', 'extra'):
            yield closeactions
            for postactions in self._iter_post_close_action_sets():
                yield (closeactions + postactions)
        for useactions in self._iter_use_action_sets('same', 'other'):
            yield useactions
            for closeactions in self._iter_close_action_sets('same', 'other'):
                actions = useactions + closeactions
                yield actions
                for postactions in self._iter_post_close_action_sets():
                    yield (actions + postactions)
            for closeactions in self._iter_close_action_sets('other', 'extra'):
                actions = useactions + closeactions
                yield actions
                for postactions in self._iter_post_close_action_sets():
                    yield (actions + postactions)
        for useactions in self._iter_use_action_sets('other', 'extra'):
            yield useactions
            for closeactions in self._iter_close_action_sets('same', 'other'):
                actions = useactions + closeactions
                yield actions
                for postactions in self._iter_post_close_action_sets():
                    yield (actions + postactions)
            for closeactions in self._iter_close_action_sets('other', 'extra'):
                actions = useactions + closeactions
                yield actions
                for postactions in self._iter_post_close_action_sets():
                    yield (actions + postactions)

    def _iter_use_action_sets(self, interp1, interp2):
        if False:
            for i in range(10):
                print('nop')
        interps = (interp1, interp2)
        yield [ChannelAction('use', 'recv', interp1)]
        yield [ChannelAction('use', 'recv', interp2)]
        yield [ChannelAction('use', 'recv', interp1), ChannelAction('use', 'recv', interp2)]
        yield [ChannelAction('use', 'send', interp1)]
        yield [ChannelAction('use', 'send', interp2)]
        yield [ChannelAction('use', 'send', interp1), ChannelAction('use', 'send', interp2)]
        for interp1 in interps:
            for interp2 in interps:
                for interp3 in interps:
                    yield [ChannelAction('use', 'send', interp1), ChannelAction('use', 'send', interp2), ChannelAction('use', 'recv', interp3)]
        for interp1 in interps:
            for interp2 in interps:
                for interp3 in interps:
                    for interp4 in interps:
                        yield [ChannelAction('use', 'send', interp1), ChannelAction('use', 'send', interp2), ChannelAction('use', 'recv', interp3), ChannelAction('use', 'recv', interp4)]

    def _iter_close_action_sets(self, interp1, interp2):
        if False:
            while True:
                i = 10
        ends = ('recv', 'send')
        interps = (interp1, interp2)
        for force in (True, False):
            op = 'force-close' if force else 'close'
            for interp in interps:
                for end in ends:
                    yield [ChannelAction(op, end, interp)]
        for recvop in ('close', 'force-close'):
            for sendop in ('close', 'force-close'):
                for recv in interps:
                    for send in interps:
                        yield [ChannelAction(recvop, 'recv', recv), ChannelAction(sendop, 'send', send)]

    def _iter_post_close_action_sets(self):
        if False:
            i = 10
            return i + 15
        for interp in ('same', 'extra', 'other'):
            yield [ChannelAction('use', 'recv', interp)]
            yield [ChannelAction('use', 'send', interp)]

    def run_actions(self, fix, actions):
        if False:
            print('Hello World!')
        for action in actions:
            self.run_action(fix, action)

    def run_action(self, fix, action, *, hideclosed=True):
        if False:
            for i in range(10):
                print('nop')
        end = action.resolve_end(fix.end)
        interp = action.resolve_interp(fix.interp, fix.other, fix.extra)
        fix.prep_interpreter(interp)
        if interp.name == 'main':
            result = run_action(fix.cid, action.action, end, fix.state, hideclosed=hideclosed)
            fix.record_action(action, result)
        else:
            _cid = interpreters.channel_create()
            run_interp(interp.id, f"\n                result = helpers.run_action(\n                    {fix.cid},\n                    {repr(action.action)},\n                    {repr(end)},\n                    {repr(fix.state)},\n                    hideclosed={hideclosed},\n                    )\n                interpreters.channel_send({_cid}, result.pending.to_bytes(1, 'little'))\n                interpreters.channel_send({_cid}, b'X' if result.closed else b'')\n                ")
            result = ChannelState(pending=int.from_bytes(interpreters.channel_recv(_cid), 'little'), closed=bool(interpreters.channel_recv(_cid)))
            fix.record_action(action, result)

    def iter_fixtures(self):
        if False:
            while True:
                i = 10
        interpreters = [('main', 'interp', 'extra'), ('interp', 'main', 'extra'), ('interp1', 'interp2', 'extra'), ('interp1', 'interp2', 'main')]
        for (interp, other, extra) in interpreters:
            for creator in ('same', 'other', 'creator'):
                for end in ('send', 'recv'):
                    yield ChannelCloseFixture(end, interp, other, extra, creator)

    def _close(self, fix, *, force):
        if False:
            while True:
                i = 10
        op = 'force-close' if force else 'close'
        close = ChannelAction(op, fix.end, 'same')
        if not fix.expect_closed_error():
            self.run_action(fix, close, hideclosed=False)
        else:
            with self.assertRaises(interpreters.ChannelClosedError):
                self.run_action(fix, close, hideclosed=False)

    def _assert_closed_in_interp(self, fix, interp=None):
        if False:
            return 10
        if interp is None or interp.name == 'main':
            with self.assertRaises(interpreters.ChannelClosedError):
                interpreters.channel_recv(fix.cid)
            with self.assertRaises(interpreters.ChannelClosedError):
                interpreters.channel_send(fix.cid, b'spam')
            with self.assertRaises(interpreters.ChannelClosedError):
                interpreters.channel_close(fix.cid)
            with self.assertRaises(interpreters.ChannelClosedError):
                interpreters.channel_close(fix.cid, force=True)
        else:
            run_interp(interp.id, f'\n                with helpers.expect_channel_closed():\n                    interpreters.channel_recv(cid)\n                ')
            run_interp(interp.id, f"\n                with helpers.expect_channel_closed():\n                    interpreters.channel_send(cid, b'spam')\n                ")
            run_interp(interp.id, f'\n                with helpers.expect_channel_closed():\n                    interpreters.channel_close(cid)\n                ')
            run_interp(interp.id, f'\n                with helpers.expect_channel_closed():\n                    interpreters.channel_close(cid, force=True)\n                ')

    def _assert_closed(self, fix):
        if False:
            print('Hello World!')
        self.assertTrue(fix.state.closed)
        for _ in range(fix.state.pending):
            interpreters.channel_recv(fix.cid)
        self._assert_closed_in_interp(fix)
        for interp in ('same', 'other'):
            interp = fix.get_interpreter(interp)
            if interp.name == 'main':
                continue
            self._assert_closed_in_interp(fix, interp)
        interp = fix.get_interpreter('fresh')
        self._assert_closed_in_interp(fix, interp)

    def _iter_close_tests(self, verbose=False):
        if False:
            while True:
                i = 10
        i = 0
        for actions in self.iter_action_sets():
            print()
            for fix in self.iter_fixtures():
                i += 1
                if i > 1000:
                    return
                if verbose:
                    if (i - 1) % 6 == 0:
                        print()
                    print(i, fix, '({} actions)'.format(len(actions)))
                else:
                    if (i - 1) % 6 == 0:
                        print(' ', end='')
                    print('.', end='')
                    sys.stdout.flush()
                yield (i, fix, actions)
            if verbose:
                print('---')
        print()

    def _skim_close_tests(self):
        if False:
            i = 10
            return i + 15
        ChannelCloseFixture.QUICK = True
        for (i, fix, actions) in self._iter_close_tests():
            pass

    def test_close(self):
        if False:
            return 10
        for (i, fix, actions) in self._iter_close_tests():
            with self.subTest('{} {}  {}'.format(i, fix, actions)):
                fix.prep_interpreter(fix.interp)
                self.run_actions(fix, actions)
                self._close(fix, force=False)
                self._assert_closed(fix)
            fix.clean_up()

    def test_force_close(self):
        if False:
            i = 10
            return i + 15
        for (i, fix, actions) in self._iter_close_tests():
            with self.subTest('{} {}  {}'.format(i, fix, actions)):
                fix.prep_interpreter(fix.interp)
                self.run_actions(fix, actions)
                self._close(fix, force=True)
                self._assert_closed(fix)
            fix.clean_up()
if __name__ == '__main__':
    unittest.main()