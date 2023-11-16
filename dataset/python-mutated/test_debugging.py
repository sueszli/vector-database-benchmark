import os
import sys
from typing import List
import _pytest._code
import pytest
from _pytest.debugging import _validate_usepdb_cls
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import Pytester
_ENVIRON_PYTHONBREAKPOINT = os.environ.get('PYTHONBREAKPOINT', '')

@pytest.fixture(autouse=True)
def pdb_env(request):
    if False:
        while True:
            i = 10
    if 'pytester' in request.fixturenames:
        pytester = request.getfixturevalue('pytester')
        pytester._monkeypatch.setenv('PDBPP_HIJACK_PDB', '0')

def runpdb(pytester: Pytester, source: str):
    if False:
        return 10
    p = pytester.makepyfile(source)
    return pytester.runpytest_inprocess('--pdb', p)

def runpdb_and_get_stdout(pytester: Pytester, source: str):
    if False:
        i = 10
        return i + 15
    result = runpdb(pytester, source)
    return result.stdout.str()

def runpdb_and_get_report(pytester: Pytester, source: str):
    if False:
        i = 10
        return i + 15
    result = runpdb(pytester, source)
    reports = result.reprec.getreports('pytest_runtest_logreport')
    assert len(reports) == 3, reports
    return reports[1]

@pytest.fixture
def custom_pdb_calls() -> List[str]:
    if False:
        while True:
            i = 10
    called = []

    class _CustomPdb:
        quitting = False

        def __init__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            called.append('init')

        def reset(self):
            if False:
                i = 10
                return i + 15
            called.append('reset')

        def interaction(self, *args):
            if False:
                return 10
            called.append('interaction')
    _pytest._CustomPdb = _CustomPdb
    return called

@pytest.fixture
def custom_debugger_hook():
    if False:
        i = 10
        return i + 15
    called = []

    class _CustomDebugger:

        def __init__(self, *args, **kwargs):
            if False:
                return 10
            called.append('init')

        def reset(self):
            if False:
                i = 10
                return i + 15
            called.append('reset')

        def interaction(self, *args):
            if False:
                print('Hello World!')
            called.append('interaction')

        def set_trace(self, frame):
            if False:
                while True:
                    i = 10
            print('**CustomDebugger**')
            called.append('set_trace')
    _pytest._CustomDebugger = _CustomDebugger
    yield called
    del _pytest._CustomDebugger

class TestPDB:

    @pytest.fixture
    def pdblist(self, request):
        if False:
            i = 10
            return i + 15
        monkeypatch = request.getfixturevalue('monkeypatch')
        pdblist = []

        def mypdb(*args):
            if False:
                return 10
            pdblist.append(args)
        plugin = request.config.pluginmanager.getplugin('debugging')
        monkeypatch.setattr(plugin, 'post_mortem', mypdb)
        return pdblist

    def test_pdb_on_fail(self, pytester: Pytester, pdblist) -> None:
        if False:
            return 10
        rep = runpdb_and_get_report(pytester, '\n            def test_func():\n                assert 0\n        ')
        assert rep.failed
        assert len(pdblist) == 1
        tb = _pytest._code.Traceback(pdblist[0][0])
        assert tb[-1].name == 'test_func'

    def test_pdb_on_xfail(self, pytester: Pytester, pdblist) -> None:
        if False:
            return 10
        rep = runpdb_and_get_report(pytester, '\n            import pytest\n            @pytest.mark.xfail\n            def test_func():\n                assert 0\n        ')
        assert 'xfail' in rep.keywords
        assert not pdblist

    def test_pdb_on_skip(self, pytester, pdblist) -> None:
        if False:
            while True:
                i = 10
        rep = runpdb_and_get_report(pytester, '\n            import pytest\n            def test_func():\n                pytest.skip("hello")\n        ')
        assert rep.skipped
        assert len(pdblist) == 0

    def test_pdb_on_top_level_raise_skiptest(self, pytester, pdblist) -> None:
        if False:
            for i in range(10):
                print('nop')
        stdout = runpdb_and_get_stdout(pytester, '\n            import unittest\n            raise unittest.SkipTest("This is a common way to skip an entire file.")\n        ')
        assert 'entering PDB' not in stdout, stdout

    def test_pdb_on_BdbQuit(self, pytester, pdblist) -> None:
        if False:
            for i in range(10):
                print('nop')
        rep = runpdb_and_get_report(pytester, '\n            import bdb\n            def test_func():\n                raise bdb.BdbQuit\n        ')
        assert rep.failed
        assert len(pdblist) == 0

    def test_pdb_on_KeyboardInterrupt(self, pytester, pdblist) -> None:
        if False:
            return 10
        rep = runpdb_and_get_report(pytester, '\n            def test_func():\n                raise KeyboardInterrupt\n        ')
        assert rep.failed
        assert len(pdblist) == 1

    @staticmethod
    def flush(child):
        if False:
            i = 10
            return i + 15
        if child.isalive():
            child.read()
            child.wait()
        assert not child.isalive()

    def test_pdb_unittest_postmortem(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p1 = pytester.makepyfile("\n            import unittest\n            class Blub(unittest.TestCase):\n                def tearDown(self):\n                    self.filename = None\n                def test_false(self):\n                    self.filename = 'debug' + '.me'\n                    assert 0\n        ")
        child = pytester.spawn_pytest(f'--pdb {p1}')
        child.expect('Pdb')
        child.sendline('p self.filename')
        child.sendeof()
        rest = child.read().decode('utf8')
        assert 'debug.me' in rest
        self.flush(child)

    def test_pdb_unittest_skip(self, pytester: Pytester) -> None:
        if False:
            return 10
        'Test for issue #2137'
        p1 = pytester.makepyfile("\n            import unittest\n            @unittest.skipIf(True, 'Skipping also with pdb active')\n            class MyTestCase(unittest.TestCase):\n                def test_one(self):\n                    assert 0\n        ")
        child = pytester.spawn_pytest(f'-rs --pdb {p1}')
        child.expect('Skipping also with pdb active')
        child.expect_exact('= 1 skipped in')
        child.sendeof()
        self.flush(child)

    def test_pdb_print_captured_stdout_and_stderr(self, pytester: Pytester) -> None:
        if False:
            return 10
        p1 = pytester.makepyfile('\n            def test_1():\n                import sys\n                sys.stderr.write("get\\x20rekt")\n                print("get\\x20rekt")\n                assert False\n\n            def test_not_called_due_to_quit():\n                pass\n        ')
        child = pytester.spawn_pytest('--pdb %s' % p1)
        child.expect('captured stdout')
        child.expect('get rekt')
        child.expect('captured stderr')
        child.expect('get rekt')
        child.expect('traceback')
        child.expect('def test_1')
        child.expect('Pdb')
        child.sendeof()
        rest = child.read().decode('utf8')
        assert 'Exit: Quitting debugger' in rest
        assert '= 1 failed in' in rest
        assert 'def test_1' not in rest
        assert 'get rekt' not in rest
        self.flush(child)

    def test_pdb_dont_print_empty_captured_stdout_and_stderr(self, pytester: Pytester) -> None:
        if False:
            return 10
        p1 = pytester.makepyfile('\n            def test_1():\n                assert False\n        ')
        child = pytester.spawn_pytest('--pdb %s' % p1)
        child.expect('Pdb')
        output = child.before.decode('utf8')
        child.sendeof()
        assert 'captured stdout' not in output
        assert 'captured stderr' not in output
        self.flush(child)

    @pytest.mark.parametrize('showcapture', ['all', 'no', 'log'])
    def test_pdb_print_captured_logs(self, pytester, showcapture: str) -> None:
        if False:
            while True:
                i = 10
        p1 = pytester.makepyfile('\n            def test_1():\n                import logging\n                logging.warning("get " + "rekt")\n                assert False\n        ')
        child = pytester.spawn_pytest(f'--show-capture={showcapture} --pdb {p1}')
        if showcapture in ('all', 'log'):
            child.expect('captured log')
            child.expect('get rekt')
        child.expect('Pdb')
        child.sendeof()
        rest = child.read().decode('utf8')
        assert '1 failed' in rest
        self.flush(child)

    def test_pdb_print_captured_logs_nologging(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p1 = pytester.makepyfile('\n            def test_1():\n                import logging\n                logging.warning("get " + "rekt")\n                assert False\n        ')
        child = pytester.spawn_pytest('--show-capture=all --pdb -p no:logging %s' % p1)
        child.expect('get rekt')
        output = child.before.decode('utf8')
        assert 'captured log' not in output
        child.expect('Pdb')
        child.sendeof()
        rest = child.read().decode('utf8')
        assert '1 failed' in rest
        self.flush(child)

    def test_pdb_interaction_exception(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p1 = pytester.makepyfile('\n            import pytest\n            def globalfunc():\n                pass\n            def test_1():\n                pytest.raises(ValueError, globalfunc)\n        ')
        child = pytester.spawn_pytest('--pdb %s' % p1)
        child.expect('.*def test_1')
        child.expect('.*pytest.raises.*globalfunc')
        child.expect('Pdb')
        child.sendline('globalfunc')
        child.expect('.*function')
        child.sendeof()
        child.expect('1 failed')
        self.flush(child)

    def test_pdb_interaction_on_collection_issue181(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        p1 = pytester.makepyfile('\n            import pytest\n            xxx\n        ')
        child = pytester.spawn_pytest('--pdb %s' % p1)
        child.expect('Pdb')
        child.sendline('c')
        child.expect('1 error')
        self.flush(child)

    def test_pdb_interaction_on_internal_error(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makeconftest('\n            def pytest_runtest_protocol():\n                0/0\n        ')
        p1 = pytester.makepyfile('def test_func(): pass')
        child = pytester.spawn_pytest('--pdb %s' % p1)
        child.expect('Pdb')
        assert len([x for x in child.before.decode().splitlines() if x.startswith('INTERNALERROR> Traceback')]) == 1
        child.sendeof()
        self.flush(child)

    def test_pdb_prevent_ConftestImportFailure_hiding_exception(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('def test_func(): pass')
        sub_dir = pytester.path.joinpath('ns')
        sub_dir.mkdir()
        sub_dir.joinpath('conftest').with_suffix('.py').write_text('import unknown', 'utf-8')
        sub_dir.joinpath('test_file').with_suffix('.py').write_text('def test_func(): pass', 'utf-8')
        result = pytester.runpytest_subprocess('--pdb', '.')
        result.stdout.fnmatch_lines(['-> import unknown'])

    @pytest.mark.xfail(reason='#10042')
    def test_pdb_interaction_capturing_simple(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        p1 = pytester.makepyfile('\n            import pytest\n            def test_1():\n                i = 0\n                print("hello17")\n                pytest.set_trace()\n                i == 1\n                assert 0\n        ')
        child = pytester.spawn_pytest(str(p1))
        child.expect('test_1\\(\\)')
        child.expect('i == 1')
        child.expect('Pdb')
        child.sendline('c')
        rest = child.read().decode('utf-8')
        assert 'AssertionError' in rest
        assert '1 failed' in rest
        assert 'def test_1' in rest
        assert 'hello17' in rest
        self.flush(child)

    def test_pdb_set_trace_kwargs(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        p1 = pytester.makepyfile('\n            import pytest\n            def test_1():\n                i = 0\n                print("hello17")\n                pytest.set_trace(header="== my_header ==")\n                x = 3\n                assert 0\n        ')
        child = pytester.spawn_pytest(str(p1))
        child.expect('== my_header ==')
        assert 'PDB set_trace' not in child.before.decode()
        child.expect('Pdb')
        child.sendline('c')
        rest = child.read().decode('utf-8')
        assert '1 failed' in rest
        assert 'def test_1' in rest
        assert 'hello17' in rest
        self.flush(child)

    def test_pdb_set_trace_interception(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p1 = pytester.makepyfile('\n            import pdb\n            def test_1():\n                pdb.set_trace()\n        ')
        child = pytester.spawn_pytest(str(p1))
        child.expect('test_1')
        child.expect('Pdb')
        child.sendline('q')
        rest = child.read().decode('utf8')
        assert 'no tests ran' in rest
        assert 'reading from stdin while output' not in rest
        assert 'BdbQuit' not in rest
        self.flush(child)

    def test_pdb_and_capsys(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        p1 = pytester.makepyfile('\n            import pytest\n            def test_1(capsys):\n                print("hello1")\n                pytest.set_trace()\n        ')
        child = pytester.spawn_pytest(str(p1))
        child.expect('test_1')
        child.send('capsys.readouterr()\n')
        child.expect('hello1')
        child.sendeof()
        child.read()
        self.flush(child)

    def test_pdb_with_caplog_on_pdb_invocation(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        p1 = pytester.makepyfile('\n            def test_1(capsys, caplog):\n                import logging\n                logging.getLogger(__name__).warning("some_warning")\n                assert 0\n        ')
        child = pytester.spawn_pytest('--pdb %s' % str(p1))
        child.send('caplog.record_tuples\n')
        child.expect_exact("[('test_pdb_with_caplog_on_pdb_invocation', 30, 'some_warning')]")
        child.sendeof()
        child.read()
        self.flush(child)

    def test_set_trace_capturing_afterwards(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        p1 = pytester.makepyfile('\n            import pdb\n            def test_1():\n                pdb.set_trace()\n            def test_2():\n                print("hello")\n                assert 0\n        ')
        child = pytester.spawn_pytest(str(p1))
        child.expect('test_1')
        child.send('c\n')
        child.expect('test_2')
        child.expect('Captured')
        child.expect('hello')
        child.sendeof()
        child.read()
        self.flush(child)

    def test_pdb_interaction_doctest(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        p1 = pytester.makepyfile("\n            def function_1():\n                '''\n                >>> i = 0\n                >>> assert i == 1\n                '''\n        ")
        child = pytester.spawn_pytest('--doctest-modules --pdb %s' % p1)
        child.expect('Pdb')
        assert 'UNEXPECTED EXCEPTION: AssertionError()' in child.before.decode('utf8')
        child.sendline("'i=%i.' % i")
        child.expect('Pdb')
        assert "\r\n'i=0.'\r\n" in child.before.decode('utf8')
        child.sendeof()
        rest = child.read().decode('utf8')
        assert '! _pytest.outcomes.Exit: Quitting debugger !' in rest
        assert 'BdbQuit' not in rest
        assert '1 failed' in rest
        self.flush(child)

    def test_doctest_set_trace_quit(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        p1 = pytester.makepyfile("\n            def function_1():\n                '''\n                >>> __import__('pdb').set_trace()\n                '''\n        ")
        child = pytester.spawn_pytest('--doctest-modules --pdb -s %s' % p1)
        child.expect('Pdb')
        child.sendline('q')
        rest = child.read().decode('utf8')
        assert '! _pytest.outcomes.Exit: Quitting debugger !' in rest
        assert '= no tests ran in' in rest
        assert 'BdbQuit' not in rest
        assert 'UNEXPECTED EXCEPTION' not in rest

    @pytest.mark.xfail(reason='#10042')
    def test_pdb_interaction_capturing_twice(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        p1 = pytester.makepyfile('\n            import pytest\n            def test_1():\n                i = 0\n                print("hello17")\n                pytest.set_trace()\n                x = 3\n                print("hello18")\n                pytest.set_trace()\n                x = 4\n                assert 0\n        ')
        child = pytester.spawn_pytest(str(p1))
        child.expect('PDB set_trace \\(IO-capturing turned off\\)')
        child.expect('test_1')
        child.expect('x = 3')
        child.expect('Pdb')
        child.sendline('c')
        child.expect('PDB continue \\(IO-capturing resumed\\)')
        child.expect('PDB set_trace \\(IO-capturing turned off\\)')
        child.expect('x = 4')
        child.expect('Pdb')
        child.sendline('c')
        child.expect('_ test_1 _')
        child.expect('def test_1')
        rest = child.read().decode('utf8')
        assert 'Captured stdout call' in rest
        assert 'hello17' in rest
        assert 'hello18' in rest
        assert '1 failed' in rest
        self.flush(child)

    @pytest.mark.xfail(reason='#10042')
    def test_pdb_with_injected_do_debug(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        'Simulates pdbpp, which injects Pdb into do_debug, and uses\n        self.__class__ in do_continue.\n        '
        p1 = pytester.makepyfile(mytest='\n            import pdb\n            import pytest\n\n            count_continue = 0\n\n            class CustomPdb(pdb.Pdb, object):\n                def do_debug(self, arg):\n                    import sys\n                    import types\n\n                    do_debug_func = pdb.Pdb.do_debug\n\n                    newglobals = do_debug_func.__globals__.copy()\n                    newglobals[\'Pdb\'] = self.__class__\n                    orig_do_debug = types.FunctionType(\n                        do_debug_func.__code__, newglobals,\n                        do_debug_func.__name__, do_debug_func.__defaults__,\n                    )\n                    return orig_do_debug(self, arg)\n                do_debug.__doc__ = pdb.Pdb.do_debug.__doc__\n\n                def do_continue(self, *args, **kwargs):\n                    global count_continue\n                    count_continue += 1\n                    return super(CustomPdb, self).do_continue(*args, **kwargs)\n\n            def foo():\n                print("print_from_foo")\n\n            def test_1():\n                i = 0\n                print("hello17")\n                pytest.set_trace()\n                x = 3\n                print("hello18")\n\n                assert count_continue == 2, "unexpected_failure: %d != 2" % count_continue\n                pytest.fail("expected_failure")\n        ')
        child = pytester.spawn_pytest('--pdbcls=mytest:CustomPdb %s' % str(p1))
        child.expect('PDB set_trace \\(IO-capturing turned off\\)')
        child.expect('\\n\\(Pdb')
        child.sendline('debug foo()')
        child.expect('ENTERING RECURSIVE DEBUGGER')
        child.expect('\\n\\(\\(Pdb')
        child.sendline('c')
        child.expect('LEAVING RECURSIVE DEBUGGER')
        assert b'PDB continue' not in child.before
        assert child.before.endswith(b'c\r\nprint_from_foo\r\n')
        child.sendline('debug 42')
        child.sendline('q')
        child.expect('LEAVING RECURSIVE DEBUGGER')
        assert b'ENTERING RECURSIVE DEBUGGER' in child.before
        assert b'Quitting debugger' not in child.before
        child.sendline('c')
        child.expect('PDB continue \\(IO-capturing resumed\\)')
        rest = child.read().decode('utf8')
        assert 'hello17' in rest
        assert 'hello18' in rest
        assert '1 failed' in rest
        assert 'Failed: expected_failure' in rest
        assert 'AssertionError: unexpected_failure' not in rest
        self.flush(child)

    def test_pdb_without_capture(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        p1 = pytester.makepyfile('\n            import pytest\n            def test_1():\n                pytest.set_trace()\n        ')
        child = pytester.spawn_pytest('-s %s' % p1)
        child.expect('>>> PDB set_trace >>>')
        child.expect('Pdb')
        child.sendline('c')
        child.expect('>>> PDB continue >>>')
        child.expect('1 passed')
        self.flush(child)

    @pytest.mark.parametrize('capture_arg', ('', '-s', '-p no:capture'))
    def test_pdb_continue_with_recursive_debug(self, capture_arg, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        'Full coverage for do_debug without capturing.\n\n        This is very similar to test_pdb_interaction_continue_recursive in general,\n        but mocks out ``pdb.set_trace`` for providing more coverage.\n        '
        p1 = pytester.makepyfile('\n            try:\n                input = raw_input\n            except NameError:\n                pass\n\n            def set_trace():\n                __import__(\'pdb\').set_trace()\n\n            def test_1(monkeypatch):\n                import _pytest.debugging\n\n                class pytestPDBTest(_pytest.debugging.pytestPDB):\n                    @classmethod\n                    def set_trace(cls, *args, **kwargs):\n                        # Init PytestPdbWrapper to handle capturing.\n                        _pdb = cls._init_pdb("set_trace", *args, **kwargs)\n\n                        # Mock out pdb.Pdb.do_continue.\n                        import pdb\n                        pdb.Pdb.do_continue = lambda self, arg: None\n\n                        print("===" + " SET_TRACE ===")\n                        assert input() == "debug set_trace()"\n\n                        # Simulate PytestPdbWrapper.do_debug\n                        cls._recursive_debug += 1\n                        print("ENTERING RECURSIVE DEBUGGER")\n                        print("===" + " SET_TRACE_2 ===")\n\n                        assert input() == "c"\n                        _pdb.do_continue("")\n                        print("===" + " SET_TRACE_3 ===")\n\n                        # Simulate PytestPdbWrapper.do_debug\n                        print("LEAVING RECURSIVE DEBUGGER")\n                        cls._recursive_debug -= 1\n\n                        print("===" + " SET_TRACE_4 ===")\n                        assert input() == "c"\n                        _pdb.do_continue("")\n\n                    def do_continue(self, arg):\n                        print("=== do_continue")\n\n                monkeypatch.setattr(_pytest.debugging, "pytestPDB", pytestPDBTest)\n\n                import pdb\n                monkeypatch.setattr(pdb, "set_trace", pytestPDBTest.set_trace)\n\n                set_trace()\n        ')
        child = pytester.spawn_pytest(f'--tb=short {p1} {capture_arg}')
        child.expect('=== SET_TRACE ===')
        before = child.before.decode('utf8')
        if not capture_arg:
            assert '>>> PDB set_trace (IO-capturing turned off) >>>' in before
        else:
            assert '>>> PDB set_trace >>>' in before
        child.sendline('debug set_trace()')
        child.expect('=== SET_TRACE_2 ===')
        before = child.before.decode('utf8')
        assert '\r\nENTERING RECURSIVE DEBUGGER\r\n' in before
        child.sendline('c')
        child.expect('=== SET_TRACE_3 ===')
        before = child.before.decode('utf8')
        assert '>>> PDB continue ' not in before
        child.sendline('c')
        child.expect('=== SET_TRACE_4 ===')
        before = child.before.decode('utf8')
        assert '\r\nLEAVING RECURSIVE DEBUGGER\r\n' in before
        child.sendline('c')
        rest = child.read().decode('utf8')
        if not capture_arg:
            assert '> PDB continue (IO-capturing resumed) >' in rest
        else:
            assert '> PDB continue >' in rest
        assert '= 1 passed in' in rest

    def test_pdb_used_outside_test(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        p1 = pytester.makepyfile('\n            import pytest\n            pytest.set_trace()\n            x = 5\n        ')
        child = pytester.spawn(f'{sys.executable} {p1}')
        child.expect('x = 5')
        child.expect('Pdb')
        child.sendeof()
        self.flush(child)

    def test_pdb_used_in_generate_tests(self, pytester: Pytester) -> None:
        if False:
            return 10
        p1 = pytester.makepyfile('\n            import pytest\n            def pytest_generate_tests(metafunc):\n                pytest.set_trace()\n                x = 5\n            def test_foo(a):\n                pass\n        ')
        child = pytester.spawn_pytest(str(p1))
        child.expect('x = 5')
        child.expect('Pdb')
        child.sendeof()
        self.flush(child)

    def test_pdb_collection_failure_is_shown(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        p1 = pytester.makepyfile('xxx')
        result = pytester.runpytest_subprocess('--pdb', p1)
        result.stdout.fnmatch_lines(['E   NameError: *xxx*', '*! *Exit: Quitting debugger !*'])

    @pytest.mark.parametrize('post_mortem', (False, True))
    def test_enter_leave_pdb_hooks_are_called(self, post_mortem, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makeconftest('\n            mypdb = None\n\n            def pytest_configure(config):\n                config.testing_verification = \'configured\'\n\n            def pytest_enter_pdb(config, pdb):\n                assert config.testing_verification == \'configured\'\n                print(\'enter_pdb_hook\')\n\n                global mypdb\n                mypdb = pdb\n                mypdb.set_attribute = "bar"\n\n            def pytest_leave_pdb(config, pdb):\n                assert config.testing_verification == \'configured\'\n                print(\'leave_pdb_hook\')\n\n                global mypdb\n                assert mypdb is pdb\n                assert mypdb.set_attribute == "bar"\n        ')
        p1 = pytester.makepyfile('\n            import pytest\n\n            def test_set_trace():\n                pytest.set_trace()\n                assert 0\n\n            def test_post_mortem():\n                assert 0\n        ')
        if post_mortem:
            child = pytester.spawn_pytest(str(p1) + ' --pdb -s -k test_post_mortem')
        else:
            child = pytester.spawn_pytest(str(p1) + ' -k test_set_trace')
        child.expect('enter_pdb_hook')
        child.sendline('c')
        if post_mortem:
            child.expect('PDB continue')
        else:
            child.expect('PDB continue \\(IO-capturing resumed\\)')
            child.expect('Captured stdout call')
        rest = child.read().decode('utf8')
        assert 'leave_pdb_hook' in rest
        assert '1 failed' in rest
        self.flush(child)

    def test_pdb_custom_cls(self, pytester: Pytester, custom_pdb_calls: List[str]) -> None:
        if False:
            print('Hello World!')
        p1 = pytester.makepyfile('xxx ')
        result = pytester.runpytest_inprocess('--pdb', '--pdbcls=_pytest:_CustomPdb', p1)
        result.stdout.fnmatch_lines(['*NameError*xxx*', '*1 error*'])
        assert custom_pdb_calls == ['init', 'reset', 'interaction']

    def test_pdb_custom_cls_invalid(self, pytester: Pytester) -> None:
        if False:
            return 10
        result = pytester.runpytest_inprocess('--pdbcls=invalid')
        result.stderr.fnmatch_lines(["*: error: argument --pdbcls: 'invalid' is not in the format 'modname:classname'"])

    def test_pdb_validate_usepdb_cls(self):
        if False:
            print('Hello World!')
        assert _validate_usepdb_cls('os.path:dirname.__name__') == ('os.path', 'dirname.__name__')
        assert _validate_usepdb_cls('pdb:DoesNotExist') == ('pdb', 'DoesNotExist')

    def test_pdb_custom_cls_without_pdb(self, pytester: Pytester, custom_pdb_calls: List[str]) -> None:
        if False:
            while True:
                i = 10
        p1 = pytester.makepyfile('xxx ')
        result = pytester.runpytest_inprocess('--pdbcls=_pytest:_CustomPdb', p1)
        result.stdout.fnmatch_lines(['*NameError*xxx*', '*1 error*'])
        assert custom_pdb_calls == []

    def test_pdb_custom_cls_with_set_trace(self, pytester: Pytester, monkeypatch: MonkeyPatch) -> None:
        if False:
            return 10
        pytester.makepyfile(custom_pdb='\n            class CustomPdb(object):\n                def __init__(self, *args, **kwargs):\n                    skip = kwargs.pop("skip")\n                    assert skip == ["foo.*"]\n                    print("__init__")\n                    super(CustomPdb, self).__init__(*args, **kwargs)\n\n                def set_trace(*args, **kwargs):\n                    print(\'custom set_trace>\')\n         ')
        p1 = pytester.makepyfile("\n            import pytest\n\n            def test_foo():\n                pytest.set_trace(skip=['foo.*'])\n        ")
        monkeypatch.setenv('PYTHONPATH', str(pytester.path))
        child = pytester.spawn_pytest('--pdbcls=custom_pdb:CustomPdb %s' % str(p1))
        child.expect('__init__')
        child.expect('custom set_trace>')
        self.flush(child)

class TestDebuggingBreakpoints:

    @pytest.mark.parametrize('arg', ['--pdb', ''])
    def test_sys_breakpointhook_configure_and_unconfigure(self, pytester: Pytester, arg: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that sys.breakpointhook is set to the custom Pdb class once configured, test that\n        hook is reset to system value once pytest has been unconfigured\n        '
        pytester.makeconftest('\n            import sys\n            from pytest import hookimpl\n            from _pytest.debugging import pytestPDB\n\n            def pytest_configure(config):\n                config.add_cleanup(check_restored)\n\n            def check_restored():\n                assert sys.breakpointhook == sys.__breakpointhook__\n\n            def test_check():\n                assert sys.breakpointhook == pytestPDB.set_trace\n        ')
        pytester.makepyfile('\n            def test_nothing(): pass\n        ')
        args = (arg,) if arg else ()
        result = pytester.runpytest_subprocess(*args)
        result.stdout.fnmatch_lines(['*1 passed in *'])

    def test_pdb_custom_cls(self, pytester: Pytester, custom_debugger_hook) -> None:
        if False:
            while True:
                i = 10
        p1 = pytester.makepyfile('\n            def test_nothing():\n                breakpoint()\n        ')
        result = pytester.runpytest_inprocess('--pdb', '--pdbcls=_pytest:_CustomDebugger', p1)
        result.stdout.fnmatch_lines(['*CustomDebugger*', '*1 passed*'])
        assert custom_debugger_hook == ['init', 'set_trace']

    @pytest.mark.parametrize('arg', ['--pdb', ''])
    def test_environ_custom_class(self, pytester: Pytester, custom_debugger_hook, arg: str) -> None:
        if False:
            print('Hello World!')
        pytester.makeconftest("\n            import os\n            import sys\n\n            os.environ['PYTHONBREAKPOINT'] = '_pytest._CustomDebugger.set_trace'\n\n            def pytest_configure(config):\n                config.add_cleanup(check_restored)\n\n            def check_restored():\n                assert sys.breakpointhook == sys.__breakpointhook__\n\n            def test_check():\n                import _pytest\n                assert sys.breakpointhook is _pytest._CustomDebugger.set_trace\n        ")
        pytester.makepyfile('\n            def test_nothing(): pass\n        ')
        args = (arg,) if arg else ()
        result = pytester.runpytest_subprocess(*args)
        result.stdout.fnmatch_lines(['*1 passed in *'])

    @pytest.mark.skipif(not _ENVIRON_PYTHONBREAKPOINT == '', reason='Requires breakpoint() default value')
    def test_sys_breakpoint_interception(self, pytester: Pytester) -> None:
        if False:
            return 10
        p1 = pytester.makepyfile('\n            def test_1():\n                breakpoint()\n        ')
        child = pytester.spawn_pytest(str(p1))
        child.expect('test_1')
        child.expect('Pdb')
        child.sendline('quit')
        rest = child.read().decode('utf8')
        assert 'Quitting debugger' in rest
        assert 'reading from stdin while output' not in rest
        TestPDB.flush(child)

    @pytest.mark.xfail(reason='#10042')
    def test_pdb_not_altered(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p1 = pytester.makepyfile('\n            import pdb\n            def test_1():\n                pdb.set_trace()\n                assert 0\n        ')
        child = pytester.spawn_pytest(str(p1))
        child.expect('test_1')
        child.expect('Pdb')
        child.sendline('c')
        rest = child.read().decode('utf8')
        assert '1 failed' in rest
        assert 'reading from stdin while output' not in rest
        TestPDB.flush(child)

class TestTraceOption:

    def test_trace_sets_breakpoint(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p1 = pytester.makepyfile('\n            def test_1():\n                assert True\n\n            def test_2():\n                pass\n\n            def test_3():\n                pass\n            ')
        child = pytester.spawn_pytest('--trace ' + str(p1))
        child.expect('test_1')
        child.expect('Pdb')
        child.sendline('c')
        child.expect('test_2')
        child.expect('Pdb')
        child.sendline('c')
        child.expect('test_3')
        child.expect('Pdb')
        child.sendline('q')
        child.expect_exact('Exit: Quitting debugger')
        rest = child.read().decode('utf8')
        assert '= 2 passed in' in rest
        assert 'reading from stdin while output' not in rest
        assert 'Exit: Quitting debugger' not in child.before.decode('utf8')
        TestPDB.flush(child)

    def test_trace_with_parametrize_handles_shared_fixtureinfo(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        p1 = pytester.makepyfile('\n            import pytest\n            @pytest.mark.parametrize(\'myparam\', [1,2])\n            def test_1(myparam, request):\n                assert myparam in (1, 2)\n                assert request.function.__name__ == "test_1"\n            @pytest.mark.parametrize(\'func\', [1,2])\n            def test_func(func, request):\n                assert func in (1, 2)\n                assert request.function.__name__ == "test_func"\n            @pytest.mark.parametrize(\'myparam\', [1,2])\n            def test_func_kw(myparam, request, func="func_kw"):\n                assert myparam in (1, 2)\n                assert func == "func_kw"\n                assert request.function.__name__ == "test_func_kw"\n            ')
        child = pytester.spawn_pytest('--trace ' + str(p1))
        for (func, argname) in [('test_1', 'myparam'), ('test_func', 'func'), ('test_func_kw', 'myparam')]:
            child.expect_exact('> PDB runcall (IO-capturing turned off) >')
            child.expect_exact(func)
            child.expect_exact('Pdb')
            child.sendline('args')
            child.expect_exact(f'{argname} = 1\r\n')
            child.expect_exact('Pdb')
            child.sendline('c')
            child.expect_exact('Pdb')
            child.sendline('args')
            child.expect_exact(f'{argname} = 2\r\n')
            child.expect_exact('Pdb')
            child.sendline('c')
            child.expect_exact('> PDB continue (IO-capturing resumed) >')
        rest = child.read().decode('utf8')
        assert '= 6 passed in' in rest
        assert 'reading from stdin while output' not in rest
        assert 'Exit: Quitting debugger' not in child.before.decode('utf8')
        TestPDB.flush(child)

def test_trace_after_runpytest(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    "Test that debugging's pytest_configure is re-entrant."
    p1 = pytester.makepyfile('\n        from _pytest.debugging import pytestPDB\n\n        def test_outer(pytester) -> None:\n            assert len(pytestPDB._saved) == 1\n\n            pytester.makepyfile(\n                """\n                from _pytest.debugging import pytestPDB\n\n                def test_inner():\n                    assert len(pytestPDB._saved) == 2\n                    print()\n                    print("test_inner_" + "end")\n                """\n            )\n\n            result = pytester.runpytest("-s", "-k", "test_inner")\n            assert result.ret == 0\n\n            assert len(pytestPDB._saved) == 1\n    ')
    result = pytester.runpytest_subprocess('-s', '-p', 'pytester', str(p1))
    result.stdout.fnmatch_lines(['test_inner_end'])
    assert result.ret == 0

def test_quit_with_swallowed_SystemExit(pytester: Pytester) -> None:
    if False:
        return 10
    "Test that debugging's pytest_configure is re-entrant."
    p1 = pytester.makepyfile("\n        def call_pdb_set_trace():\n            __import__('pdb').set_trace()\n\n\n        def test_1():\n            try:\n                call_pdb_set_trace()\n            except SystemExit:\n                pass\n\n\n        def test_2():\n            pass\n    ")
    child = pytester.spawn_pytest(str(p1))
    child.expect('Pdb')
    child.sendline('q')
    child.expect_exact('Exit: Quitting debugger')
    rest = child.read().decode('utf8')
    assert 'no tests ran' in rest
    TestPDB.flush(child)

@pytest.mark.parametrize('fixture', ('capfd', 'capsys'))
@pytest.mark.xfail(reason='#10042')
def test_pdb_suspends_fixture_capturing(pytester: Pytester, fixture: str) -> None:
    if False:
        i = 10
        return i + 15
    'Using "-s" with pytest should suspend/resume fixture capturing.'
    p1 = pytester.makepyfile('\n        def test_inner({fixture}):\n            import sys\n\n            print("out_inner_before")\n            sys.stderr.write("err_inner_before\\n")\n\n            __import__("pdb").set_trace()\n\n            print("out_inner_after")\n            sys.stderr.write("err_inner_after\\n")\n\n            out, err = {fixture}.readouterr()\n            assert out =="out_inner_before\\nout_inner_after\\n"\n            assert err =="err_inner_before\\nerr_inner_after\\n"\n        '.format(fixture=fixture))
    child = pytester.spawn_pytest(str(p1) + ' -s')
    child.expect('Pdb')
    before = child.before.decode('utf8')
    assert '> PDB set_trace (IO-capturing turned off for fixture %s) >' % fixture in before
    child.sendline('p 40 + 2')
    child.expect('Pdb')
    assert '\r\n42\r\n' in child.before.decode('utf8')
    child.sendline('c')
    rest = child.read().decode('utf8')
    assert 'out_inner' not in rest
    assert 'err_inner' not in rest
    TestPDB.flush(child)
    assert child.exitstatus == 0
    assert '= 1 passed in' in rest
    assert '> PDB continue (IO-capturing resumed for fixture %s) >' % fixture in rest

def test_pdbcls_via_local_module(pytester: Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    'It should be imported in pytest_configure or later only.'
    p1 = pytester.makepyfile('\n        def test():\n            print("before_set_trace")\n            __import__("pdb").set_trace()\n        ', mypdb='\n        class Wrapped:\n            class MyPdb:\n                def set_trace(self, *args):\n                    print("set_trace_called", args)\n\n                def runcall(self, *args, **kwds):\n                    print("runcall_called", args, kwds)\n        ')
    result = pytester.runpytest(str(p1), '--pdbcls=really.invalid:Value', syspathinsert=True)
    result.stdout.fnmatch_lines(['*= FAILURES =*', "E * --pdbcls: could not import 'really.invalid:Value': No module named *really*"])
    assert result.ret == 1
    result = pytester.runpytest(str(p1), '--pdbcls=mypdb:Wrapped.MyPdb', syspathinsert=True)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*set_trace_called*', '* 1 passed in *'])
    result = pytester.runpytest(str(p1), '--pdbcls=mypdb:Wrapped.MyPdb', '--trace', syspathinsert=True)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*runcall_called*', '* 1 passed in *'])

def test_raises_bdbquit_with_eoferror(pytester: Pytester) -> None:
    if False:
        return 10
    "It is not guaranteed that DontReadFromInput's read is called."
    p1 = pytester.makepyfile('\n        def input_without_read(*args, **kwargs):\n            raise EOFError()\n\n        def test(monkeypatch):\n            import builtins\n            monkeypatch.setattr(builtins, "input", input_without_read)\n            __import__(\'pdb\').set_trace()\n        ')
    result = pytester.runpytest(str(p1))
    result.stdout.fnmatch_lines(['E *BdbQuit', '*= 1 failed in*'])
    assert result.ret == 1

def test_pdb_wrapper_class_is_reused(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    p1 = pytester.makepyfile('\n        def test():\n            __import__("pdb").set_trace()\n            __import__("pdb").set_trace()\n\n            import mypdb\n            instances = mypdb.instances\n            assert len(instances) == 2\n            assert instances[0].__class__ is instances[1].__class__\n        ', mypdb='\n        instances = []\n\n        class MyPdb:\n            def __init__(self, *args, **kwargs):\n                instances.append(self)\n\n            def set_trace(self, *args):\n                print("set_trace_called", args)\n        ')
    result = pytester.runpytest(str(p1), '--pdbcls=mypdb:MyPdb', syspathinsert=True)
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*set_trace_called*', '*set_trace_called*', '* 1 passed in *'])