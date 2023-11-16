"""Discover and run std-library "unittest" style tests."""
import sys
import traceback
import types
from typing import Any
from typing import Callable
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import _pytest._code
import pytest
from _pytest.compat import getimfunc
from _pytest.compat import is_async_function
from _pytest.config import hookimpl
from _pytest.fixtures import FixtureRequest
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import exit
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.outcomes import xfail
from _pytest.python import Class
from _pytest.python import Function
from _pytest.python import Module
from _pytest.runner import CallInfo
from _pytest.scope import Scope
if TYPE_CHECKING:
    import unittest
    import twisted.trial.unittest
    _SysExcInfoType = Union[Tuple[Type[BaseException], BaseException, types.TracebackType], Tuple[None, None, None]]

def pytest_pycollect_makeitem(collector: Union[Module, Class], name: str, obj: object) -> Optional['UnitTestCase']:
    if False:
        return 10
    try:
        ut = sys.modules['unittest']
        if not issubclass(obj, ut.TestCase):
            return None
    except Exception:
        return None
    item: UnitTestCase = UnitTestCase.from_parent(collector, name=name, obj=obj)
    return item

class UnitTestCase(Class):
    nofuncargs = True

    def collect(self) -> Iterable[Union[Item, Collector]]:
        if False:
            for i in range(10):
                print('nop')
        from unittest import TestLoader
        cls = self.obj
        if not getattr(cls, '__test__', True):
            return
        skipped = _is_skipped(cls)
        if not skipped:
            self._inject_setup_teardown_fixtures(cls)
            self._inject_setup_class_fixture()
        self.session._fixturemanager.parsefactories(self, unittest=True)
        loader = TestLoader()
        foundsomething = False
        for name in loader.getTestCaseNames(self.obj):
            x = getattr(self.obj, name)
            if not getattr(x, '__test__', True):
                continue
            funcobj = getimfunc(x)
            yield TestCaseFunction.from_parent(self, name=name, callobj=funcobj)
            foundsomething = True
        if not foundsomething:
            runtest = getattr(self.obj, 'runTest', None)
            if runtest is not None:
                ut = sys.modules.get('twisted.trial.unittest', None)
                if ut is None or runtest != ut.TestCase.runTest:
                    yield TestCaseFunction.from_parent(self, name='runTest')

    def _inject_setup_teardown_fixtures(self, cls: type) -> None:
        if False:
            while True:
                i = 10
        'Injects a hidden auto-use fixture to invoke setUpClass/setup_method and corresponding\n        teardown functions (#517).'
        class_fixture = _make_xunit_fixture(cls, 'setUpClass', 'tearDownClass', 'doClassCleanups', scope=Scope.Class, pass_self=False)
        if class_fixture:
            cls.__pytest_class_setup = class_fixture
        method_fixture = _make_xunit_fixture(cls, 'setup_method', 'teardown_method', None, scope=Scope.Function, pass_self=True)
        if method_fixture:
            cls.__pytest_method_setup = method_fixture

def _make_xunit_fixture(obj: type, setup_name: str, teardown_name: str, cleanup_name: Optional[str], scope: Scope, pass_self: bool):
    if False:
        print('Hello World!')
    setup = getattr(obj, setup_name, None)
    teardown = getattr(obj, teardown_name, None)
    if setup is None and teardown is None:
        return None
    if cleanup_name:
        cleanup = getattr(obj, cleanup_name, lambda *args: None)
    else:

        def cleanup(*args):
            if False:
                for i in range(10):
                    print('nop')
            pass

    @pytest.fixture(scope=scope.value, autouse=True, name=f'_unittest_{setup_name}_fixture_{obj.__qualname__}')
    def fixture(self, request: FixtureRequest) -> Generator[None, None, None]:
        if False:
            return 10
        if _is_skipped(self):
            reason = self.__unittest_skip_why__
            raise pytest.skip.Exception(reason, _use_item_location=True)
        if setup is not None:
            try:
                if pass_self:
                    setup(self, request.function)
                else:
                    setup()
            except Exception:
                if pass_self:
                    cleanup(self)
                else:
                    cleanup()
                raise
        yield
        try:
            if teardown is not None:
                if pass_self:
                    teardown(self, request.function)
                else:
                    teardown()
        finally:
            if pass_self:
                cleanup(self)
            else:
                cleanup()
    return fixture

class TestCaseFunction(Function):
    nofuncargs = True
    _excinfo: Optional[List[_pytest._code.ExceptionInfo[BaseException]]] = None
    _testcase: Optional['unittest.TestCase'] = None

    def _getobj(self):
        if False:
            print('Hello World!')
        assert self.parent is not None
        return getattr(self.parent.obj, self.originalname)

    def setup(self) -> None:
        if False:
            while True:
                i = 10
        self._explicit_tearDown: Optional[Callable[[], None]] = None
        assert self.parent is not None
        self._testcase = self.parent.obj(self.name)
        self._obj = getattr(self._testcase, self.name)
        super().setup()

    def teardown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().teardown()
        if self._explicit_tearDown is not None:
            self._explicit_tearDown()
            self._explicit_tearDown = None
        self._testcase = None
        self._obj = None

    def startTest(self, testcase: 'unittest.TestCase') -> None:
        if False:
            i = 10
            return i + 15
        pass

    def _addexcinfo(self, rawexcinfo: '_SysExcInfoType') -> None:
        if False:
            return 10
        rawexcinfo = getattr(rawexcinfo, '_rawexcinfo', rawexcinfo)
        try:
            excinfo = _pytest._code.ExceptionInfo[BaseException].from_exc_info(rawexcinfo)
            excinfo.value
            excinfo.traceback
        except TypeError:
            try:
                try:
                    values = traceback.format_exception(*rawexcinfo)
                    values.insert(0, 'NOTE: Incompatible Exception Representation, displaying natively:\n\n')
                    fail(''.join(values), pytrace=False)
                except (fail.Exception, KeyboardInterrupt):
                    raise
                except BaseException:
                    fail('ERROR: Unknown Incompatible Exception representation:\n%r' % (rawexcinfo,), pytrace=False)
            except KeyboardInterrupt:
                raise
            except fail.Exception:
                excinfo = _pytest._code.ExceptionInfo.from_current()
        self.__dict__.setdefault('_excinfo', []).append(excinfo)

    def addError(self, testcase: 'unittest.TestCase', rawexcinfo: '_SysExcInfoType') -> None:
        if False:
            while True:
                i = 10
        try:
            if isinstance(rawexcinfo[1], exit.Exception):
                exit(rawexcinfo[1].msg)
        except TypeError:
            pass
        self._addexcinfo(rawexcinfo)

    def addFailure(self, testcase: 'unittest.TestCase', rawexcinfo: '_SysExcInfoType') -> None:
        if False:
            while True:
                i = 10
        self._addexcinfo(rawexcinfo)

    def addSkip(self, testcase: 'unittest.TestCase', reason: str) -> None:
        if False:
            while True:
                i = 10
        try:
            raise pytest.skip.Exception(reason, _use_item_location=True)
        except skip.Exception:
            self._addexcinfo(sys.exc_info())

    def addExpectedFailure(self, testcase: 'unittest.TestCase', rawexcinfo: '_SysExcInfoType', reason: str='') -> None:
        if False:
            print('Hello World!')
        try:
            xfail(str(reason))
        except xfail.Exception:
            self._addexcinfo(sys.exc_info())

    def addUnexpectedSuccess(self, testcase: 'unittest.TestCase', reason: Optional['twisted.trial.unittest.Todo']=None) -> None:
        if False:
            print('Hello World!')
        msg = 'Unexpected success'
        if reason:
            msg += f': {reason.reason}'
        try:
            fail(msg, pytrace=False)
        except fail.Exception:
            self._addexcinfo(sys.exc_info())

    def addSuccess(self, testcase: 'unittest.TestCase') -> None:
        if False:
            i = 10
            return i + 15
        pass

    def stopTest(self, testcase: 'unittest.TestCase') -> None:
        if False:
            i = 10
            return i + 15
        pass

    def addDuration(self, testcase: 'unittest.TestCase', elapsed: float) -> None:
        if False:
            return 10
        pass

    def runtest(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        from _pytest.debugging import maybe_wrap_pytest_function_for_tracing
        assert self._testcase is not None
        maybe_wrap_pytest_function_for_tracing(self)
        if is_async_function(self.obj):
            self._testcase(result=self)
        else:
            assert isinstance(self.parent, UnitTestCase)
            skipped = _is_skipped(self.obj) or _is_skipped(self.parent.obj)
            if self.config.getoption('usepdb') and (not skipped):
                self._explicit_tearDown = self._testcase.tearDown
                setattr(self._testcase, 'tearDown', lambda *args: None)
            setattr(self._testcase, self.name, self.obj)
            try:
                self._testcase(result=self)
            finally:
                delattr(self._testcase, self.name)

    def _traceback_filter(self, excinfo: _pytest._code.ExceptionInfo[BaseException]) -> _pytest._code.Traceback:
        if False:
            print('Hello World!')
        traceback = super()._traceback_filter(excinfo)
        ntraceback = traceback.filter(lambda x: not x.frame.f_globals.get('__unittest'))
        if not ntraceback:
            ntraceback = traceback
        return ntraceback

@hookimpl(tryfirst=True)
def pytest_runtest_makereport(item: Item, call: CallInfo[None]) -> None:
    if False:
        print('Hello World!')
    if isinstance(item, TestCaseFunction):
        if item._excinfo:
            call.excinfo = item._excinfo.pop(0)
            try:
                del call.result
            except AttributeError:
                pass
    unittest = sys.modules.get('unittest')
    if unittest and call.excinfo and isinstance(call.excinfo.value, unittest.SkipTest):
        excinfo = call.excinfo
        call2 = CallInfo[None].from_call(lambda : pytest.skip(str(excinfo.value)), call.when)
        call.excinfo = call2.excinfo

@hookimpl(wrapper=True)
def pytest_runtest_protocol(item: Item) -> Generator[None, object, object]:
    if False:
        while True:
            i = 10
    if isinstance(item, TestCaseFunction) and 'twisted.trial.unittest' in sys.modules:
        ut: Any = sys.modules['twisted.python.failure']
        Failure__init__ = ut.Failure.__init__
        check_testcase_implements_trial_reporter()

        def excstore(self, exc_value=None, exc_type=None, exc_tb=None, captureVars=None):
            if False:
                print('Hello World!')
            if exc_value is None:
                self._rawexcinfo = sys.exc_info()
            else:
                if exc_type is None:
                    exc_type = type(exc_value)
                self._rawexcinfo = (exc_type, exc_value, exc_tb)
            try:
                Failure__init__(self, exc_value, exc_type, exc_tb, captureVars=captureVars)
            except TypeError:
                Failure__init__(self, exc_value, exc_type, exc_tb)
        ut.Failure.__init__ = excstore
        try:
            res = (yield)
        finally:
            ut.Failure.__init__ = Failure__init__
    else:
        res = (yield)
    return res

def check_testcase_implements_trial_reporter(done: List[int]=[]) -> None:
    if False:
        i = 10
        return i + 15
    if done:
        return
    from zope.interface import classImplements
    from twisted.trial.itrial import IReporter
    classImplements(TestCaseFunction, IReporter)
    done.append(1)

def _is_skipped(obj) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Return True if the given object has been marked with @unittest.skip.'
    return bool(getattr(obj, '__unittest_skip__', False))