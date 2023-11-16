from __future__ import print_function
import sys
import struct
from gevent.testing import sysinfo

class Condition(object):
    __slots__ = ()

    def __and__(self, other):
        if False:
            print('Hello World!')
        return AndCondition(self, other)

    def __or__(self, other):
        if False:
            while True:
                i = 10
        return OrCondition(self, other)

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

class AbstractBinaryCondition(Condition):
    __slots__ = ('lhs', 'rhs')
    OP = None

    def __init__(self, lhs, rhs):
        if False:
            for i in range(10):
                print('nop')
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '(%r %s %r)' % (self.lhs, self.OP, self.rhs)

class OrCondition(AbstractBinaryCondition):
    __slots__ = ()
    OP = '|'

    def __bool__(self):
        if False:
            return 10
        return bool(self.lhs) or bool(self.rhs)

class AndCondition(AbstractBinaryCondition):
    __slots__ = ()
    OP = '&'

    def __bool__(self):
        if False:
            print('Hello World!')
        return bool(self.lhs) and bool(self.rhs)

class ConstantCondition(Condition):
    __slots__ = ('value', '__name__')

    def __init__(self, value, name=None):
        if False:
            return 10
        self.value = bool(value)
        self.__name__ = name or str(value)

    def __bool__(self):
        if False:
            print('Hello World!')
        return self.value

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.__name__
ALWAYS = ConstantCondition(True)
NEVER = ConstantCondition(False)

class _AttrCondition(ConstantCondition):
    __slots__ = ()

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        ConstantCondition.__init__(self, getattr(sysinfo, name), name)
PYPY = _AttrCondition('PYPY')
PYPY3 = _AttrCondition('PYPY3')
PY3 = _AttrCondition('PY3')
PY2 = _AttrCondition('PY2')
OSX = _AttrCondition('OSX')
LIBUV = _AttrCondition('LIBUV')
WIN = _AttrCondition('WIN')
APPVEYOR = _AttrCondition('RUNNING_ON_APPVEYOR')
TRAVIS = _AttrCondition('RUNNING_ON_TRAVIS')
CI = _AttrCondition('RUNNING_ON_CI')
LEAKTEST = _AttrCondition('RUN_LEAKCHECKS')
COVERAGE = _AttrCondition('RUN_COVERAGE')
RESOLVER_NOT_SYSTEM = _AttrCondition('RESOLVER_NOT_SYSTEM')
BIT_64 = ConstantCondition(struct.calcsize('P') * 8 == 64, 'BIT_64')
PY380_EXACTLY = ConstantCondition(sys.version_info[:3] == (3, 8, 0), 'PY380_EXACTLY')
PY312B3_EXACTLY = ConstantCondition(sys.version_info == (3, 12, 0, 'beta', 3))
PY312B4_EXACTLY = ConstantCondition(sys.version_info == (3, 12, 0, 'beta', 4))

class _Definition(object):
    __slots__ = ('__name__', 'when', 'run_alone', 'ignore_coverage', 'options')

    def __init__(self, when, run_alone, ignore_coverage, options):
        if False:
            i = 10
            return i + 15
        assert isinstance(when, Condition)
        assert isinstance(run_alone, Condition)
        assert isinstance(ignore_coverage, Condition)
        self.when = when
        self.__name__ = None
        self.run_alone = run_alone
        self.ignore_coverage = ignore_coverage
        if options:
            for v in options.values():
                assert isinstance(v, tuple) and len(v) == 2
                assert isinstance(v[0], Condition)
        self.options = options

    def __set_name__(self, owner, name):
        if False:
            return 10
        self.__name__ = name

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<%s for %s when=%r=%s run_alone=%r=%s>' % (type(self).__name__, self.__name__, self.when, bool(self.when), self.run_alone, bool(self.run_alone))

class _Action(_Definition):
    __slots__ = ('reason',)

    def __init__(self, reason='', when=ALWAYS, run_alone=NEVER, ignore_coverage=NEVER, options=None):
        if False:
            while True:
                i = 10
        _Definition.__init__(self, when, run_alone, ignore_coverage, options)
        self.reason = reason

class RunAlone(_Action):
    __slots__ = ()

    def __init__(self, reason='', when=ALWAYS, ignore_coverage=NEVER):
        if False:
            i = 10
            return i + 15
        _Action.__init__(self, reason, run_alone=when, ignore_coverage=ignore_coverage)

class Failing(_Action):
    __slots__ = ()

class Flaky(Failing):
    __slots__ = ()

class Ignored(_Action):
    __slots__ = ()

class Multi(object):

    def __init__(self):
        if False:
            return 10
        self._conds = []

    def flaky(self, reason='', when=True, ignore_coverage=NEVER, run_alone=NEVER):
        if False:
            return 10
        self._conds.append(Flaky(reason, when=when, ignore_coverage=ignore_coverage, run_alone=run_alone))
        return self

    def ignored(self, reason='', when=True):
        if False:
            return 10
        self._conds.append(Ignored(reason, when=when))
        return self

    def __set_name__(self, owner, name):
        if False:
            print('Hello World!')
        for c in self._conds:
            c.__set_name__(owner, name)

class DefinitionsMeta(type):

    @classmethod
    def __prepare__(mcs, name, bases):
        if False:
            for i in range(10):
                print('nop')
        return SetOnceMapping()

class SetOnceMapping(dict):

    def __setitem__(self, name, value):
        if False:
            while True:
                i = 10
        if name in self:
            raise AttributeError(name)
        dict.__setitem__(self, name, value)
som = SetOnceMapping()
som[1] = 1
try:
    som[1] = 2
except AttributeError:
    del som
else:
    raise AssertionError('SetOnceMapping is broken')
DefinitionsBase = DefinitionsMeta('DefinitionsBase', (object,), {})

class Definitions(DefinitionsBase):
    test__util = RunAlone("\n        If we have extra greenlets hanging around due to changes in GC, we won't\n        match the expected output.\n\n        So far, this is only seen on one version, in CI environment.\n        ", when=CI & (PY312B3_EXACTLY | PY312B4_EXACTLY))
    test__issue6 = Flaky("test__issue6 (see comments in test file) is really flaky on both Travis and Appveyor;\n        on Travis we could just run the test again (but that gets old fast), but on appveyor\n        we don't have that option without a new commit---and sometimes we really need a build\n        to succeed in order to get a release wheel")
    test__core_fork = Ignored("fork watchers don't get called on windows\n        because fork is not a concept windows has.\n        See this file for a detailed explanation.", when=WIN)
    test__greenletset = Flaky(when=WIN, ignore_coverage=PYPY)
    test__example_udp_client = test__example_udp_server = Flaky("\n        These both run on port 9000 and can step on each other...seems\n        like the appveyor containers aren't fully port safe? Or it\n        takes longer for the processes to shut down? Or we run them in\n        a different order in the process pool than we do other places?\n\n        On PyPy on Travis, this fails to get the correct results,\n        sometimes. I can't reproduce locally\n        ", when=APPVEYOR | PYPY & TRAVIS)
    test__server_pywsgi = Flaky(when=APPVEYOR)
    test_threading = Multi().ignored('\n        This one seems to just stop right after patching is done. It\n        passes on a local win 10 vm, and the main test_threading_2.py\n        does as well. Based on the printouts we added, it appears to\n        not even finish importing:\n        https://ci.appveyor.com/project/denik/gevent/build/1.0.1277/job/tpvhesij5gldjxqw#L1190\n        Ignored because it takes two minutes to time out.\n        ', when=APPVEYOR & LIBUV & PYPY).flaky("\n        test_set_and_clear in Py3 relies on 5 threads all starting and\n        coming to an Event wait point while a sixth thread sleeps for a half\n        second. The sixth thread then does something and checks that\n        the 5 threads were all at the wait point. But the timing is sometimes\n        too tight for appveyor. This happens even if Event isn't\n        monkey-patched\n        ", when=APPVEYOR & PY3)
    test_ftplib = Flaky('\n        could be a problem of appveyor - not sure\n         ======================================================================\n          ERROR: test_af (__main__.TestIPv6Environment)\n         ----------------------------------------------------------------------\n          File "C:\\Python27-x64\\lib\\ftplib.py", line 135, in connect\n            self.sock = socket.create_connection((self.host, self.port), self.timeout)\n          File "c:\\projects\\gevent\\gevent\\socket.py", line 73, in create_connection\n            raise err\n          error: [Errno 10049] [Error 10049] The requested address is not valid in its context.\n        XXX: On Jan 3 2016 this suddenly started passing on Py27/64; no idea why, the python version\n        was 2.7.11 before and after.\n        ', when=APPVEYOR & BIT_64)
    test__backdoor = Flaky(when=LEAKTEST | PYPY)
    test__socket_errors = Flaky(when=LEAKTEST)
    test_signal = Multi().flaky('On Travis, this very frequently fails due to timing', when=TRAVIS & LEAKTEST, run_alone=APPVEYOR).ignored('\n        This fails to run a single test. It looks like just importing the module\n        can hang. All I see is the output from patch_all()\n        ', when=APPVEYOR & PYPY3)
    test__monkey_sigchld_2 = Ignored("\n        This hangs for no apparent reason when run by the testrunner,\n        even wher maked standalone when run standalone from the\n        command line, it's fine. Issue in pypy2 6.0?\n        ", when=PYPY & LIBUV)
    test_ssl = Ignored("\n        PyPy 7.0 and 7.1 on Travis with Ubunto Xenial 16.04 can't\n        allocate SSL Context objects, either in Python 2.7 or 3.6.\n        There must be some library incompatibility. No point even\n        running them. XXX: Remember to turn this back on.\n\n        On Windows, with PyPy3.7 7.3.7, there seem to be all kind of certificate\n        errors.\n        ", when=PYPY & TRAVIS | PYPY3 & WIN)
    test_httpservers = Ignored('\n        All the CGI tests hang. There appear to be subprocess problems.\n        ', when=PYPY3 & WIN)
    test__pywsgi = Ignored('\n        XXX: Re-enable this when we can investigate more. This has\n        started crashing with a SystemError. I cannot reproduce with\n        the same version on macOS and I cannot reproduce with the same\n        version in a Linux vm. Commenting out individual tests just\n        moves the crash around.\n        https://bitbucket.org/pypy/pypy/issues/2769/systemerror-unexpected-internal-exception\n\n        On Appveyor 3.8.0, for some reason this takes *way* too long, about 100s, which\n        often goes just over the default timeout of 100s. This makes no sense.\n        But it also takes nearly that long in 3.7. 3.6 and earlier are much faster.\n\n        It also takes just over 100s on PyPy 3.7.\n        ', when=PYPY & TRAVIS & LIBUV | PY380_EXACTLY, run_alone=CI & LEAKTEST & PY3 | PYPY & LIBUV, options={'timeout': (CI & PYPY, 180)})
    test_subprocess = Multi().flaky("Unknown, can't reproduce locally; times out one test", when=PYPY & PY3 & TRAVIS, ignore_coverage=ALWAYS).ignored("Tests don't even start before the process times out.", when=PYPY3 & WIN)
    test__threadpool = Ignored('\n        XXX: Re-enable these when we have more time to investigate.\n\n        This test, which normally takes ~60s, sometimes\n        hangs forever after running several tests. I cannot reproduce,\n        it seems highly load dependent. Observed with both libev and libuv.\n        ', when=TRAVIS & (PYPY | OSX), options={'timeout': (CI & PYPY, 180)})
    test__threading_2 = Ignored('\n        This test, which normally takes 4-5s, sometimes\n        hangs forever after running two tests. I cannot reproduce,\n        it seems highly load dependent. Observed with both libev and libuv.\n        ', when=TRAVIS & (PYPY | OSX), options={'timeout': (CI & PYPY, 180)})
    test__issue230 = Ignored('\n        This rarely hangs for unknown reasons. I cannot reproduce\n        locally.\n        ', when=TRAVIS & OSX)
    test_selectors = Flaky('\n        Timing issues on appveyor.\n        ', when=PY3 & APPVEYOR, ignore_coverage=ALWAYS)
    test__example_portforwarder = Flaky('\n        This one sometimes times out, often after output "The process\n        with PID XXX could not be terminated. Reason: There is no\n        running instance of the task.",\n        ', when=APPVEYOR | COVERAGE)
    test__issue302monkey = test__threading_vs_settrace = Flaky('\n        The gevent concurrency plugin tends to slow things\n        down and get us past our default timeout value. These\n        tests in particular are sensitive to it. So in fact we just turn them\n        off.\n        ', when=COVERAGE, ignore_coverage=ALWAYS)
    test__hub_join_timeout = Ignored("\n        This sometimes times out. It appears to happen when the\n        times take too long and a test raises a FlakyTestTimeout error,\n        aka a unittest.SkipTest error. This probably indicates that we're\n        not cleaning something up correctly:\n\n        .....ss\n        GEVENTTEST_USE_RESOURCES=-network C:\\Python38-x64\\python.exe -u \\\n           -mgevent.tests.test__hub_join_timeout [code TIMEOUT] [took 100.4s]\n        ", when=APPVEYOR)
    test__example_wsgiserver = test__example_webproxy = RunAlone('\n        These share the same port, which means they can conflict\n        between concurrent test runs too\n        XXX: Fix this by dynamically picking a port.\n        ')
    test__pool = RunAlone('\n        On a heavily loaded box, these can all take upwards of 200s.\n        ', when=CI & LEAKTEST | PYPY3 & APPVEYOR)
    test_socket = RunAlone('Sometimes has unexpected timeouts', when=CI & PYPY & PY3, ignore_coverage=ALWAYS)
    test__refcount = Ignored('Sometimes fails to connect for no reason', when=CI & OSX | CI & PYPY | APPVEYOR, ignore_coverage=PYPY)
    test__doctests = Ignored('Sometimes times out during/after gevent._config.Config', when=CI & OSX)
IGNORE_COVERAGE = []
TEST_FILE_OPTIONS = {}
FAILING_TESTS = []
IGNORED_TESTS = []
RUN_ALONE = []

def populate():
    if False:
        for i in range(10):
            print('nop')
    for (k, v) in Definitions.__dict__.items():
        if isinstance(v, Multi):
            actions = v._conds
        else:
            actions = (v,)
        test_name = k + '.py'
        del k, v
        for action in actions:
            if not isinstance(action, _Action):
                continue
            if action.run_alone:
                RUN_ALONE.append(test_name)
            if action.ignore_coverage:
                IGNORE_COVERAGE.append(test_name)
            if action.options:
                for (opt_name, (condition, value)) in action.options.items():
                    if condition:
                        TEST_FILE_OPTIONS.setdefault(test_name, {})[opt_name] = value
            if action.when:
                if isinstance(action, Ignored):
                    IGNORED_TESTS.append(test_name)
                elif isinstance(action, Flaky):
                    FAILING_TESTS.append('FLAKY ' + test_name)
                elif isinstance(action, Failing):
                    FAILING_TESTS.append(test_name)
    FAILING_TESTS.sort()
    IGNORED_TESTS.sort()
    RUN_ALONE.sort()
populate()
if __name__ == '__main__':
    print('known_failures:\n', FAILING_TESTS)
    print('ignored tests:\n', IGNORED_TESTS)
    print('run alone:\n', RUN_ALONE)
    print('options:\n', TEST_FILE_OPTIONS)
    print('ignore during coverage:\n', IGNORE_COVERAGE)