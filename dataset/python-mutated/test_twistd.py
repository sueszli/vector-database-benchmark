"""
Tests for L{twisted.application.app} and L{twisted.scripts.twistd}.
"""
import errno
import inspect
import os
import pickle
import signal
import sys
try:
    import grp as _grp
    import pwd as _pwd
except ImportError:
    pwd = None
    grp = None
else:
    pwd = _pwd
    grp = _grp
from io import StringIO
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import internet, logger, plugin
from twisted.application import app, reactors, service
from twisted.application.service import IServiceMaker
from twisted.internet.base import ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReactorDaemonize, _ISupportsExitSignalCapturing
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.internet.testing import MemoryReactor
from twisted.logger import ILogObserver, globalLogBeginner, globalLogPublisher
from twisted.python import util
from twisted.python.components import Componentized
from twisted.python.fakepwd import UserDatabase
from twisted.python.log import ILogObserver as LegacyILogObserver, textFromEventDict
from twisted.python.reflect import requireModule
from twisted.python.runtime import platformType
from twisted.python.usage import UsageError
from twisted.scripts import twistd
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
_twistd_unix = requireModule('twisted.scripts._twistd_unix')
if _twistd_unix:
    from twisted.scripts._twistd_unix import UnixApplicationRunner, UnixAppLogger, checkPID
syslog = requireModule('twisted.python.syslog')
profile = requireModule('profile')
pstats = requireModule('pstats')
cProfile = requireModule('cProfile')

def patchUserDatabase(patch, user, uid, group, gid):
    if False:
        i = 10
        return i + 15
    '\n    Patch L{pwd.getpwnam} so that it behaves as though only one user exists\n    and patch L{grp.getgrnam} so that it behaves as though only one group\n    exists.\n\n    @param patch: A function like L{TestCase.patch} which will be used to\n        install the fake implementations.\n\n    @type user: C{str}\n    @param user: The name of the single user which will exist.\n\n    @type uid: C{int}\n    @param uid: The UID of the single user which will exist.\n\n    @type group: C{str}\n    @param group: The name of the single user which will exist.\n\n    @type gid: C{int}\n    @param gid: The GID of the single group which will exist.\n    '
    pwent = pwd.getpwuid(os.getuid())
    grent = grp.getgrgid(os.getgid())
    database = UserDatabase()
    database.addUser(user, pwent.pw_passwd, uid, gid, pwent.pw_gecos, pwent.pw_dir, pwent.pw_shell)

    def getgrnam(name):
        if False:
            while True:
                i = 10
        result = list(grent)
        result[result.index(grent.gr_name)] = group
        result[result.index(grent.gr_gid)] = gid
        result = tuple(result)
        return {group: result}[name]
    patch(pwd, 'getpwnam', database.getpwnam)
    patch(grp, 'getgrnam', getgrnam)
    patch(pwd, 'getpwuid', database.getpwuid)

class MockServiceMaker:
    """
    A non-implementation of L{twisted.application.service.IServiceMaker}.
    """
    tapname = 'ueoa'

    def makeService(self, options):
        if False:
            print('Hello World!')
        '\n        Take a L{usage.Options} instance and return a\n        L{service.IService} provider.\n        '
        self.options = options
        self.service = service.Service()
        return self.service

class CrippledAppLogger(app.AppLogger):
    """
    @see: CrippledApplicationRunner.
    """

    def start(self, application):
        if False:
            while True:
                i = 10
        pass

class CrippledApplicationRunner(twistd._SomeApplicationRunner):
    """
    An application runner that cripples the platform-specific runner and
    nasty side-effect-having code so that we can use it without actually
    running any environment-affecting code.
    """
    loggerFactory = CrippledAppLogger

    def preApplication(self):
        if False:
            i = 10
            return i + 15
        pass

    def postApplication(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class ServerOptionsTests(TestCase):
    """
    Non-platform-specific tests for the platform-specific ServerOptions class.
    """

    def test_subCommands(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        subCommands is built from IServiceMaker plugins, and is sorted\n        alphabetically.\n        '

        class FakePlugin:

            def __init__(self, name):
                if False:
                    i = 10
                    return i + 15
                self.tapname = name
                self._options = 'options for ' + name
                self.description = 'description of ' + name

            def options(self):
                if False:
                    i = 10
                    return i + 15
                return self._options
        apple = FakePlugin('apple')
        banana = FakePlugin('banana')
        coconut = FakePlugin('coconut')
        donut = FakePlugin('donut')

        def getPlugins(interface):
            if False:
                print('Hello World!')
            self.assertEqual(interface, IServiceMaker)
            yield coconut
            yield banana
            yield donut
            yield apple
        config = twistd.ServerOptions()
        self.assertEqual(config._getPlugins, plugin.getPlugins)
        config._getPlugins = getPlugins
        subCommands = config.subCommands
        expectedOrder = [apple, banana, coconut, donut]
        for (subCommand, expectedCommand) in zip(subCommands, expectedOrder):
            (name, shortcut, parserClass, documentation) = subCommand
            self.assertEqual(name, expectedCommand.tapname)
            self.assertIsNone(shortcut)
            (self.assertEqual(parserClass(), expectedCommand._options),)
            self.assertEqual(documentation, expectedCommand.description)

    def test_sortedReactorHelp(self):
        if False:
            while True:
                i = 10
        '\n        Reactor names are listed alphabetically by I{--help-reactors}.\n        '

        class FakeReactorInstaller:

            def __init__(self, name):
                if False:
                    return 10
                self.shortName = 'name of ' + name
                self.description = 'description of ' + name
                self.moduleName = 'twisted.internet.default'
        apple = FakeReactorInstaller('apple')
        banana = FakeReactorInstaller('banana')
        coconut = FakeReactorInstaller('coconut')
        donut = FakeReactorInstaller('donut')

        def getReactorTypes():
            if False:
                return 10
            yield coconut
            yield banana
            yield donut
            yield apple
        config = twistd.ServerOptions()
        self.assertEqual(config._getReactorTypes, reactors.getReactorTypes)
        config._getReactorTypes = getReactorTypes
        config.messageOutput = StringIO()
        self.assertRaises(SystemExit, config.parseOptions, ['--help-reactors'])
        helpOutput = config.messageOutput.getvalue()
        indexes = []
        for reactor in (apple, banana, coconut, donut):

            def getIndex(s):
                if False:
                    while True:
                        i = 10
                self.assertIn(s, helpOutput)
                indexes.append(helpOutput.index(s))
            getIndex(reactor.shortName)
            getIndex(reactor.description)
        self.assertEqual(indexes, sorted(indexes), 'reactor descriptions were not in alphabetical order: {!r}'.format(helpOutput))

    def test_postOptionsSubCommandCausesNoSave(self):
        if False:
            print('Hello World!')
        '\n        postOptions should set no_save to True when a subcommand is used.\n        '
        config = twistd.ServerOptions()
        config.subCommand = 'ueoa'
        config.postOptions()
        self.assertTrue(config['no_save'])

    def test_postOptionsNoSubCommandSavesAsUsual(self):
        if False:
            while True:
                i = 10
        '\n        If no sub command is used, postOptions should not touch no_save.\n        '
        config = twistd.ServerOptions()
        config.postOptions()
        self.assertFalse(config['no_save'])

    def test_listAllProfilers(self):
        if False:
            return 10
        '\n        All the profilers that can be used in L{app.AppProfiler} are listed in\n        the help output.\n        '
        config = twistd.ServerOptions()
        helpOutput = str(config)
        for profiler in app.AppProfiler.profilers:
            self.assertIn(profiler, helpOutput)

    @skipIf(not _twistd_unix, 'twistd unix not available')
    def test_defaultUmask(self):
        if False:
            return 10
        '\n        The default value for the C{umask} option is L{None}.\n        '
        config = twistd.ServerOptions()
        self.assertIsNone(config['umask'])

    @skipIf(not _twistd_unix, 'twistd unix not available')
    def test_umask(self):
        if False:
            i = 10
            return i + 15
        '\n        The value given for the C{umask} option is parsed as an octal integer\n        literal.\n        '
        config = twistd.ServerOptions()
        config.parseOptions(['--umask', '123'])
        self.assertEqual(config['umask'], 83)
        config.parseOptions(['--umask', '0123'])
        self.assertEqual(config['umask'], 83)

    @skipIf(not _twistd_unix, 'twistd unix not available')
    def test_invalidUmask(self):
        if False:
            return 10
        '\n        If a value is given for the C{umask} option which cannot be parsed as\n        an integer, L{UsageError} is raised by L{ServerOptions.parseOptions}.\n        '
        config = twistd.ServerOptions()
        self.assertRaises(UsageError, config.parseOptions, ['--umask', 'abcdef'])

    def test_unimportableConfiguredLogObserver(self):
        if False:
            i = 10
            return i + 15
        '\n        C{--logger} with an unimportable module raises a L{UsageError}.\n        '
        config = twistd.ServerOptions()
        e = self.assertRaises(UsageError, config.parseOptions, ['--logger', 'no.such.module.I.hope'])
        self.assertTrue(e.args[0].startswith("Logger 'no.such.module.I.hope' could not be imported: 'no.such.module.I.hope' does not name an object"))
        self.assertNotIn('\n', e.args[0])

    def test_badAttributeWithConfiguredLogObserver(self):
        if False:
            return 10
        '\n        C{--logger} with a non-existent object raises a L{UsageError}.\n        '
        config = twistd.ServerOptions()
        e = self.assertRaises(UsageError, config.parseOptions, ['--logger', 'twisted.test.test_twistd.FOOBAR'])
        self.assertTrue(e.args[0].startswith("Logger 'twisted.test.test_twistd.FOOBAR' could not be imported: module 'twisted.test.test_twistd' has no attribute 'FOOBAR'"))
        self.assertNotIn('\n', e.args[0])

    def test_version(self):
        if False:
            print('Hello World!')
        '\n        C{--version} prints the version.\n        '
        from twisted import copyright
        if platformType == 'win32':
            name = '(the Twisted Windows runner)'
        else:
            name = '(the Twisted daemon)'
        expectedOutput = 'twistd {} {}\n{}\n'.format(name, copyright.version, copyright.copyright)
        stdout = StringIO()
        config = twistd.ServerOptions(stdout=stdout)
        e = self.assertRaises(SystemExit, config.parseOptions, ['--version'])
        self.assertIs(e.code, None)
        self.assertEqual(stdout.getvalue(), expectedOutput)

    def test_printSubCommandForUsageError(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Command is printed when an invalid option is requested.\n        '
        stdout = StringIO()
        config = twistd.ServerOptions(stdout=stdout)
        self.assertRaises(UsageError, config.parseOptions, ['web --foo'])

@skipIf(not _twistd_unix, 'twistd unix not available')
class CheckPIDTests(TestCase):
    """
    Tests for L{checkPID}.
    """

    def test_notExists(self):
        if False:
            print('Hello World!')
        '\n        Nonexistent PID file is not an error.\n        '
        self.patch(os.path, 'exists', lambda _: False)
        checkPID('non-existent PID file')

    def test_nonNumeric(self):
        if False:
            while True:
                i = 10
        '\n        Non-numeric content in a PID file causes a system exit.\n        '
        pidfile = self.mktemp()
        with open(pidfile, 'w') as f:
            f.write('non-numeric')
        e = self.assertRaises(SystemExit, checkPID, pidfile)
        self.assertIn('non-numeric value', e.code)

    def test_anotherRunning(self):
        if False:
            return 10
        '\n        Another running twistd server causes a system exit.\n        '
        pidfile = self.mktemp()
        with open(pidfile, 'w') as f:
            f.write('42')

        def kill(pid, sig):
            if False:
                return 10
            pass
        self.patch(os, 'kill', kill)
        e = self.assertRaises(SystemExit, checkPID, pidfile)
        self.assertIn('Another twistd server', e.code)

    def test_stale(self):
        if False:
            while True:
                i = 10
        '\n        Stale PID file is removed without causing a system exit.\n        '
        pidfile = self.mktemp()
        with open(pidfile, 'w') as f:
            f.write(str(os.getpid() + 1))

        def kill(pid, sig):
            if False:
                for i in range(10):
                    print('nop')
            raise OSError(errno.ESRCH, 'fake')
        self.patch(os, 'kill', kill)
        checkPID(pidfile)
        self.assertFalse(os.path.exists(pidfile))

    def test_unexpectedOSError(self):
        if False:
            while True:
                i = 10
        '\n        An unexpected L{OSError} when checking the validity of a\n        PID in a C{pidfile} terminates the process via L{SystemExit}.\n        '
        pidfile = self.mktemp()
        with open(pidfile, 'w') as f:
            f.write('3581')

        def kill(pid, sig):
            if False:
                while True:
                    i = 10
            raise OSError(errno.EBADF, 'fake')
        self.patch(os, 'kill', kill)
        e = self.assertRaises(SystemExit, checkPID, pidfile)
        self.assertIsNot(e.code, None)
        self.assertTrue(e.args[0].startswith("Can't check status of PID"))

class TapFileTests(TestCase):
    """
    Test twistd-related functionality that requires a tap file on disk.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        '\n        Create a trivial Application and put it in a tap file on disk.\n        '
        self.tapfile = self.mktemp()
        with open(self.tapfile, 'wb') as f:
            pickle.dump(service.Application('Hi!'), f)

    def test_createOrGetApplicationWithTapFile(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Ensure that the createOrGetApplication call that 'twistd -f foo.tap'\n        makes will load the Application out of foo.tap.\n        "
        config = twistd.ServerOptions()
        config.parseOptions(['-f', self.tapfile])
        application = CrippledApplicationRunner(config).createOrGetApplication()
        self.assertEqual(service.IService(application).name, 'Hi!')

class TestLoggerFactory:
    """
    A logger factory for L{TestApplicationRunner}.
    """

    def __init__(self, runner):
        if False:
            for i in range(10):
                print('nop')
        self.runner = runner

    def start(self, application):
        if False:
            print('Hello World!')
        '\n        Save the logging start on the C{runner} instance.\n        '
        self.runner.order.append('log')
        self.runner.hadApplicationLogObserver = hasattr(self.runner, 'application')

    def stop(self):
        if False:
            print('Hello World!')
        "\n        Don't log anything.\n        "

class TestApplicationRunner(app.ApplicationRunner):
    """
    An ApplicationRunner which tracks the environment in which its methods are
    called.
    """

    def __init__(self, options):
        if False:
            return 10
        app.ApplicationRunner.__init__(self, options)
        self.order = []
        self.logger = TestLoggerFactory(self)

    def preApplication(self):
        if False:
            i = 10
            return i + 15
        self.order.append('pre')
        self.hadApplicationPreApplication = hasattr(self, 'application')

    def postApplication(self):
        if False:
            for i in range(10):
                print('nop')
        self.order.append('post')
        self.hadApplicationPostApplication = hasattr(self, 'application')

class ApplicationRunnerTests(TestCase):
    """
    Non-platform-specific tests for the platform-specific ApplicationRunner.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        config = twistd.ServerOptions()
        self.serviceMaker = MockServiceMaker()
        config.loadedPlugins = {'test_command': self.serviceMaker}
        config.subOptions = object()
        config.subCommand = 'test_command'
        self.config = config

    def test_applicationRunnerGetsCorrectApplication(self):
        if False:
            while True:
                i = 10
        '\n        Ensure that a twistd plugin gets used in appropriate ways: it\n        is passed its Options instance, and the service it returns is\n        added to the application.\n        '
        arunner = CrippledApplicationRunner(self.config)
        arunner.run()
        self.assertIs(self.serviceMaker.options, self.config.subOptions, 'ServiceMaker.makeService needs to be passed the correct sub Command object.')
        self.assertIs(self.serviceMaker.service, service.IService(arunner.application).services[0], "ServiceMaker.makeService's result needs to be set as a child of the Application.")

    def test_preAndPostApplication(self):
        if False:
            i = 10
            return i + 15
        '\n        Test thet preApplication and postApplication methods are\n        called by ApplicationRunner.run() when appropriate.\n        '
        s = TestApplicationRunner(self.config)
        s.run()
        self.assertFalse(s.hadApplicationPreApplication)
        self.assertTrue(s.hadApplicationPostApplication)
        self.assertTrue(s.hadApplicationLogObserver)
        self.assertEqual(s.order, ['pre', 'log', 'post'])

    def _applicationStartsWithConfiguredID(self, argv, uid, gid):
        if False:
            return 10
        '\n        Assert that given a particular command line, an application is started\n        as a particular UID/GID.\n\n        @param argv: A list of strings giving the options to parse.\n        @param uid: An integer giving the expected UID.\n        @param gid: An integer giving the expected GID.\n        '
        self.config.parseOptions(argv)
        events = []

        class FakeUnixApplicationRunner(twistd._SomeApplicationRunner):

            def setupEnvironment(self, chroot, rundir, nodaemon, umask, pidfile):
                if False:
                    print('Hello World!')
                events.append('environment')

            def shedPrivileges(self, euid, uid, gid):
                if False:
                    return 10
                events.append(('privileges', euid, uid, gid))

            def startReactor(self, reactor, oldstdout, oldstderr):
                if False:
                    print('Hello World!')
                events.append('reactor')

            def removePID(self, pidfile):
                if False:
                    for i in range(10):
                        print('nop')
                pass

        @implementer(service.IService, service.IProcess)
        class FakeService:
            parent = None
            running = None
            name = None
            processName = None
            uid = None
            gid = None

            def setName(self, name):
                if False:
                    print('Hello World!')
                pass

            def setServiceParent(self, parent):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def disownServiceParent(self):
                if False:
                    return 10
                pass

            def privilegedStartService(self):
                if False:
                    while True:
                        i = 10
                events.append('privilegedStartService')

            def startService(self):
                if False:
                    while True:
                        i = 10
                events.append('startService')

            def stopService(self):
                if False:
                    i = 10
                    return i + 15
                pass
        application = FakeService()
        verifyObject(service.IService, application)
        verifyObject(service.IProcess, application)
        runner = FakeUnixApplicationRunner(self.config)
        runner.preApplication()
        runner.application = application
        runner.postApplication()
        self.assertEqual(events, ['environment', 'privilegedStartService', ('privileges', False, uid, gid), 'startService', 'reactor'])

    @skipIf(not getattr(os, 'setuid', None), 'Platform does not support --uid/--gid twistd options.')
    def test_applicationStartsWithConfiguredNumericIDs(self):
        if False:
            print('Hello World!')
        '\n        L{postApplication} should change the UID and GID to the values\n        specified as numeric strings by the configuration after running\n        L{service.IService.privilegedStartService} and before running\n        L{service.IService.startService}.\n        '
        uid = 1234
        gid = 4321
        self._applicationStartsWithConfiguredID(['--uid', str(uid), '--gid', str(gid)], uid, gid)

    @skipIf(not getattr(os, 'setuid', None), 'Platform does not support --uid/--gid twistd options.')
    def test_applicationStartsWithConfiguredNameIDs(self):
        if False:
            i = 10
            return i + 15
        '\n        L{postApplication} should change the UID and GID to the values\n        specified as user and group names by the configuration after running\n        L{service.IService.privilegedStartService} and before running\n        L{service.IService.startService}.\n        '
        user = 'foo'
        uid = 1234
        group = 'bar'
        gid = 4321
        patchUserDatabase(self.patch, user, uid, group, gid)
        self._applicationStartsWithConfiguredID(['--uid', user, '--gid', group], uid, gid)

    def test_startReactorRunsTheReactor(self):
        if False:
            return 10
        '\n        L{startReactor} calls L{reactor.run}.\n        '
        reactor = DummyReactor()
        runner = app.ApplicationRunner({'profile': False, 'profiler': 'profile', 'debug': False})
        runner.startReactor(reactor, None, None)
        self.assertTrue(reactor.called, 'startReactor did not call reactor.run()')

    def test_applicationRunnerChoosesReactorIfNone(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{ApplicationRunner} chooses a reactor if none is specified.\n        '
        reactor = DummyReactor()
        self.patch(internet, 'reactor', reactor)
        runner = app.ApplicationRunner({'profile': False, 'profiler': 'profile', 'debug': False})
        runner.startReactor(None, None, None)
        self.assertTrue(reactor.called)

    def test_applicationRunnerCapturesSignal(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the reactor exits with a signal, the application runner caches\n        the signal.\n        '

        class DummyReactorWithSignal(ReactorBase):
            """
            A dummy reactor, providing a C{run} method, and setting the
            _exitSignal attribute to a nonzero value.
            """

            def installWaker(self):
                if False:
                    for i in range(10):
                        print('nop')
                '\n                Dummy method, does nothing.\n                '

            def run(self):
                if False:
                    print('Hello World!')
                '\n                A fake run method setting _exitSignal to a nonzero value\n                '
                self._exitSignal = 2
        reactor = DummyReactorWithSignal()
        runner = app.ApplicationRunner({'profile': False, 'profiler': 'profile', 'debug': False})
        runner.startReactor(reactor, None, None)
        self.assertEquals(2, runner._exitSignal)

    def test_applicationRunnerIgnoresNoSignal(self):
        if False:
            print('Hello World!')
        '\n        The runner sets its _exitSignal instance attribute to None if\n        the reactor does not implement L{_ISupportsExitSignalCapturing}.\n        '

        class DummyReactorWithExitSignalAttribute:
            """
            A dummy reactor, providing a C{run} method, and setting the
            _exitSignal attribute to a nonzero value.
            """

            def installWaker(self):
                if False:
                    return 10
                '\n                Dummy method, does nothing.\n                '

            def run(self):
                if False:
                    print('Hello World!')
                '\n                A fake run method setting _exitSignal to a nonzero value\n                that should be ignored.\n                '
                self._exitSignal = 2
        reactor = DummyReactorWithExitSignalAttribute()
        runner = app.ApplicationRunner({'profile': False, 'profiler': 'profile', 'debug': False})
        runner.startReactor(reactor, None, None)
        self.assertEquals(None, runner._exitSignal)

@skipIf(not _twistd_unix, 'twistd unix not available')
class UnixApplicationRunnerSetupEnvironmentTests(TestCase):
    """
    Tests for L{UnixApplicationRunner.setupEnvironment}.

    @ivar root: The root of the filesystem, or C{unset} if none has been
        specified with a call to L{os.chroot} (patched for this TestCase with
        L{UnixApplicationRunnerSetupEnvironmentTests.chroot}).

    @ivar cwd: The current working directory of the process, or C{unset} if
        none has been specified with a call to L{os.chdir} (patched for this
        TestCase with L{UnixApplicationRunnerSetupEnvironmentTests.chdir}).

    @ivar mask: The current file creation mask of the process, or C{unset} if
        none has been specified with a call to L{os.umask} (patched for this
        TestCase with L{UnixApplicationRunnerSetupEnvironmentTests.umask}).

    @ivar daemon: A boolean indicating whether daemonization has been performed
        by a call to L{_twistd_unix.daemonize} (patched for this TestCase with
        L{UnixApplicationRunnerSetupEnvironmentTests}.
    """
    unset = object()

    def setUp(self):
        if False:
            print('Hello World!')
        self.root = self.unset
        self.cwd = self.unset
        self.mask = self.unset
        self.daemon = False
        self.pid = os.getpid()
        self.patch(os, 'chroot', lambda path: setattr(self, 'root', path))
        self.patch(os, 'chdir', lambda path: setattr(self, 'cwd', path))
        self.patch(os, 'umask', lambda mask: setattr(self, 'mask', mask))
        self.runner = UnixApplicationRunner(twistd.ServerOptions())
        self.runner.daemonize = self.daemonize

    def daemonize(self, reactor):
        if False:
            while True:
                i = 10
        '\n        Indicate that daemonization has happened and change the PID so that the\n        value written to the pidfile can be tested in the daemonization case.\n        '
        self.daemon = True
        self.patch(os, 'getpid', lambda : self.pid + 1)

    def test_chroot(self):
        if False:
            i = 10
            return i + 15
        '\n        L{UnixApplicationRunner.setupEnvironment} changes the root of the\n        filesystem if passed a non-L{None} value for the C{chroot} parameter.\n        '
        self.runner.setupEnvironment('/foo/bar', '.', True, None, None)
        self.assertEqual(self.root, '/foo/bar')

    def test_noChroot(self):
        if False:
            while True:
                i = 10
        '\n        L{UnixApplicationRunner.setupEnvironment} does not change the root of\n        the filesystem if passed L{None} for the C{chroot} parameter.\n        '
        self.runner.setupEnvironment(None, '.', True, None, None)
        self.assertIs(self.root, self.unset)

    def test_changeWorkingDirectory(self):
        if False:
            i = 10
            return i + 15
        '\n        L{UnixApplicationRunner.setupEnvironment} changes the working directory\n        of the process to the path given for the C{rundir} parameter.\n        '
        self.runner.setupEnvironment(None, '/foo/bar', True, None, None)
        self.assertEqual(self.cwd, '/foo/bar')

    def test_daemonize(self):
        if False:
            i = 10
            return i + 15
        '\n        L{UnixApplicationRunner.setupEnvironment} daemonizes the process if\n        C{False} is passed for the C{nodaemon} parameter.\n        '
        with AlternateReactor(FakeDaemonizingReactor()):
            self.runner.setupEnvironment(None, '.', False, None, None)
        self.assertTrue(self.daemon)

    def test_noDaemonize(self):
        if False:
            i = 10
            return i + 15
        '\n        L{UnixApplicationRunner.setupEnvironment} does not daemonize the\n        process if C{True} is passed for the C{nodaemon} parameter.\n        '
        self.runner.setupEnvironment(None, '.', True, None, None)
        self.assertFalse(self.daemon)

    def test_nonDaemonPIDFile(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        L{UnixApplicationRunner.setupEnvironment} writes the process's PID to\n        the file specified by the C{pidfile} parameter.\n        "
        pidfile = self.mktemp()
        self.runner.setupEnvironment(None, '.', True, None, pidfile)
        with open(pidfile, 'rb') as f:
            pid = int(f.read())
        self.assertEqual(pid, self.pid)

    def test_daemonPIDFile(self):
        if False:
            return 10
        "\n        L{UnixApplicationRunner.setupEnvironment} writes the daemonized\n        process's PID to the file specified by the C{pidfile} parameter if\n        C{nodaemon} is C{False}.\n        "
        pidfile = self.mktemp()
        with AlternateReactor(FakeDaemonizingReactor()):
            self.runner.setupEnvironment(None, '.', False, None, pidfile)
        with open(pidfile, 'rb') as f:
            pid = int(f.read())
        self.assertEqual(pid, self.pid + 1)

    def test_umask(self):
        if False:
            print('Hello World!')
        '\n        L{UnixApplicationRunner.setupEnvironment} changes the process umask to\n        the value specified by the C{umask} parameter.\n        '
        with AlternateReactor(FakeDaemonizingReactor()):
            self.runner.setupEnvironment(None, '.', False, 123, None)
        self.assertEqual(self.mask, 123)

    def test_noDaemonizeNoUmask(self):
        if False:
            i = 10
            return i + 15
        "\n        L{UnixApplicationRunner.setupEnvironment} doesn't change the process\n        umask if L{None} is passed for the C{umask} parameter and C{True} is\n        passed for the C{nodaemon} parameter.\n        "
        self.runner.setupEnvironment(None, '.', True, None, None)
        self.assertIs(self.mask, self.unset)

    def test_daemonizedNoUmask(self):
        if False:
            return 10
        '\n        L{UnixApplicationRunner.setupEnvironment} changes the process umask to\n        C{0077} if L{None} is passed for the C{umask} parameter and C{False} is\n        passed for the C{nodaemon} parameter.\n        '
        with AlternateReactor(FakeDaemonizingReactor()):
            self.runner.setupEnvironment(None, '.', False, None, None)
        self.assertEqual(self.mask, 63)

@skipIf(not _twistd_unix, 'twistd unix not available')
class UnixApplicationRunnerStartApplicationTests(TestCase):
    """
    Tests for L{UnixApplicationRunner.startApplication}.
    """

    def test_setupEnvironment(self):
        if False:
            i = 10
            return i + 15
        '\n        L{UnixApplicationRunner.startApplication} calls\n        L{UnixApplicationRunner.setupEnvironment} with the chroot, rundir,\n        nodaemon, umask, and pidfile parameters from the configuration it is\n        constructed with.\n        '
        options = twistd.ServerOptions()
        options.parseOptions(['--nodaemon', '--umask', '0070', '--chroot', '/foo/chroot', '--rundir', '/foo/rundir', '--pidfile', '/foo/pidfile'])
        application = service.Application('test_setupEnvironment')
        self.runner = UnixApplicationRunner(options)
        args = []

        def fakeSetupEnvironment(self, chroot, rundir, nodaemon, umask, pidfile):
            if False:
                i = 10
                return i + 15
            args.extend((chroot, rundir, nodaemon, umask, pidfile))
        setupEnvironmentParameters = inspect.signature(self.runner.setupEnvironment).parameters
        fakeSetupEnvironmentParameters = inspect.signature(fakeSetupEnvironment).parameters
        fakeSetupEnvironmentParameters = fakeSetupEnvironmentParameters.copy()
        fakeSetupEnvironmentParameters.pop('self')
        self.assertEqual(setupEnvironmentParameters, fakeSetupEnvironmentParameters)
        self.patch(UnixApplicationRunner, 'setupEnvironment', fakeSetupEnvironment)
        self.patch(UnixApplicationRunner, 'shedPrivileges', lambda *a, **kw: None)
        self.patch(app, 'startApplication', lambda *a, **kw: None)
        self.runner.startApplication(application)
        self.assertEqual(args, ['/foo/chroot', '/foo/rundir', True, 56, '/foo/pidfile'])

    def test_shedPrivileges(self):
        if False:
            i = 10
            return i + 15
        '\n        L{UnixApplicationRunner.shedPrivileges} switches the user ID\n        of the process.\n        '

        def switchUIDPass(uid, gid, euid):
            if False:
                i = 10
                return i + 15
            self.assertEqual(uid, 200)
            self.assertEqual(gid, 54)
            self.assertEqual(euid, 35)
        self.patch(_twistd_unix, 'switchUID', switchUIDPass)
        runner = UnixApplicationRunner({})
        runner.shedPrivileges(35, 200, 54)

    def test_shedPrivilegesError(self):
        if False:
            print('Hello World!')
        '\n        An unexpected L{OSError} when calling\n        L{twisted.scripts._twistd_unix.shedPrivileges}\n        terminates the process via L{SystemExit}.\n        '

        def switchUIDFail(uid, gid, euid):
            if False:
                print('Hello World!')
            raise OSError(errno.EBADF, 'fake')
        runner = UnixApplicationRunner({})
        self.patch(_twistd_unix, 'switchUID', switchUIDFail)
        exc = self.assertRaises(SystemExit, runner.shedPrivileges, 35, 200, None)
        self.assertEqual(exc.code, 1)

    def _setUID(self, wantedUser, wantedUid, wantedGroup, wantedGid, pidFile):
        if False:
            for i in range(10):
                print('nop')
        '\n        Common code for tests which try to pass the the UID to\n        L{UnixApplicationRunner}.\n        '
        patchUserDatabase(self.patch, wantedUser, wantedUid, wantedGroup, wantedGid)

        def initgroups(uid, gid):
            if False:
                while True:
                    i = 10
            self.assertEqual(uid, wantedUid)
            self.assertEqual(gid, wantedGid)

        def setuid(uid):
            if False:
                while True:
                    i = 10
            self.assertEqual(uid, wantedUid)

        def setgid(gid):
            if False:
                i = 10
                return i + 15
            self.assertEqual(gid, wantedGid)
        self.patch(util, 'initgroups', initgroups)
        self.patch(os, 'setuid', setuid)
        self.patch(os, 'setgid', setgid)
        options = twistd.ServerOptions()
        options.parseOptions(['--nodaemon', '--uid', str(wantedUid), '--pidfile', pidFile])
        application = service.Application('test_setupEnvironment')
        self.runner = UnixApplicationRunner(options)
        runner = UnixApplicationRunner(options)
        runner.startApplication(application)

    def test_setUidWithoutGid(self):
        if False:
            return 10
        '\n        Starting an application with L{UnixApplicationRunner} configured\n        with a UID and no GUID will result in the GUID being\n        set to the default GUID for that UID.\n        '
        self._setUID('foo', 5151, 'bar', 4242, self.mktemp() + '_test_setUidWithoutGid.pid')

    def test_setUidSameAsCurrentUid(self):
        if False:
            print('Hello World!')
        '\n        If the specified UID is the same as the current UID of the process,\n        then a warning is displayed.\n        '
        currentUid = os.getuid()
        self._setUID('morefoo', currentUid, 'morebar', 4343, 'test_setUidSameAsCurrentUid.pid')
        warningsShown = self.flushWarnings()
        expectedWarning = 'tried to drop privileges and setuid {} but uid is already {}; should we be root? Continuing.'.format(currentUid, currentUid)
        self.assertEqual(expectedWarning, warningsShown[0]['message'])
        self.assertEqual(1, len(warningsShown), warningsShown)

@skipIf(not _twistd_unix, 'twistd unix not available')
class UnixApplicationRunnerRemovePIDTests(TestCase):
    """
    Tests for L{UnixApplicationRunner.removePID}.
    """

    def test_removePID(self):
        if False:
            while True:
                i = 10
        '\n        L{UnixApplicationRunner.removePID} deletes the file the name of\n        which is passed to it.\n        '
        runner = UnixApplicationRunner({})
        path = self.mktemp()
        os.makedirs(path)
        pidfile = os.path.join(path, 'foo.pid')
        open(pidfile, 'w').close()
        runner.removePID(pidfile)
        self.assertFalse(os.path.exists(pidfile))

    def test_removePIDErrors(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calling L{UnixApplicationRunner.removePID} with a non-existent filename\n        logs an OSError.\n        '
        runner = UnixApplicationRunner({})
        runner.removePID('fakepid')
        errors = self.flushLoggedErrors(OSError)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].value.errno, errno.ENOENT)

class FakeNonDaemonizingReactor:
    """
    A dummy reactor, providing C{beforeDaemonize} and C{afterDaemonize}
    methods, but not announcing this, and logging whether the methods have been
    called.

    @ivar _beforeDaemonizeCalled: if C{beforeDaemonize} has been called or not.
    @type _beforeDaemonizeCalled: C{bool}
    @ivar _afterDaemonizeCalled: if C{afterDaemonize} has been called or not.
    @type _afterDaemonizeCalled: C{bool}
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._beforeDaemonizeCalled = False
        self._afterDaemonizeCalled = False

    def beforeDaemonize(self):
        if False:
            return 10
        self._beforeDaemonizeCalled = True

    def afterDaemonize(self):
        if False:
            i = 10
            return i + 15
        self._afterDaemonizeCalled = True

    def addSystemEventTrigger(self, *args, **kw):
        if False:
            print('Hello World!')
        '\n        Skip event registration.\n        '

@implementer(IReactorDaemonize)
class FakeDaemonizingReactor(FakeNonDaemonizingReactor):
    """
    A dummy reactor, providing C{beforeDaemonize} and C{afterDaemonize}
    methods, announcing this, and logging whether the methods have been called.
    """

class DummyReactor:
    """
    A dummy reactor, only providing a C{run} method and checking that it
    has been called.

    @ivar called: if C{run} has been called or not.
    @type called: C{bool}
    """
    called = False

    def run(self):
        if False:
            i = 10
            return i + 15
        "\n        A fake run method, checking that it's been called one and only time.\n        "
        if self.called:
            raise RuntimeError('Already called')
        self.called = True

class AppProfilingTests(TestCase):
    """
    Tests for L{app.AppProfiler}.
    """

    @skipIf(not profile, 'profile module not available')
    def test_profile(self):
        if False:
            while True:
                i = 10
        '\n        L{app.ProfileRunner.run} should call the C{run} method of the reactor\n        and save profile data in the specified file.\n        '
        config = twistd.ServerOptions()
        config['profile'] = self.mktemp()
        config['profiler'] = 'profile'
        profiler = app.AppProfiler(config)
        reactor = DummyReactor()
        profiler.run(reactor)
        self.assertTrue(reactor.called)
        with open(config['profile']) as f:
            data = f.read()
        self.assertIn('DummyReactor.run', data)
        self.assertIn('function calls', data)

    def _testStats(self, statsClass, profile):
        if False:
            i = 10
            return i + 15
        out = StringIO()
        stdout = self.patch(sys, 'stdout', out)
        stats = statsClass(profile)
        stats.print_stats()
        stdout.restore()
        data = out.getvalue()
        self.assertIn('function calls', data)
        self.assertIn('(run)', data)

    @skipIf(not profile, 'profile module not available')
    def test_profileSaveStats(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        With the C{savestats} option specified, L{app.ProfileRunner.run}\n        should save the raw stats object instead of a summary output.\n        '
        config = twistd.ServerOptions()
        config['profile'] = self.mktemp()
        config['profiler'] = 'profile'
        config['savestats'] = True
        profiler = app.AppProfiler(config)
        reactor = DummyReactor()
        profiler.run(reactor)
        self.assertTrue(reactor.called)
        self._testStats(pstats.Stats, config['profile'])

    def test_withoutProfile(self):
        if False:
            while True:
                i = 10
        '\n        When the C{profile} module is not present, L{app.ProfilerRunner.run}\n        should raise a C{SystemExit} exception.\n        '
        savedModules = sys.modules.copy()
        config = twistd.ServerOptions()
        config['profiler'] = 'profile'
        profiler = app.AppProfiler(config)
        sys.modules['profile'] = None
        try:
            self.assertRaises(SystemExit, profiler.run, None)
        finally:
            sys.modules.clear()
            sys.modules.update(savedModules)

    @skipIf(not profile, 'profile module not available')
    def test_profilePrintStatsError(self):
        if False:
            print('Hello World!')
        '\n        When an error happens during the print of the stats, C{sys.stdout}\n        should be restored to its initial value.\n        '

        class ErroneousProfile(profile.Profile):

            def print_stats(self):
                if False:
                    i = 10
                    return i + 15
                raise RuntimeError('Boom')
        self.patch(profile, 'Profile', ErroneousProfile)
        config = twistd.ServerOptions()
        config['profile'] = self.mktemp()
        config['profiler'] = 'profile'
        profiler = app.AppProfiler(config)
        reactor = DummyReactor()
        oldStdout = sys.stdout
        self.assertRaises(RuntimeError, profiler.run, reactor)
        self.assertIs(sys.stdout, oldStdout)

    @skipIf(not cProfile, 'cProfile module not available')
    def test_cProfile(self):
        if False:
            print('Hello World!')
        '\n        L{app.CProfileRunner.run} should call the C{run} method of the\n        reactor and save profile data in the specified file.\n        '
        config = twistd.ServerOptions()
        config['profile'] = self.mktemp()
        config['profiler'] = 'cProfile'
        profiler = app.AppProfiler(config)
        reactor = DummyReactor()
        profiler.run(reactor)
        self.assertTrue(reactor.called)
        with open(config['profile']) as f:
            data = f.read()
        self.assertIn('run', data)
        self.assertIn('function calls', data)

    @skipIf(not cProfile, 'cProfile module not available')
    def test_cProfileSaveStats(self):
        if False:
            while True:
                i = 10
        '\n        With the C{savestats} option specified,\n        L{app.CProfileRunner.run} should save the raw stats object\n        instead of a summary output.\n        '
        config = twistd.ServerOptions()
        config['profile'] = self.mktemp()
        config['profiler'] = 'cProfile'
        config['savestats'] = True
        profiler = app.AppProfiler(config)
        reactor = DummyReactor()
        profiler.run(reactor)
        self.assertTrue(reactor.called)
        self._testStats(pstats.Stats, config['profile'])

    def test_withoutCProfile(self):
        if False:
            while True:
                i = 10
        '\n        When the C{cProfile} module is not present,\n        L{app.CProfileRunner.run} should raise a C{SystemExit}\n        exception and log the C{ImportError}.\n        '
        savedModules = sys.modules.copy()
        sys.modules['cProfile'] = None
        config = twistd.ServerOptions()
        config['profiler'] = 'cProfile'
        profiler = app.AppProfiler(config)
        try:
            self.assertRaises(SystemExit, profiler.run, None)
        finally:
            sys.modules.clear()
            sys.modules.update(savedModules)

    def test_unknownProfiler(self):
        if False:
            while True:
                i = 10
        '\n        Check that L{app.AppProfiler} raises L{SystemExit} when given an\n        unknown profiler name.\n        '
        config = twistd.ServerOptions()
        config['profile'] = self.mktemp()
        config['profiler'] = 'foobar'
        error = self.assertRaises(SystemExit, app.AppProfiler, config)
        self.assertEqual(str(error), 'Unsupported profiler name: foobar')

    def test_defaultProfiler(self):
        if False:
            print('Hello World!')
        '\n        L{app.Profiler} defaults to the cprofile profiler if not specified.\n        '
        profiler = app.AppProfiler({})
        self.assertEqual(profiler.profiler, 'cprofile')

    def test_profilerNameCaseInsentive(self):
        if False:
            while True:
                i = 10
        '\n        The case of the profiler name passed to L{app.AppProfiler} is not\n        relevant.\n        '
        profiler = app.AppProfiler({'profiler': 'CprOfile'})
        self.assertEqual(profiler.profiler, 'cprofile')

def _patchTextFileLogObserver(patch):
    if False:
        while True:
            i = 10
    '\n    Patch L{logger.textFileLogObserver} to record every call and keep a\n    reference to the passed log file for tests.\n\n    @param patch: a callback for patching (usually L{TestCase.patch}).\n\n    @return: the list that keeps track of the log files.\n    @rtype: C{list}\n    '
    logFiles = []
    oldFileLogObserver = logger.textFileLogObserver

    def observer(logFile, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        logFiles.append(logFile)
        return oldFileLogObserver(logFile, *args, **kwargs)
    patch(logger, 'textFileLogObserver', observer)
    return logFiles

def _setupSyslog(testCase):
    if False:
        i = 10
        return i + 15
    '\n    Make fake syslog, and return list to which prefix and then log\n    messages will be appended if it is used.\n    '
    logMessages = []

    class fakesyslogobserver:

        def __init__(self, prefix):
            if False:
                while True:
                    i = 10
            logMessages.append(prefix)

        def emit(self, eventDict):
            if False:
                for i in range(10):
                    print('nop')
            logMessages.append(eventDict)
    testCase.patch(syslog, 'SyslogObserver', fakesyslogobserver)
    return logMessages

class AppLoggerTests(TestCase):
    """
    Tests for L{app.AppLogger}.

    @ivar observers: list of observers installed during the tests.
    @type observers: C{list}
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        '\n        Override L{globaLogBeginner.beginLoggingTo} so that we can trace the\n        observers installed in C{self.observers}.\n        '
        self.observers = []

        def beginLoggingTo(observers):
            if False:
                i = 10
                return i + 15
            for observer in observers:
                self.observers.append(observer)
                globalLogPublisher.addObserver(observer)
        self.patch(globalLogBeginner, 'beginLoggingTo', beginLoggingTo)

    def tearDown(self):
        if False:
            while True:
                i = 10
        '\n        Remove all installed observers.\n        '
        for observer in self.observers:
            globalLogPublisher.removeObserver(observer)

    def _makeObserver(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make a new observer which captures all logs sent to it.\n\n        @return: An observer that stores all logs sent to it.\n        @rtype: Callable that implements L{ILogObserver}.\n        '

        @implementer(ILogObserver)
        class TestObserver:
            _logs = []

            def __call__(self, event):
                if False:
                    i = 10
                    return i + 15
                self._logs.append(event)
        return TestObserver()

    def _checkObserver(self, observer):
        if False:
            print('Hello World!')
        '\n        Ensure that initial C{twistd} logs are written to logs.\n\n        @param observer: The observer made by L{self._makeObserver).\n        '
        self.assertEqual(self.observers, [observer])
        self.assertIn('starting up', observer._logs[0]['log_format'])
        self.assertIn('reactor class', observer._logs[1]['log_format'])

    def test_start(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{app.AppLogger.start} calls L{globalLogBeginner.addObserver}, and then\n        writes some messages about twistd and the reactor.\n        '
        logger = app.AppLogger({})
        observer = self._makeObserver()
        logger._getLogObserver = lambda : observer
        logger.start(Componentized())
        self._checkObserver(observer)

    def test_startUsesApplicationLogObserver(self):
        if False:
            print('Hello World!')
        '\n        When the L{ILogObserver} component is available on the application,\n        that object will be used as the log observer instead of constructing a\n        new one.\n        '
        application = Componentized()
        observer = self._makeObserver()
        application.setComponent(ILogObserver, observer)
        logger = app.AppLogger({})
        logger.start(application)
        self._checkObserver(observer)

    def _setupConfiguredLogger(self, application, extraLogArgs={}, appLogger=app.AppLogger):
        if False:
            while True:
                i = 10
        '\n        Set up an AppLogger which exercises the C{logger} configuration option.\n\n        @type application: L{Componentized}\n        @param application: The L{Application} object to pass to\n            L{app.AppLogger.start}.\n        @type extraLogArgs: C{dict}\n        @param extraLogArgs: extra values to pass to AppLogger.\n        @type appLogger: L{AppLogger} class, or a subclass\n        @param appLogger: factory for L{AppLogger} instances.\n\n        @rtype: C{list}\n        @return: The logs accumulated by the log observer.\n        '
        observer = self._makeObserver()
        logArgs = {'logger': lambda : observer}
        logArgs.update(extraLogArgs)
        logger = appLogger(logArgs)
        logger.start(application)
        return observer

    def test_startUsesConfiguredLogObserver(self):
        if False:
            return 10
        '\n        When the C{logger} key is specified in the configuration dictionary\n        (i.e., when C{--logger} is passed to twistd), the initial log observer\n        will be the log observer returned from the callable which the value\n        refers to in FQPN form.\n        '
        application = Componentized()
        self._checkObserver(self._setupConfiguredLogger(application))

    def test_configuredLogObserverBeatsComponent(self):
        if False:
            print('Hello World!')
        '\n        C{--logger} takes precedence over a L{ILogObserver} component set on\n        Application.\n        '
        observer = self._makeObserver()
        application = Componentized()
        application.setComponent(ILogObserver, observer)
        self._checkObserver(self._setupConfiguredLogger(application))
        self.assertEqual(observer._logs, [])

    def test_configuredLogObserverBeatsLegacyComponent(self):
        if False:
            while True:
                i = 10
        '\n        C{--logger} takes precedence over a L{LegacyILogObserver} component\n        set on Application.\n        '
        nonlogs = []
        application = Componentized()
        application.setComponent(LegacyILogObserver, nonlogs.append)
        self._checkObserver(self._setupConfiguredLogger(application))
        self.assertEqual(nonlogs, [])

    def test_loggerComponentBeatsLegacyLoggerComponent(self):
        if False:
            print('Hello World!')
        '\n        A L{ILogObserver} takes precedence over a L{LegacyILogObserver}\n        component set on Application.\n        '
        nonlogs = []
        observer = self._makeObserver()
        application = Componentized()
        application.setComponent(ILogObserver, observer)
        application.setComponent(LegacyILogObserver, nonlogs.append)
        logger = app.AppLogger({})
        logger.start(application)
        self._checkObserver(observer)
        self.assertEqual(nonlogs, [])

    @skipIf(not _twistd_unix, 'twistd unix not available')
    @skipIf(not syslog, 'syslog not available')
    def test_configuredLogObserverBeatsSyslog(self):
        if False:
            print('Hello World!')
        '\n        C{--logger} takes precedence over a C{--syslog} command line\n        argument.\n        '
        logs = _setupSyslog(self)
        application = Componentized()
        self._checkObserver(self._setupConfiguredLogger(application, {'syslog': True}, UnixAppLogger))
        self.assertEqual(logs, [])

    def test_configuredLogObserverBeatsLogfile(self):
        if False:
            while True:
                i = 10
        '\n        C{--logger} takes precedence over a C{--logfile} command line\n        argument.\n        '
        application = Componentized()
        path = self.mktemp()
        self._checkObserver(self._setupConfiguredLogger(application, {'logfile': 'path'}))
        self.assertFalse(os.path.exists(path))

    def test_getLogObserverStdout(self):
        if False:
            while True:
                i = 10
        '\n        When logfile is empty or set to C{-}, L{app.AppLogger._getLogObserver}\n        returns a log observer pointing at C{sys.stdout}.\n        '
        logger = app.AppLogger({'logfile': '-'})
        logFiles = _patchTextFileLogObserver(self.patch)
        logger._getLogObserver()
        self.assertEqual(len(logFiles), 1)
        self.assertIs(logFiles[0], sys.stdout)
        logger = app.AppLogger({'logfile': ''})
        logger._getLogObserver()
        self.assertEqual(len(logFiles), 2)
        self.assertIs(logFiles[1], sys.stdout)

    def test_getLogObserverFile(self):
        if False:
            print('Hello World!')
        '\n        When passing the C{logfile} option, L{app.AppLogger._getLogObserver}\n        returns a log observer pointing at the specified path.\n        '
        logFiles = _patchTextFileLogObserver(self.patch)
        filename = self.mktemp()
        sut = app.AppLogger({'logfile': filename})
        observer = sut._getLogObserver()
        self.addCleanup(observer._outFile.close)
        self.assertEqual(len(logFiles), 1)
        self.assertEqual(logFiles[0].path, os.path.abspath(filename))

    def test_stop(self):
        if False:
            while True:
                i = 10
        "\n        L{app.AppLogger.stop} removes the observer created in C{start}, and\n        reinitialize its C{_observer} so that if C{stop} is called several\n        times it doesn't break.\n        "
        removed = []
        observer = object()

        def remove(observer):
            if False:
                i = 10
                return i + 15
            removed.append(observer)
        self.patch(globalLogPublisher, 'removeObserver', remove)
        logger = app.AppLogger({})
        logger._observer = observer
        logger.stop()
        self.assertEqual(removed, [observer])
        logger.stop()
        self.assertEqual(removed, [observer])
        self.assertIsNone(logger._observer)

    def test_legacyObservers(self):
        if False:
            while True:
                i = 10
        '\n        L{app.AppLogger} using a legacy logger observer still works, wrapping\n        it in a compat shim.\n        '
        logs = []
        logger = app.AppLogger({})

        @implementer(LegacyILogObserver)
        class LoggerObserver:
            """
            An observer which implements the legacy L{LegacyILogObserver}.
            """

            def __call__(self, x):
                if False:
                    print('Hello World!')
                '\n                Add C{x} to the logs list.\n                '
                logs.append(x)
        logger._observerFactory = lambda : LoggerObserver()
        logger.start(Componentized())
        self.assertIn('starting up', textFromEventDict(logs[0]))
        warnings = self.flushWarnings([self.test_legacyObservers])
        self.assertEqual(len(warnings), 0, warnings)

    def test_unmarkedObserversDeprecated(self):
        if False:
            while True:
                i = 10
        '\n        L{app.AppLogger} using a logger observer which does not implement\n        L{ILogObserver} or L{LegacyILogObserver} will be wrapped in a compat\n        shim and raise a L{DeprecationWarning}.\n        '
        logs = []
        logger = app.AppLogger({})
        logger._getLogObserver = lambda : logs.append
        logger.start(Componentized())
        self.assertIn('starting up', textFromEventDict(logs[0]))
        warnings = self.flushWarnings([self.test_unmarkedObserversDeprecated])
        self.assertEqual(warnings[0]['message'], 'Passing a logger factory which makes log observers which do not implement twisted.logger.ILogObserver or twisted.python.log.ILogObserver to twisted.application.app.AppLogger was deprecated in Twisted 16.2. Please use a factory that produces twisted.logger.ILogObserver (or the legacy twisted.python.log.ILogObserver) implementing objects instead.')
        self.assertEqual(len(warnings), 1, warnings)

@skipIf(not _twistd_unix, 'twistd unix not available')
class UnixAppLoggerTests(TestCase):
    """
    Tests for L{UnixAppLogger}.

    @ivar signals: list of signal handlers installed.
    @type signals: C{list}
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fake C{signal.signal} for not installing the handlers but saving them\n        in C{self.signals}.\n        '
        self.signals = []

        def fakeSignal(sig, f):
            if False:
                while True:
                    i = 10
            self.signals.append((sig, f))
        self.patch(signal, 'signal', fakeSignal)

    def test_getLogObserverStdout(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When non-daemonized and C{logfile} is empty or set to C{-},\n        L{UnixAppLogger._getLogObserver} returns a log observer pointing at\n        C{sys.stdout}.\n        '
        logFiles = _patchTextFileLogObserver(self.patch)
        logger = UnixAppLogger({'logfile': '-', 'nodaemon': True})
        logger._getLogObserver()
        self.assertEqual(len(logFiles), 1)
        self.assertIs(logFiles[0], sys.stdout)
        logger = UnixAppLogger({'logfile': '', 'nodaemon': True})
        logger._getLogObserver()
        self.assertEqual(len(logFiles), 2)
        self.assertIs(logFiles[1], sys.stdout)

    def test_getLogObserverStdoutDaemon(self):
        if False:
            return 10
        '\n        When daemonized and C{logfile} is set to C{-},\n        L{UnixAppLogger._getLogObserver} raises C{SystemExit}.\n        '
        logger = UnixAppLogger({'logfile': '-', 'nodaemon': False})
        error = self.assertRaises(SystemExit, logger._getLogObserver)
        self.assertEqual(str(error), 'Daemons cannot log to stdout, exiting!')

    def test_getLogObserverFile(self):
        if False:
            while True:
                i = 10
        '\n        When C{logfile} contains a file name, L{app.AppLogger._getLogObserver}\n        returns a log observer pointing at the specified path, and a signal\n        handler rotating the log is installed.\n        '
        logFiles = _patchTextFileLogObserver(self.patch)
        filename = self.mktemp()
        sut = UnixAppLogger({'logfile': filename})
        observer = sut._getLogObserver()
        self.addCleanup(observer._outFile.close)
        self.assertEqual(len(logFiles), 1)
        self.assertEqual(logFiles[0].path, os.path.abspath(filename))
        self.assertEqual(len(self.signals), 1)
        self.assertEqual(self.signals[0][0], signal.SIGUSR1)
        d = Deferred()

        def rotate():
            if False:
                print('Hello World!')
            d.callback(None)
        logFiles[0].rotate = rotate
        rotateLog = self.signals[0][1]
        rotateLog(None, None)
        return d

    def test_getLogObserverDontOverrideSignalHandler(self):
        if False:
            print('Hello World!')
        "\n        If a signal handler is already installed,\n        L{UnixAppLogger._getLogObserver} doesn't override it.\n        "

        def fakeGetSignal(sig):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(sig, signal.SIGUSR1)
            return object()
        self.patch(signal, 'getsignal', fakeGetSignal)
        filename = self.mktemp()
        sut = UnixAppLogger({'logfile': filename})
        observer = sut._getLogObserver()
        self.addCleanup(observer._outFile.close)
        self.assertEqual(self.signals, [])

    def test_getLogObserverDefaultFile(self):
        if False:
            print('Hello World!')
        '\n        When daemonized and C{logfile} is empty, the observer returned by\n        L{UnixAppLogger._getLogObserver} points at C{twistd.log} in the current\n        directory.\n        '
        logFiles = _patchTextFileLogObserver(self.patch)
        logger = UnixAppLogger({'logfile': '', 'nodaemon': False})
        observer = logger._getLogObserver()
        self.addCleanup(observer._outFile.close)
        self.assertEqual(len(logFiles), 1)
        self.assertEqual(logFiles[0].path, os.path.abspath('twistd.log'))

    @skipIf(not _twistd_unix, 'twistd unix not available')
    def test_getLogObserverSyslog(self):
        if False:
            print('Hello World!')
        '\n        If C{syslog} is set to C{True}, L{UnixAppLogger._getLogObserver} starts\n        a L{syslog.SyslogObserver} with given C{prefix}.\n        '
        logs = _setupSyslog(self)
        logger = UnixAppLogger({'syslog': True, 'prefix': 'test-prefix'})
        observer = logger._getLogObserver()
        self.assertEqual(logs, ['test-prefix'])
        observer({'a': 'b'})
        self.assertEqual(logs, ['test-prefix', {'a': 'b'}])

@skipIf(not _twistd_unix, 'twistd unix support not available')
class DaemonizeTests(TestCase):
    """
    Tests for L{_twistd_unix.UnixApplicationRunner} daemonization.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.mockos = MockOS()
        self.config = twistd.ServerOptions()
        self.patch(_twistd_unix, 'os', self.mockos)
        self.runner = _twistd_unix.UnixApplicationRunner(self.config)
        self.runner.application = service.Application('Hi!')
        self.runner.oldstdout = sys.stdout
        self.runner.oldstderr = sys.stderr
        self.runner.startReactor = lambda *args: None

    def test_success(self):
        if False:
            return 10
        '\n        When double fork succeeded in C{daemonize}, the child process writes\n        B{0} to the status pipe.\n        '
        with AlternateReactor(FakeDaemonizingReactor()):
            self.runner.postApplication()
        self.assertEqual(self.mockos.actions, [('chdir', '.'), ('umask', 63), ('fork', True), 'setsid', ('fork', True), ('write', -2, b'0'), ('unlink', 'twistd.pid')])
        self.assertEqual(self.mockos.closed, [-3, -2])

    def test_successInParent(self):
        if False:
            return 10
        '\n        The parent process initiating the C{daemonize} call reads data from the\n        status pipe and then exit the process.\n        '
        self.mockos.child = False
        self.mockos.readData = b'0'
        with AlternateReactor(FakeDaemonizingReactor()):
            self.assertRaises(SystemError, self.runner.postApplication)
        self.assertEqual(self.mockos.actions, [('chdir', '.'), ('umask', 63), ('fork', True), ('read', -1, 100), ('exit', 0), ('unlink', 'twistd.pid')])
        self.assertEqual(self.mockos.closed, [-1])

    def test_successEINTR(self):
        if False:
            while True:
                i = 10
        '\n        If the C{os.write} call to the status pipe raises an B{EINTR} error,\n        the process child retries to write.\n        '
        written = []

        def raisingWrite(fd, data):
            if False:
                i = 10
                return i + 15
            written.append((fd, data))
            if len(written) == 1:
                raise OSError(errno.EINTR)
        self.mockos.write = raisingWrite
        with AlternateReactor(FakeDaemonizingReactor()):
            self.runner.postApplication()
        self.assertEqual(self.mockos.actions, [('chdir', '.'), ('umask', 63), ('fork', True), 'setsid', ('fork', True), ('unlink', 'twistd.pid')])
        self.assertEqual(self.mockos.closed, [-3, -2])
        self.assertEqual([(-2, b'0'), (-2, b'0')], written)

    def test_successInParentEINTR(self):
        if False:
            print('Hello World!')
        '\n        If the C{os.read} call on the status pipe raises an B{EINTR} error, the\n        parent child retries to read.\n        '
        read = []

        def raisingRead(fd, size):
            if False:
                return 10
            read.append((fd, size))
            if len(read) == 1:
                raise OSError(errno.EINTR)
            return b'0'
        self.mockos.read = raisingRead
        self.mockos.child = False
        with AlternateReactor(FakeDaemonizingReactor()):
            self.assertRaises(SystemError, self.runner.postApplication)
        self.assertEqual(self.mockos.actions, [('chdir', '.'), ('umask', 63), ('fork', True), ('exit', 0), ('unlink', 'twistd.pid')])
        self.assertEqual(self.mockos.closed, [-1])
        self.assertEqual([(-1, 100), (-1, 100)], read)

    def assertErrorWritten(self, raised, reported):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assert L{UnixApplicationRunner.postApplication} writes\n        C{reported} to its status pipe if the service raises an\n        exception whose message is C{raised}.\n        '

        class FakeService(service.Service):

            def startService(self):
                if False:
                    return 10
                raise RuntimeError(raised)
        errorService = FakeService()
        errorService.setServiceParent(self.runner.application)
        with AlternateReactor(FakeDaemonizingReactor()):
            self.assertRaises(RuntimeError, self.runner.postApplication)
        self.assertEqual(self.mockos.actions, [('chdir', '.'), ('umask', 63), ('fork', True), 'setsid', ('fork', True), ('write', -2, reported), ('unlink', 'twistd.pid')])
        self.assertEqual(self.mockos.closed, [-3, -2])

    def test_error(self):
        if False:
            i = 10
            return i + 15
        '\n        If an error happens during daemonization, the child process writes the\n        exception error to the status pipe.\n        '
        self.assertErrorWritten(raised='Something is wrong', reported=b'1 RuntimeError: Something is wrong')

    def test_unicodeError(self):
        if False:
            return 10
        "\n        If an error happens during daemonization, and that error's\n        message is Unicode, the child encodes the message as ascii\n        with backslash Unicode code points.\n        "
        self.assertErrorWritten(raised='•', reported=b'1 RuntimeError: \\u2022')

    def assertErrorInParentBehavior(self, readData, errorMessage, mockOSActions):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make L{os.read} appear to return C{readData}, and assert that\n        L{UnixApplicationRunner.postApplication} writes\n        C{errorMessage} to standard error and executes the calls\n        against L{os} functions specified in C{mockOSActions}.\n        '
        self.mockos.child = False
        self.mockos.readData = readData
        errorIO = StringIO()
        self.patch(sys, '__stderr__', errorIO)
        with AlternateReactor(FakeDaemonizingReactor()):
            self.assertRaises(SystemError, self.runner.postApplication)
        self.assertEqual(errorIO.getvalue(), errorMessage)
        self.assertEqual(self.mockos.actions, mockOSActions)
        self.assertEqual(self.mockos.closed, [-1])

    def test_errorInParent(self):
        if False:
            print('Hello World!')
        '\n        When the child writes an error message to the status pipe\n        during daemonization, the parent writes the repr of the\n        message to C{stderr} and exits with non-zero status code.\n        '
        self.assertErrorInParentBehavior(readData=b'1 Exception: An identified error', errorMessage="An error has occurred: b'Exception: An identified error'\nPlease look at log file for more information.\n", mockOSActions=[('chdir', '.'), ('umask', 63), ('fork', True), ('read', -1, 100), ('exit', 1), ('unlink', 'twistd.pid')])

    def test_nonASCIIErrorInParent(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When the child writes a non-ASCII error message to the status\n        pipe during daemonization, the parent writes the repr of the\n        message to C{stderr} and exits with a non-zero status code.\n        '
        self.assertErrorInParentBehavior(readData=b'1 Exception: \xff', errorMessage="An error has occurred: b'Exception: \\xff'\nPlease look at log file for more information.\n", mockOSActions=[('chdir', '.'), ('umask', 63), ('fork', True), ('read', -1, 100), ('exit', 1), ('unlink', 'twistd.pid')])

    def test_errorInParentWithTruncatedUnicode(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When the child writes a non-ASCII error message to the status\n        pipe during daemonization, and that message is too longer, the\n        parent writes the repr of the truncated message to C{stderr}\n        and exits with a non-zero status code.\n        '
        truncatedMessage = b'1 RuntimeError: ' + b'\\u2022' * 14
        reportedMessage = "b'RuntimeError: {}'".format('\\\\u2022' * 14)
        self.assertErrorInParentBehavior(readData=truncatedMessage, errorMessage='An error has occurred: {}\nPlease look at log file for more information.\n'.format(reportedMessage), mockOSActions=[('chdir', '.'), ('umask', 63), ('fork', True), ('read', -1, 100), ('exit', 1), ('unlink', 'twistd.pid')])

    def test_errorMessageTruncated(self):
        if False:
            print('Hello World!')
        "\n        If an error occurs during daemonization and its message is too\n        long, it's truncated by the child.\n        "
        self.assertErrorWritten(raised='x' * 200, reported=b'1 RuntimeError: ' + b'x' * 84)

    def test_unicodeErrorMessageTruncated(self):
        if False:
            return 10
        "\n        If an error occurs during daemonization and its message is\n        unicode and too long, it's truncated by the child, even if\n        this splits a unicode escape sequence.\n        "
        self.assertErrorWritten(raised='•' * 30, reported=b'1 RuntimeError: ' + b'\\u2022' * 14)

    def test_hooksCalled(self):
        if False:
            while True:
                i = 10
        '\n        C{daemonize} indeed calls L{IReactorDaemonize.beforeDaemonize} and\n        L{IReactorDaemonize.afterDaemonize} if the reactor implements\n        L{IReactorDaemonize}.\n        '
        reactor = FakeDaemonizingReactor()
        self.runner.daemonize(reactor)
        self.assertTrue(reactor._beforeDaemonizeCalled)
        self.assertTrue(reactor._afterDaemonizeCalled)

    def test_hooksNotCalled(self):
        if False:
            while True:
                i = 10
        '\n        C{daemonize} does NOT call L{IReactorDaemonize.beforeDaemonize} or\n        L{IReactorDaemonize.afterDaemonize} if the reactor does NOT implement\n        L{IReactorDaemonize}.\n        '
        reactor = FakeNonDaemonizingReactor()
        self.runner.daemonize(reactor)
        self.assertFalse(reactor._beforeDaemonizeCalled)
        self.assertFalse(reactor._afterDaemonizeCalled)

@implementer(_ISupportsExitSignalCapturing)
class SignalCapturingMemoryReactor(MemoryReactor):
    """
    MemoryReactor that implements the _ISupportsExitSignalCapturing interface,
    all other operations identical to MemoryReactor.
    """

    @property
    def _exitSignal(self):
        if False:
            for i in range(10):
                print('nop')
        return self._val

    @_exitSignal.setter
    def _exitSignal(self, val):
        if False:
            i = 10
            return i + 15
        self._val = val

class StubApplicationRunnerWithSignal(twistd._SomeApplicationRunner):
    """
    An application runner that uses a SignalCapturingMemoryReactor and
    has a _signalValue attribute that it will set in the reactor.

    @ivar _signalValue: The signal value to set on the reactor's _exitSignal
        attribute.
    """
    loggerFactory = CrippledAppLogger

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        self._signalValue = None

    def preApplication(self):
        if False:
            i = 10
            return i + 15
        '\n        Does nothing.\n        '

    def postApplication(self):
        if False:
            print('Hello World!')
        '\n        Instantiate a SignalCapturingMemoryReactor and start it\n        in the runner.\n        '
        reactor = SignalCapturingMemoryReactor()
        reactor._exitSignal = self._signalValue
        self.startReactor(reactor, sys.stdout, sys.stderr)

def stubApplicationRunnerFactoryCreator(signum):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a factory function to instantiate a\n    StubApplicationRunnerWithSignal that will report signum as the captured\n    signal..\n\n    @param signum: The integer signal number or None\n    @type signum: C{int} or C{None}\n\n    @return: A factory function to create stub runners.\n    @rtype: stubApplicationRunnerFactory\n    '

    def stubApplicationRunnerFactory(config):
        if False:
            print('Hello World!')
        '\n        Create a StubApplicationRunnerWithSignal using a reactor that\n        implements _ISupportsExitSignalCapturing and whose _exitSignal\n        attribute is set to signum.\n\n        @param config: The runner configuration, platform dependent.\n        @type config: L{twisted.scripts.twistd.ServerOptions}\n\n        @return: A runner to use for the test.\n        @rtype: twisted.test.test_twistd.StubApplicationRunnerWithSignal\n        '
        runner = StubApplicationRunnerWithSignal(config)
        runner._signalValue = signum
        return runner
    return stubApplicationRunnerFactory

class ExitWithSignalTests(TestCase):
    """
    Tests for L{twisted.application.app._exitWithSignal}.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up the server options and a fake for use by test cases.\n        '
        self.config = twistd.ServerOptions()
        self.config.loadedPlugins = {'test_command': MockServiceMaker()}
        self.config.subOptions = object()
        self.config.subCommand = 'test_command'
        self.fakeKillArgs = [None, None]

        def fakeKill(pid, sig):
            if False:
                return 10
            '\n            Fake method to capture arguments passed to os.kill.\n\n            @param pid: The pid of the process being killed.\n\n            @param sig: The signal sent to the process.\n            '
            self.fakeKillArgs[0] = pid
            self.fakeKillArgs[1] = sig
        self.patch(os, 'kill', fakeKill)

    def test_exitWithSignal(self):
        if False:
            print('Hello World!')
        '\n        exitWithSignal replaces the existing signal handler with the default\n        handler and sends the replaced signal to the current process.\n        '
        fakeSignalArgs = [None, None]

        def fake_signal(sig, handler):
            if False:
                while True:
                    i = 10
            fakeSignalArgs[0] = sig
            fakeSignalArgs[1] = handler
        self.patch(signal, 'signal', fake_signal)
        app._exitWithSignal(signal.SIGINT)
        self.assertEquals(fakeSignalArgs[0], signal.SIGINT)
        self.assertEquals(fakeSignalArgs[1], signal.SIG_DFL)
        self.assertEquals(self.fakeKillArgs[0], os.getpid())
        self.assertEquals(self.fakeKillArgs[1], signal.SIGINT)

    def test_normalExit(self):
        if False:
            i = 10
            return i + 15
        '\n        _exitWithSignal is not called if the runner does not exit with a\n        signal.\n        '
        self.patch(twistd, '_SomeApplicationRunner', stubApplicationRunnerFactoryCreator(None))
        twistd.runApp(self.config)
        self.assertIsNone(self.fakeKillArgs[0])
        self.assertIsNone(self.fakeKillArgs[1])

    def test_runnerExitsWithSignal(self):
        if False:
            print('Hello World!')
        '\n        _exitWithSignal is called when the runner exits with a signal.\n        '
        self.patch(twistd, '_SomeApplicationRunner', stubApplicationRunnerFactoryCreator(signal.SIGINT))
        twistd.runApp(self.config)
        self.assertEquals(self.fakeKillArgs[0], os.getpid())
        self.assertEquals(self.fakeKillArgs[1], signal.SIGINT)