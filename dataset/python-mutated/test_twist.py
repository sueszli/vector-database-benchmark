"""
Tests for L{twisted.application.twist._twist}.
"""
from sys import stdout
from typing import Any, Dict, List
import twisted.trial.unittest
from twisted.internet.interfaces import IReactorCore
from twisted.internet.testing import MemoryReactor
from twisted.logger import LogLevel, jsonFileLogObserver
from twisted.test.test_twistd import SignalCapturingMemoryReactor
from ...runner._exit import ExitStatus
from ...runner._runner import Runner
from ...runner.test.test_runner import DummyExit
from ...service import IService, MultiService
from ...twist import _twist
from .._options import TwistOptions
from .._twist import Twist

class TwistTests(twisted.trial.unittest.TestCase):
    """
    Tests for L{Twist}.
    """

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.patchInstallReactor()

    def patchExit(self) -> None:
        if False:
            print('Hello World!')
        '\n        Patch L{_twist.exit} so we can capture usage and prevent actual exits.\n        '
        self.exit = DummyExit()
        self.patch(_twist, 'exit', self.exit)

    def patchInstallReactor(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Patch C{_options.installReactor} so we can capture usage and prevent\n        actual installs.\n        '
        self.installedReactors: Dict[str, IReactorCore] = {}

        def installReactor(_: TwistOptions, name: str) -> IReactorCore:
            if False:
                i = 10
                return i + 15
            reactor = MemoryReactor()
            self.installedReactors[name] = reactor
            return reactor
        self.patch(TwistOptions, 'installReactor', installReactor)

    def patchStartService(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Patch L{MultiService.startService} so we can capture usage and prevent\n        actual starts.\n        '
        self.serviceStarts: List[IService] = []

        def startService(service: IService) -> None:
            if False:
                print('Hello World!')
            self.serviceStarts.append(service)
        self.patch(MultiService, 'startService', startService)

    def test_optionsValidArguments(self) -> None:
        if False:
            return 10
        '\n        L{Twist.options} given valid arguments returns options.\n        '
        options = Twist.options(['twist', 'web'])
        self.assertIsInstance(options, TwistOptions)

    def test_optionsInvalidArguments(self) -> None:
        if False:
            return 10
        '\n        L{Twist.options} given invalid arguments exits with\n        L{ExitStatus.EX_USAGE} and an error/usage message.\n        '
        self.patchExit()
        Twist.options(['twist', '--bogus-bagels'])
        self.assertIdentical(self.exit.status, ExitStatus.EX_USAGE)
        self.assertIsNotNone(self.exit.message)
        self.assertTrue(self.exit.message.startswith('Error: '))
        self.assertTrue(self.exit.message.endswith(f'\n\n{TwistOptions()}'))

    def test_service(self) -> None:
        if False:
            while True:
                i = 10
        '\n        L{Twist.service} returns an L{IService}.\n        '
        options = Twist.options(['twist', 'web'])
        service = Twist.service(options.plugins['web'], options.subOptions)
        self.assertTrue(IService.providedBy(service))

    def test_startService(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{Twist.startService} starts the service and registers a trigger to\n        stop the service when the reactor shuts down.\n        '
        options = Twist.options(['twist', 'web'])
        reactor = options['reactor']
        subCommand = options.subCommand
        assert subCommand is not None
        service = Twist.service(plugin=options.plugins[subCommand], options=options.subOptions)
        self.patchStartService()
        Twist.startService(reactor, service)
        self.assertEqual(self.serviceStarts, [service])
        self.assertEqual(reactor.triggers['before']['shutdown'], [(service.stopService, (), {})])

    def test_run(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{Twist.run} runs the runner with arguments corresponding to the given\n        options.\n        '
        argsSeen = []
        self.patch(Runner, '__init__', lambda self, **args: argsSeen.append(args))
        self.patch(Runner, 'run', lambda self: None)
        twistOptions = Twist.options(['twist', '--reactor=default', '--log-format=json', 'web'])
        Twist.run(twistOptions)
        self.assertEqual(len(argsSeen), 1)
        self.assertEqual(argsSeen[0], dict(reactor=self.installedReactors['default'], defaultLogLevel=LogLevel.info, logFile=stdout, fileLogObserverFactory=jsonFileLogObserver))

    def test_main(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{Twist.main} runs the runner with arguments corresponding to the given\n        command line arguments.\n        '
        self.patchStartService()
        runners = []

        class Runner:

            def __init__(self, **kwargs: Any) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.args = kwargs
                self.runs = 0
                runners.append(self)

            def run(self) -> None:
                if False:
                    i = 10
                    return i + 15
                self.runs += 1
        self.patch(_twist, 'Runner', Runner)
        Twist.main(['twist', '--reactor=default', '--log-format=json', 'web'])
        self.assertEqual(len(self.serviceStarts), 1)
        self.assertEqual(len(runners), 1)
        self.assertEqual(runners[0].args, dict(reactor=self.installedReactors['default'], defaultLogLevel=LogLevel.info, logFile=stdout, fileLogObserverFactory=jsonFileLogObserver))
        self.assertEqual(runners[0].runs, 1)

class TwistExitTests(twisted.trial.unittest.TestCase):
    """
    Tests to verify that the Twist script takes the expected actions related
    to signals and the reactor.
    """

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.exitWithSignalCalled = False

        def fakeExitWithSignal(sig: int) -> None:
            if False:
                i = 10
                return i + 15
            '\n            Fake to capture whether L{twisted.application._exitWithSignal\n            was called.\n\n            @param sig: Signal value\n            @type sig: C{int}\n            '
            self.exitWithSignalCalled = True
        self.patch(_twist, '_exitWithSignal', fakeExitWithSignal)

        def startLogging(_: Runner) -> None:
            if False:
                print('Hello World!')
            '\n            Prevent Runner from adding new log observers or other\n            tests outside this module will fail.\n\n            @param _: Unused self param\n            '
        self.patch(Runner, 'startLogging', startLogging)

    def test_twistReactorDoesntExitWithSignal(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        _exitWithSignal is not called if the reactor's _exitSignal attribute\n        is zero.\n        "
        reactor = SignalCapturingMemoryReactor()
        reactor._exitSignal = None
        options = TwistOptions()
        options['reactor'] = reactor
        options['fileLogObserverFactory'] = jsonFileLogObserver
        Twist.run(options)
        self.assertFalse(self.exitWithSignalCalled)

    def test_twistReactorHasNoExitSignalAttr(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        _exitWithSignal is not called if the runner's reactor does not\n        implement L{twisted.internet.interfaces._ISupportsExitSignalCapturing}\n        "
        reactor = MemoryReactor()
        options = TwistOptions()
        options['reactor'] = reactor
        options['fileLogObserverFactory'] = jsonFileLogObserver
        Twist.run(options)
        self.assertFalse(self.exitWithSignalCalled)

    def test_twistReactorExitsWithSignal(self) -> None:
        if False:
            while True:
                i = 10
        "\n        _exitWithSignal is called if the runner's reactor exits due\n        to a signal.\n        "
        reactor = SignalCapturingMemoryReactor()
        reactor._exitSignal = 2
        options = TwistOptions()
        options['reactor'] = reactor
        options['fileLogObserverFactory'] = jsonFileLogObserver
        Twist.run(options)
        self.assertTrue(self.exitWithSignalCalled)