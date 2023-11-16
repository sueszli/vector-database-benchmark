"""
Run a Twisted application.
"""
import sys
from typing import Sequence
from twisted.application.app import _exitWithSignal
from twisted.internet.interfaces import IReactorCore, _ISupportsExitSignalCapturing
from twisted.python.usage import Options, UsageError
from ..runner._exit import ExitStatus, exit
from ..runner._runner import Runner
from ..service import Application, IService, IServiceMaker
from ._options import TwistOptions

class Twist:
    """
    Run a Twisted application.
    """

    @staticmethod
    def options(argv: Sequence[str]) -> TwistOptions:
        if False:
            i = 10
            return i + 15
        '\n        Parse command line options.\n\n        @param argv: Command line arguments.\n        @return: The parsed options.\n        '
        options = TwistOptions()
        try:
            options.parseOptions(argv[1:])
        except UsageError as e:
            exit(ExitStatus.EX_USAGE, f'Error: {e}\n\n{options}')
        return options

    @staticmethod
    def service(plugin: IServiceMaker, options: Options) -> IService:
        if False:
            print('Hello World!')
        '\n        Create the application service.\n\n        @param plugin: The name of the plugin that implements the service\n            application to run.\n        @param options: Options to pass to the application.\n        @return: The created application service.\n        '
        service = plugin.makeService(options)
        application = Application(plugin.tapname)
        service.setServiceParent(application)
        return IService(application)

    @staticmethod
    def startService(reactor: IReactorCore, service: IService) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Start the application service.\n\n        @param reactor: The reactor to run the service with.\n        @param service: The application service to run.\n        '
        service.startService()
        reactor.addSystemEventTrigger('before', 'shutdown', service.stopService)

    @staticmethod
    def run(twistOptions: TwistOptions) -> None:
        if False:
            print('Hello World!')
        '\n        Run the application service.\n\n        @param twistOptions: Command line options to convert to runner\n            arguments.\n        '
        runner = Runner(reactor=twistOptions['reactor'], defaultLogLevel=twistOptions['logLevel'], logFile=twistOptions['logFile'], fileLogObserverFactory=twistOptions['fileLogObserverFactory'])
        runner.run()
        reactor = twistOptions['reactor']
        if _ISupportsExitSignalCapturing.providedBy(reactor):
            if reactor._exitSignal is not None:
                _exitWithSignal(reactor._exitSignal)

    @classmethod
    def main(cls, argv: Sequence[str]=sys.argv) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Executable entry point for L{Twist}.\n        Processes options and run a twisted reactor with a service.\n\n        @param argv: Command line arguments.\n        @type argv: L{list}\n        '
        options = cls.options(argv)
        reactor = options['reactor']
        subCommand = options.subCommand
        assert subCommand is not None
        service = cls.service(plugin=options.plugins[subCommand], options=options.subOptions)
        cls.startService(reactor, service)
        cls.run(options)