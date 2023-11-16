import copy
from twisted.internet import defer
from twisted.python import log
from buildbot import config
from buildbot.process import buildstep
from buildbot.process import results
from buildbot.warnings import warn_deprecated

class ShellArg(results.ResultComputingConfigMixin):
    publicAttributes = results.ResultComputingConfigMixin.resultConfig + ['command', 'logname']

    def __init__(self, command=None, logname=None, logfile=None, **kwargs):
        if False:
            print('Hello World!')
        name = self.__class__.__name__
        if command is None:
            config.error(f"the 'command' parameter of {name} must not be None")
        self.command = command
        self.logname = logname
        if logfile is not None:
            warn_deprecated('2.10.0', '{}: logfile is deprecated, use logname')
            if self.logname is not None:
                config.error(("{}: the 'logfile' parameter must not be specified when 'logname' " + 'is set').format(name))
            self.logname = logfile
        for (k, v) in kwargs.items():
            if k not in self.resultConfig:
                config.error(f"the parameter '{k}' is not handled by ShellArg")
            setattr(self, k, v)

    def validateAttributes(self):
        if False:
            return 10
        if not isinstance(self.command, (str, list)):
            config.error(f'{self.command} is an invalid command, it must be a string or a list')
        if isinstance(self.command, list):
            if not all((isinstance(x, str) for x in self.command)):
                config.error(f'{self.command} must only have strings in it')
        runConfParams = [(p_attr, getattr(self, p_attr)) for p_attr in self.resultConfig]
        not_bool = [(p_attr, p_val) for (p_attr, p_val) in runConfParams if not isinstance(p_val, bool)]
        if not_bool:
            config.error(f'{repr(not_bool)} must be booleans')

    @defer.inlineCallbacks
    def getRenderingFor(self, build):
        if False:
            return 10
        rv = copy.copy(self)
        for p_attr in self.publicAttributes:
            res = (yield build.render(getattr(self, p_attr)))
            setattr(rv, p_attr, res)
        return rv

class ShellSequence(buildstep.ShellMixin, buildstep.BuildStep):
    last_command = None
    renderables = ['commands']

    def __init__(self, commands=None, **kwargs):
        if False:
            print('Hello World!')
        self.commands = commands
        kwargs = self.setupShellMixin(kwargs, prohibitArgs=['command'])
        super().__init__(**kwargs)

    def shouldRunTheCommand(self, cmd):
        if False:
            for i in range(10):
                print('nop')
        return bool(cmd)

    def getFinalState(self):
        if False:
            for i in range(10):
                print('nop')
        return self.describe(True)

    @defer.inlineCallbacks
    def runShellSequence(self, commands):
        if False:
            while True:
                i = 10
        terminate = False
        if commands is None:
            log.msg('After rendering, ShellSequence `commands` is None')
            return results.EXCEPTION
        overall_result = results.SUCCESS
        for arg in commands:
            if not isinstance(arg, ShellArg):
                log.msg('After rendering, ShellSequence `commands` list contains something that is not a ShellArg')
                return results.EXCEPTION
            try:
                arg.validateAttributes()
            except config.ConfigErrors as e:
                log.msg(f'After rendering, ShellSequence `commands` is invalid: {e}')
                return results.EXCEPTION
            command = arg.command
            if not self.shouldRunTheCommand(command):
                continue
            self.last_command = command
            cmd = (yield self.makeRemoteShellCommand(command=command, stdioLogName=arg.logname))
            yield self.runCommand(cmd)
            (overall_result, terminate) = results.computeResultAndTermination(arg, cmd.results(), overall_result)
            if terminate:
                break
        return overall_result

    def run(self):
        if False:
            print('Hello World!')
        return self.runShellSequence(self.commands)