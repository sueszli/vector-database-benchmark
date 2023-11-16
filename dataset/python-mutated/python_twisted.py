"""
BuildSteps that are specific to the Twisted source tree
"""
import re
from twisted.internet import defer
from twisted.python import log
from buildbot import util
from buildbot.process import buildstep
from buildbot.process import logobserver
from buildbot.process.results import FAILURE
from buildbot.process.results import SKIPPED
from buildbot.process.results import SUCCESS
from buildbot.process.results import WARNINGS
from buildbot.steps import shell

class HLint(buildstep.ShellMixin, buildstep.BuildStep):
    """I run a 'lint' checker over a set of .xhtml files. Any deviations
    from recommended style is flagged and put in the output log.

    This step looks at .changes in the parent Build to extract a list of
    Lore XHTML files to check."""
    name = 'hlint'
    description = 'running hlint'
    descriptionDone = 'hlint'
    warnOnWarnings = True
    warnOnFailure = True
    warnings = 0

    def __init__(self, python=None, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs = self.setupShellMixin(kwargs, prohibitArgs=['command'])
        super().__init__(**kwargs)
        self.python = python
        self.warningLines = []
        self.addLogObserver('stdio', logobserver.LineConsumerLogObserver(self.logConsumer))

    @defer.inlineCallbacks
    def run(self):
        if False:
            i = 10
            return i + 15
        html_files = set()
        for f in self.build.allFiles():
            if f.endswith('.xhtml') and (not f.startswith('sandbox/')):
                html_files.add(f)
        hlintTargets = sorted(list(html_files))
        if not hlintTargets:
            return SKIPPED
        self.hlintFiles = hlintTargets
        command = []
        if self.python:
            command.append(self.python)
        command += ['bin/lore', '-p', '--output', 'lint'] + self.hlintFiles
        cmd = (yield self.makeRemoteShellCommand(command=command))
        yield self.runCommand(cmd)
        stdio_log = (yield self.getLog('stdio'))
        yield stdio_log.finish()
        yield self.addCompleteLog('warnings', '\n'.join(self.warningLines))
        yield self.addCompleteLog('files', '\n'.join(self.hlintFiles) + '\n')
        if cmd.didFail():
            return FAILURE
        self.descriptionDone = f"{self.warnings} hlin{self.warnings == 1 and 't' or 'ts'}"
        if self.warnings:
            return WARNINGS
        return SUCCESS

    def logConsumer(self):
        if False:
            return 10
        while True:
            (_, line) = (yield)
            if ':' in line:
                self.warnings += 1
                self.warningLines.append(line)

class TrialTestCaseCounter(logobserver.LogLineObserver):
    _line_re = re.compile('^(?:Doctest: )?([\\w\\.]+) \\.\\.\\. \\[([^\\]]+)\\]$')

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.numTests = 0
        self.finished = False
        self.counts = {'total': None, 'failures': 0, 'errors': 0, 'skips': 0, 'expectedFailures': 0, 'unexpectedSuccesses': 0}

    def outLineReceived(self, line):
        if False:
            print('Hello World!')
        if line.startswith('=' * 40):
            self.finished = True
        if not self.finished:
            m = self._line_re.search(line.strip())
            if m:
                m.groups()
                self.numTests += 1
                self.step.setProgress('tests', self.numTests)
        out = re.search('Ran (\\d+) tests', line)
        if out:
            self.counts['total'] = int(out.group(1))
        if line.startswith('OK') or line.startswith('FAILED ') or line.startswith('PASSED'):
            out = re.search('failures=(\\d+)', line)
            if out:
                self.counts['failures'] = int(out.group(1))
            out = re.search('errors=(\\d+)', line)
            if out:
                self.counts['errors'] = int(out.group(1))
            out = re.search('skips=(\\d+)', line)
            if out:
                self.counts['skips'] = int(out.group(1))
            out = re.search('expectedFailures=(\\d+)', line)
            if out:
                self.counts['expectedFailures'] = int(out.group(1))
            out = re.search('unexpectedSuccesses=(\\d+)', line)
            if out:
                self.counts['unexpectedSuccesses'] = int(out.group(1))
            out = re.search('successes=(\\d+)', line)
            if out:
                self.counts['successes'] = int(out.group(1))
UNSPECIFIED = ()

class Trial(buildstep.ShellMixin, buildstep.BuildStep):
    """
    There are some class attributes which may be usefully overridden
    by subclasses. 'trialMode' and 'trialArgs' can influence the trial
    command line.
    """
    name = 'trial'
    progressMetrics = ('output', 'tests', 'test.log')
    logfiles = {'test.log': '_trial_temp/test.log'}
    renderables = ['tests', 'jobs']
    flunkOnFailure = True
    python = None
    trial = 'trial'
    trialMode = ['--reporter=bwverbose']
    trialArgs = []
    jobs = None
    testpath = UNSPECIFIED
    testChanges = False
    recurse = False
    reactor = None
    randomly = False
    tests = None
    description = 'testing'
    descriptionDone = 'tests'

    def __init__(self, reactor=UNSPECIFIED, python=None, trial=None, testpath=UNSPECIFIED, tests=None, testChanges=None, recurse=None, randomly=None, trialMode=None, trialArgs=None, jobs=None, **kwargs):
        if False:
            print('Hello World!')
        kwargs = self.setupShellMixin(kwargs, prohibitArgs=['command'])
        super().__init__(**kwargs)
        if python:
            self.python = python
        if self.python is not None:
            if isinstance(self.python, str):
                self.python = [self.python]
            for s in self.python:
                if ' ' in s:
                    log.msg("python= component '%s' has spaces")
                    log.msg("To add -Wall, use python=['python', '-Wall']")
                    why = 'python= value has spaces, probably an error'
                    raise ValueError(why)
        if trial:
            self.trial = trial
        if ' ' in self.trial:
            raise ValueError('trial= value has spaces')
        if trialMode is not None:
            self.trialMode = trialMode
        if trialArgs is not None:
            self.trialArgs = trialArgs
        if jobs is not None:
            self.jobs = jobs
        if testpath is not UNSPECIFIED:
            self.testpath = testpath
        if self.testpath is UNSPECIFIED:
            raise ValueError('You must specify testpath= (it can be None)')
        assert isinstance(self.testpath, str) or self.testpath is None
        if reactor is not UNSPECIFIED:
            self.reactor = reactor
        if tests is not None:
            self.tests = tests
        if isinstance(self.tests, str):
            self.tests = [self.tests]
        if testChanges is not None:
            self.testChanges = testChanges
        if not self.testChanges and self.tests is None:
            raise ValueError('Must either set testChanges= or provide tests=')
        if recurse is not None:
            self.recurse = recurse
        if randomly is not None:
            self.randomly = randomly
        if self.reactor:
            self.description = f'testing ({self.reactor})'
        self.observer = TrialTestCaseCounter()
        self.addLogObserver('stdio', self.observer)
        self.addLogObserver('stdio', logobserver.LineConsumerLogObserver(self.logConsumer))
        self.problems = []
        self.warnings = {}
        self.text = 'running'

    def setup_python_path(self):
        if False:
            i = 10
            return i + 15
        if self.testpath is None:
            return
        ppath = self.env.get('PYTHONPATH', self.testpath)
        if isinstance(ppath, str):
            ppath = [ppath]
        if self.testpath not in ppath:
            ppath.insert(0, self.testpath)
        self.env['PYTHONPATH'] = ppath

    @defer.inlineCallbacks
    def run(self):
        if False:
            print('Hello World!')
        output_observer = logobserver.OutputProgressObserver('test.log')
        command = []
        if self.python:
            command.extend(self.python)
        command.append(self.trial)
        command.extend(self.trialMode)
        if self.recurse:
            command.append('--recurse')
        if self.reactor:
            command.append(f'--reactor={self.reactor}')
        if self.randomly:
            command.append('--random=0')
        command.extend(self.trialArgs)
        if self.jobs is not None:
            self.jobs = int(self.jobs)
            command.append(f'--jobs={self.jobs}')
            self.logfiles = {}
            for i in range(self.jobs):
                self.logfiles[f'test.{i}.log'] = f'_trial_temp/{i}/test.log'
                self.logfiles[f'err.{i}.log'] = f'_trial_temp/{i}/err.log'
                self.logfiles[f'out.{i}.log'] = f'_trial_temp/{i}/out.log'
                self.addLogObserver(f'test.{i}.log', output_observer)
        else:
            self.addLogObserver('test.log', output_observer)
        if self.testChanges:
            for f in self.build.allFiles():
                if f.endswith('.py'):
                    command.append(f'--testmodule={f}')
        else:
            command.extend(self.tests)
        self.setup_python_path()
        cmd = (yield self.makeRemoteShellCommand(command=command))
        yield self.runCommand(cmd)
        stdio_log = (yield self.getLog('stdio'))
        yield stdio_log.finish()
        problems = '\n'.join(self.problems)
        warnings = self.warnings
        if problems:
            yield self.addCompleteLog('problems', problems)
        if warnings:
            lines = sorted(warnings.keys())
            yield self.addCompleteLog('warnings', ''.join(lines))
        return self.build_results(cmd)

    def build_results(self, cmd):
        if False:
            for i in range(10):
                print('nop')
        counts = self.observer.counts
        total = counts['total']
        failures = counts['failures']
        errors = counts['errors']
        parsed = total is not None
        desc_parts = []
        if not cmd.didFail():
            if parsed:
                results = SUCCESS
                if total:
                    desc_parts += [str(total), total == 1 and 'test' or 'tests', 'passed']
                else:
                    desc_parts += ['no tests', 'run']
            else:
                results = FAILURE
                desc_parts += ['testlog', 'unparseable']
        else:
            results = FAILURE
            if parsed:
                desc_parts += ['tests']
                if failures:
                    desc_parts += [str(failures), failures == 1 and 'failure' or 'failures']
                if errors:
                    desc_parts += [str(errors), errors == 1 and 'error' or 'errors']
            else:
                desc_parts += ['tests', 'failed']
        if counts['skips']:
            desc_parts += [str(counts['skips']), counts['skips'] == 1 and 'skip' or 'skips']
        if counts['expectedFailures']:
            desc_parts += [str(counts['expectedFailures']), 'todo' if counts['expectedFailures'] == 1 else 'todos']
        if self.reactor:
            desc_parts.append(self.rtext('({})'))
        self.descriptionDone = util.join_list(desc_parts)
        return results

    def rtext(self, fmt='{}'):
        if False:
            while True:
                i = 10
        if self.reactor:
            rtext = fmt.format(self.reactor)
            return rtext.replace('reactor', '')
        return ''

    def logConsumer(self):
        if False:
            i = 10
            return i + 15
        while True:
            (_, line) = (yield)
            if line.find(' exceptions.DeprecationWarning: ') != -1:
                warning = line
                self.warnings[warning] = self.warnings.get(warning, 0) + 1
            elif line.find(' DeprecationWarning: ') != -1 or line.find(' UserWarning: ') != -1:
                warning = line + '\n' + (yield)[1] + '\n'
                self.warnings[warning] = self.warnings.get(warning, 0) + 1
            elif line.find('Warning: ') != -1:
                warning = line
                self.warnings[warning] = self.warnings.get(warning, 0) + 1
            if line.find('=' * 60) == 0 or line.find('-' * 60) == 0:
                while True:
                    self.problems.append(line)
                    (_, line) = (yield)

class RemovePYCs(shell.ShellCommand):
    name = 'remove_pyc'
    command = ['find', '.', '-name', "'*.pyc'", '-exec', 'rm', '{}', ';']
    description = 'removing .pyc files'
    descriptionDone = 'remove .pycs'