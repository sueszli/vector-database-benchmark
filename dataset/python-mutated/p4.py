import re
from twisted.internet import defer
from twisted.python import log
from buildbot import config
from buildbot import interfaces
from buildbot.interfaces import WorkerSetupError
from buildbot.process import buildstep
from buildbot.process import remotecommand
from buildbot.process import results
from buildbot.process.properties import Interpolate
from buildbot.steps.source import Source

class P4(Source):
    """Perform Perforce checkout/update operations."""
    name = 'p4'
    renderables = ['mode', 'p4base', 'p4client', 'p4viewspec', 'p4branch', 'p4passwd', 'p4port', 'p4user']
    possible_modes = ('incremental', 'full')
    possible_client_types = (None, 'readonly', 'partitioned')

    def __init__(self, mode='incremental', method=None, p4base=None, p4branch=None, p4port=None, p4user=None, p4passwd=None, p4extra_views=(), p4line_end='local', p4viewspec=None, p4viewspec_suffix='...', p4client=Interpolate('buildbot_%(prop:workername)s_%(prop:buildername)s'), p4client_spec_options='allwrite rmdir', p4client_type=None, p4extra_args=None, p4bin='p4', use_tickets=False, stream=False, debug=False, **kwargs):
        if False:
            return 10
        self.method = method
        self.mode = mode
        self.p4branch = p4branch
        self.p4bin = p4bin
        self.p4base = p4base
        self.p4port = p4port
        self.p4user = p4user
        self.p4passwd = p4passwd
        self.p4extra_views = p4extra_views
        self.p4viewspec = p4viewspec
        self.p4viewspec_suffix = p4viewspec_suffix
        self.p4line_end = p4line_end
        self.p4client = p4client
        self.p4client_spec_options = p4client_spec_options
        self.p4client_type = p4client_type
        self.p4extra_args = p4extra_args
        self.use_tickets = use_tickets
        self.stream = stream
        self.debug = debug
        super().__init__(**kwargs)
        if self.mode not in self.possible_modes and (not interfaces.IRenderable.providedBy(self.mode)):
            config.error(f'mode {self.mode} is not an IRenderable, or one of {self.possible_modes}')
        if not p4viewspec and p4base is None:
            config.error('You must provide p4base or p4viewspec')
        if p4viewspec and (p4base or p4branch or p4extra_views):
            config.error('Either provide p4viewspec or p4base and p4branch (and optionally p4extra_views)')
        if p4viewspec and isinstance(p4viewspec, str):
            config.error('p4viewspec must not be a string, and should be a sequence of 2 element sequences')
        if not interfaces.IRenderable.providedBy(p4base) and p4base and (not p4base.startswith('/')):
            config.error(f'p4base should start with // [p4base = {p4base}]')
        if not interfaces.IRenderable.providedBy(p4base) and p4base and p4base.endswith('/'):
            config.error(f'p4base should not end with a trailing / [p4base = {p4base}]')
        if not interfaces.IRenderable.providedBy(p4branch) and p4branch and p4branch.endswith('/'):
            config.error(f'p4branch should not end with a trailing / [p4branch = {p4branch}]')
        if stream:
            if p4extra_views or p4viewspec:
                config.error("You can't use p4extra_views not p4viewspec with stream")
            if not p4base or not p4branch:
                config.error('You must specify both p4base and p4branch when using stream')
            if not interfaces.IRenderable.providedBy(p4base) and ' ' in p4base:
                config.error('p4base must not contain any whitespace')
            if not interfaces.IRenderable.providedBy(p4branch) and ' ' in p4branch:
                config.error('p4branch must not contain any whitespace')
        if self.p4client_spec_options is None:
            self.p4client_spec_options = ''
        if self.p4client_type not in self.possible_client_types and (not interfaces.IRenderable.providedBy(self.p4client_type)):
            config.error(f'p4client_type {self.p4client_type} is not an IRenderable, or one of {{self.possible_client_types}}')

    @defer.inlineCallbacks
    def run_vc(self, branch, revision, patch):
        if False:
            while True:
                i = 10
        if self.debug:
            log.msg('in run_vc')
        self.method = self._getMethod()
        self.stdio_log = (yield self.addLogForRemoteCommands('stdio'))
        installed = (yield self.checkP4())
        if not installed:
            raise WorkerSetupError('p4 is not installed on worker')
        if self.p4passwd is not None:
            if not self.workerVersionIsOlderThan('shell', '2.16'):
                self.p4passwd_arg = ('obfuscated', self.p4passwd, 'XXXXXX')
            else:
                self.p4passwd_arg = self.p4passwd
                log.msg('Worker does not understand obfuscation; p4 password will be logged')
        if self.use_tickets and self.p4passwd:
            yield self._acquireTicket()
        yield self._createClientSpec()
        self.revision = (yield self.get_sync_revision(revision))
        yield self._getAttrGroupMember('mode', self.mode)()
        self.updateSourceProperty('got_revision', self.revision)
        return results.SUCCESS

    @defer.inlineCallbacks
    def mode_full(self):
        if False:
            return 10
        if self.debug:
            log.msg('P4:full()..')
        yield self._dovccmd(['sync', '#none'])
        yield self.runRmdir(self.workdir)
        if self.revision:
            if self.debug:
                log.msg('P4: full() sync command based on :client:%s changeset:%d', self.p4client, int(self.revision))
            yield self._dovccmd(['sync', f'//{self.p4client}/...@{int(self.revision)}'], collectStdout=True)
        else:
            if self.debug:
                log.msg('P4: full() sync command based on :client:%s no revision', self.p4client)
            yield self._dovccmd(['sync'], collectStdout=True)
        if self.debug:
            log.msg('P4: full() sync done.')

    @defer.inlineCallbacks
    def mode_incremental(self):
        if False:
            return 10
        if self.debug:
            log.msg('P4:incremental()')
        command = ['sync']
        if self.revision:
            command.extend([f'//{self.p4client}/...@{int(self.revision)}'])
        if self.debug:
            log.msg('P4:incremental() command:%s revision:%s', command, self.revision)
        yield self._dovccmd(command)

    def _buildVCCommand(self, doCommand):
        if False:
            return 10
        assert doCommand, 'No command specified'
        command = [self.p4bin]
        if self.p4port:
            command.extend(['-p', self.p4port])
        if self.p4user:
            command.extend(['-u', self.p4user])
        if not self.use_tickets and self.p4passwd:
            command.extend(['-P', self.p4passwd_arg])
        if self.p4client:
            command.extend(['-c', self.p4client])
        if doCommand[0] == 'sync' and self.p4extra_args:
            command.extend(self.p4extra_args)
        command.extend(doCommand)
        return command

    @defer.inlineCallbacks
    def _dovccmd(self, command, collectStdout=False, initialStdin=None):
        if False:
            while True:
                i = 10
        command = self._buildVCCommand(command)
        if self.debug:
            log.msg(f'P4:_dovccmd():workdir->{self.workdir}')
        cmd = remotecommand.RemoteShellCommand(self.workdir, command, env=self.env, logEnviron=self.logEnviron, timeout=self.timeout, collectStdout=collectStdout, initialStdin=initialStdin)
        cmd.useLog(self.stdio_log, False)
        if self.debug:
            log.msg(f"Starting p4 command : p4 {' '.join(command)}")
        yield self.runCommand(cmd)
        if cmd.rc != 0:
            if self.debug:
                log.msg(f'P4:_dovccmd():Source step failed while running command {cmd}')
            raise buildstep.BuildStepFailed()
        if collectStdout:
            return cmd.stdout
        return cmd.rc

    def _getMethod(self):
        if False:
            print('Hello World!')
        if self.method is not None and self.mode != 'incremental':
            return self.method
        elif self.mode == 'incremental':
            return None
        elif self.method is None and self.mode == 'full':
            return 'fresh'
        return None

    @defer.inlineCallbacks
    def _createClientSpec(self):
        if False:
            return 10
        builddir = self.getProperty('builddir')
        if self.debug:
            log.msg(f'P4:_createClientSpec() builddir:{builddir}')
            log.msg(f'P4:_createClientSpec() SELF.workdir:{self.workdir}')
        prop_dict = self.getProperties().asDict()
        prop_dict['p4client'] = self.p4client
        root = self.build.path_module.normpath(self.build.path_module.join(builddir, self.workdir))
        client_spec = ''
        client_spec += f'Client: {self.p4client}\n\n'
        client_spec += f'Owner: {self.p4user}\n\n'
        client_spec += f'Description:\n\tCreated by {self.p4user}\n\n'
        client_spec += f'Root:\t{root}\n\n'
        client_spec += f'Options:\t{self.p4client_spec_options}\n\n'
        if self.p4line_end:
            client_spec += f'LineEnd:\t{self.p4line_end}\n\n'
        else:
            client_spec += 'LineEnd:\tlocal\n\n'
        if self.p4client_type is not None:
            client_spec += f'Type:\t{self.p4client_type}\n\n'
        if self.stream:
            client_spec += f'Stream:\t{self.p4base}/{self.p4branch}\n'
        else:
            client_spec += 'View:\n'

            def has_whitespace(*args):
                if False:
                    while True:
                        i = 10
                return any((re.search('\\s', i) for i in args if i is not None))
            if self.p4viewspec:
                suffix = self.p4viewspec_suffix or ''
                for (k, v) in self.p4viewspec:
                    if self.debug:
                        log.msg(f'P4:_createClientSpec():key:{k} value:{v}')
                    qa = '"' if has_whitespace(k, suffix) else ''
                    qb = '"' if has_whitespace(self.p4client, v, suffix) else ''
                    client_spec += f'\t{qa}{k}{suffix}{qa} {qb}//{self.p4client}/{v}{suffix}{qb}\n'
            else:
                qa = '"' if has_whitespace(self.p4base, self.p4branch) else ''
                client_spec += f'\t{qa}{self.p4base}'
                if self.p4branch:
                    client_spec += f'/{self.p4branch}'
                client_spec += f'/...{qa} '
                qb = '"' if has_whitespace(self.p4client) else ''
                client_spec += f'{qb}//{self.p4client}/...{qb}\n'
                if self.p4extra_views:
                    for (k, v) in self.p4extra_views:
                        qa = '"' if has_whitespace(k) else ''
                        qb = '"' if has_whitespace(k, self.p4client, v) else ''
                        client_spec += f'\t{qa}{k}/...{qa} {qb}//{self.p4client}/{v}/...{qb}\n'
        if self.debug:
            log.msg(client_spec)
        stdout = (yield self._dovccmd(['client', '-i'], collectStdout=True, initialStdin=client_spec))
        mo = re.search('Client (\\S+) (.+)$', stdout, re.M)
        return mo and (mo.group(2) == 'saved.' or mo.group(2) == 'not changed.')

    @defer.inlineCallbacks
    def _acquireTicket(self):
        if False:
            return 10
        if self.debug:
            log.msg('P4:acquireTicket()')
        initialStdin = self.p4passwd + '\n'
        yield self._dovccmd(['login'], initialStdin=initialStdin)

    @defer.inlineCallbacks
    def get_sync_revision(self, revision=None):
        if False:
            return 10
        revision = f'@{revision}' if revision else '#head'
        if self.debug:
            log.msg('P4: get_sync_revision() retrieve client actual revision at %s', revision)
        changes_command_args = ['-ztag', 'changes', '-m1', f'//{self.p4client}/...{revision}']
        command = self._buildVCCommand(changes_command_args)
        cmd = remotecommand.RemoteShellCommand(self.workdir, command, env=self.env, timeout=self.timeout, logEnviron=self.logEnviron, collectStdout=True)
        cmd.useLog(self.stdio_log, False)
        yield self.runCommand(cmd)
        stdout = cmd.stdout.splitlines(keepends=False)
        change_identifier = '... change '
        revision = next((line[len(change_identifier):] for line in stdout if line.startswith(change_identifier)), None)
        try:
            int(revision)
        except ValueError as error:
            log.msg('p4.get_sync_revision unable to parse output of %s: %s', ['p4'] + changes_command_args, stdout)
            raise buildstep.BuildStepFailed() from error
        return revision

    @defer.inlineCallbacks
    def purge(self, ignore_ignores):
        if False:
            return 10
        'Delete everything that shown up on status.'
        command = ['sync', '#none']
        if ignore_ignores:
            command.append('--no-ignore')
        yield self._dovccmd(command, collectStdout=True)

    @defer.inlineCallbacks
    def checkP4(self):
        if False:
            while True:
                i = 10
        cmd = remotecommand.RemoteShellCommand(self.workdir, [self.p4bin, '-V'], env=self.env, logEnviron=self.logEnviron)
        cmd.useLog(self.stdio_log, False)
        yield self.runCommand(cmd)
        return cmd.rc == 0

    def computeSourceRevision(self, changes):
        if False:
            i = 10
            return i + 15
        if not changes or None in [c.revision for c in changes]:
            return None
        lastChange = max((int(c.revision) for c in changes))
        return lastChange