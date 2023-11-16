import os
import re
import stat
from urllib.parse import quote as urlquote
from twisted.internet import defer
from twisted.python import log
from buildbot import config
from buildbot.changes import base
from buildbot.util import bytes2unicode
from buildbot.util import private_tempdir
from buildbot.util import runprocess
from buildbot.util.git import GitMixin
from buildbot.util.git import getSshKnownHostsContents
from buildbot.util.misc import writeLocalFile
from buildbot.util.state import StateMixin

class GitError(Exception):
    """Raised when git exits with code 128."""

class GitPoller(base.ReconfigurablePollingChangeSource, StateMixin, GitMixin):
    """This source will poll a remote git repo for changes and submit
    them to the change master."""
    compare_attrs = ('repourl', 'branches', 'workdir', 'pollInterval', 'gitbin', 'usetimestamps', 'category', 'project', 'pollAtLaunch', 'buildPushesWithNoCommits', 'sshPrivateKey', 'sshHostKey', 'sshKnownHosts', 'pollRandomDelayMin', 'pollRandomDelayMax')
    secrets = ('sshPrivateKey', 'sshHostKey', 'sshKnownHosts')

    def __init__(self, repourl, **kwargs):
        if False:
            return 10
        name = kwargs.get('name', None)
        if name is None:
            kwargs['name'] = repourl
        super().__init__(repourl, **kwargs)

    def checkConfig(self, repourl, branches=None, branch=None, workdir=None, pollInterval=10 * 60, gitbin='git', usetimestamps=True, category=None, project=None, pollinterval=-2, fetch_refspec=None, encoding='utf-8', name=None, pollAtLaunch=False, buildPushesWithNoCommits=False, only_tags=False, sshPrivateKey=None, sshHostKey=None, sshKnownHosts=None, pollRandomDelayMin=0, pollRandomDelayMax=0):
        if False:
            while True:
                i = 10
        if pollinterval != -2:
            pollInterval = pollinterval
        if only_tags and (branch or branches):
            config.error("GitPoller: can't specify only_tags and branch/branches")
        if branch and branches:
            config.error("GitPoller: can't specify both branch and branches")
        self.sshPrivateKey = sshPrivateKey
        self.sshHostKey = sshHostKey
        self.sshKnownHosts = sshKnownHosts
        self.setupGit(logname='GitPoller')
        if fetch_refspec is not None:
            config.error('GitPoller: fetch_refspec is no longer supported. Instead, only the given branches are downloaded.')
        if name is None:
            name = repourl
        super().checkConfig(name=name, pollInterval=pollInterval, pollAtLaunch=pollAtLaunch, pollRandomDelayMin=pollRandomDelayMin, pollRandomDelayMax=pollRandomDelayMax)

    @defer.inlineCallbacks
    def reconfigService(self, repourl, branches=None, branch=None, workdir=None, pollInterval=10 * 60, gitbin='git', usetimestamps=True, category=None, project=None, pollinterval=-2, fetch_refspec=None, encoding='utf-8', name=None, pollAtLaunch=False, buildPushesWithNoCommits=False, only_tags=False, sshPrivateKey=None, sshHostKey=None, sshKnownHosts=None, pollRandomDelayMin=0, pollRandomDelayMax=0):
        if False:
            for i in range(10):
                print('nop')
        if pollinterval != -2:
            pollInterval = pollinterval
        if name is None:
            name = repourl
        if project is None:
            project = ''
        if branch:
            branches = [branch]
        elif not branches:
            if only_tags:
                branches = lambda ref: ref.startswith('refs/tags/')
            else:
                branches = ['master']
        self.repourl = repourl
        self.branches = branches
        self.encoding = encoding
        self.buildPushesWithNoCommits = buildPushesWithNoCommits
        self.gitbin = gitbin
        self.workdir = workdir
        self.usetimestamps = usetimestamps
        self.category = category if callable(category) else bytes2unicode(category, encoding=self.encoding)
        self.project = bytes2unicode(project, encoding=self.encoding)
        self.changeCount = 0
        self.lastRev = {}
        self.sshPrivateKey = sshPrivateKey
        self.sshHostKey = sshHostKey
        self.sshKnownHosts = sshKnownHosts
        self.setupGit(logname='GitPoller')
        if self.workdir is None:
            self.workdir = 'gitpoller-work'
        if not os.path.isabs(self.workdir):
            self.workdir = os.path.join(self.master.basedir, self.workdir)
            log.msg(f"gitpoller: using workdir '{self.workdir}'")
        yield super().reconfigService(name=name, pollInterval=pollInterval, pollAtLaunch=pollAtLaunch, pollRandomDelayMin=pollRandomDelayMin, pollRandomDelayMax=pollRandomDelayMax)

    @defer.inlineCallbacks
    def _checkGitFeatures(self):
        if False:
            for i in range(10):
                print('nop')
        stdout = (yield self._dovccmd('--version', []))
        self.parseGitFeatures(stdout)
        if not self.gitInstalled:
            raise EnvironmentError('Git is not installed')
        if self.sshPrivateKey is not None and (not self.supportsSshPrivateKeyAsEnvOption):
            raise EnvironmentError('SSH private keys require Git 2.3.0 or newer')

    @defer.inlineCallbacks
    def activate(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.lastRev = (yield self.getState('lastRev', {}))
            super().activate()
        except Exception as e:
            log.err(e, 'while initializing GitPoller repository')

    def describe(self):
        if False:
            return 10
        str = 'GitPoller watching the remote git repository ' + bytes2unicode(self.repourl, self.encoding)
        if self.branches:
            if self.branches is True:
                str += ', branches: ALL'
            elif not callable(self.branches):
                str += ', branches: ' + ', '.join(self.branches)
        if not self.master:
            str += ' [STOPPED - check log]'
        return str

    def _getBranches(self):
        if False:
            for i in range(10):
                print('nop')
        d = self._dovccmd('ls-remote', ['--refs', self.repourl])

        @d.addCallback
        def parseRemote(rows):
            if False:
                while True:
                    i = 10
            branches = []
            for row in rows.splitlines():
                if '\t' not in row:
                    continue
                (_, ref) = row.split('\t')
                branches.append(ref)
            return branches
        return d

    def _headsFilter(self, branch):
        if False:
            i = 10
            return i + 15
        "Filter out remote references that don't begin with 'refs/heads'."
        return branch.startswith('refs/heads/')

    def _removeHeads(self, branch):
        if False:
            for i in range(10):
                print('nop')
        "Remove 'refs/heads/' prefix from remote references."
        if branch.startswith('refs/heads/'):
            branch = branch[11:]
        return branch

    def _trackerBranch(self, branch):
        if False:
            i = 10
            return i + 15
        url = urlquote(self.repourl, '').replace('~', '%7E')
        return f'refs/buildbot/{url}/{self._removeHeads(branch)}'

    def poll_should_exit(self):
        if False:
            print('Hello World!')
        return not self.doPoll.running

    @defer.inlineCallbacks
    def poll(self):
        if False:
            return 10
        yield self._checkGitFeatures()
        try:
            yield self._dovccmd('init', ['--bare', self.workdir])
        except GitError as e:
            log.msg(e.args[0])
            return
        branches = self.branches if self.branches else []
        remote_refs = (yield self._getBranches())
        if self.poll_should_exit():
            return
        if branches is True or callable(branches):
            if callable(self.branches):
                branches = [b for b in remote_refs if self.branches(b)]
            else:
                branches = [b for b in remote_refs if self._headsFilter(b)]
        elif branches and remote_refs:
            remote_branches = [self._removeHeads(b) for b in remote_refs]
            branches = sorted(list(set(branches) & set(remote_branches)))
        refspecs = [f'+{self._removeHeads(branch)}:{self._trackerBranch(branch)}' for branch in branches]
        try:
            yield self._dovccmd('fetch', ['--progress', self.repourl] + refspecs, path=self.workdir)
        except GitError as e:
            log.msg(e.args[0])
            return
        revs = {}
        log.msg(f'gitpoller: processing changes from "{self.repourl}"')
        for branch in branches:
            try:
                if self.poll_should_exit():
                    break
                rev = (yield self._dovccmd('rev-parse', [self._trackerBranch(branch)], path=self.workdir))
                revs[branch] = bytes2unicode(rev, self.encoding)
                yield self._process_changes(revs[branch], branch)
            except Exception:
                log.err(_why=f'trying to poll branch {branch} of {self.repourl}')
        self.lastRev = revs
        yield self.setState('lastRev', self.lastRev)

    def _get_commit_comments(self, rev):
        if False:
            i = 10
            return i + 15
        args = ['--no-walk', '--format=%s%n%b', rev, '--']
        d = self._dovccmd('log', args, path=self.workdir)
        return d

    def _get_commit_timestamp(self, rev):
        if False:
            for i in range(10):
                print('nop')
        args = ['--no-walk', '--format=%ct', rev, '--']
        d = self._dovccmd('log', args, path=self.workdir)

        @d.addCallback
        def process(git_output):
            if False:
                for i in range(10):
                    print('nop')
            if self.usetimestamps:
                try:
                    stamp = int(git_output)
                except Exception as e:
                    log.msg(f"gitpoller: caught exception converting output '{git_output}' to timestamp")
                    raise e
                return stamp
            return None
        return d

    def _get_commit_files(self, rev):
        if False:
            i = 10
            return i + 15
        args = ['--name-only', '--no-walk', '--format=%n', rev, '--']
        d = self._dovccmd('log', args, path=self.workdir)

        def decode_file(file):
            if False:
                while True:
                    i = 10
            match = re.match('^"(.*)"$', file)
            if match:
                file = bytes2unicode(match.groups()[0], encoding=self.encoding, errors='unicode_escape')
            return bytes2unicode(file, encoding=self.encoding)

        @d.addCallback
        def process(git_output):
            if False:
                for i in range(10):
                    print('nop')
            fileList = [decode_file(file) for file in [s for s in git_output.splitlines() if len(s)]]
            return fileList
        return d

    def _get_commit_author(self, rev):
        if False:
            for i in range(10):
                print('nop')
        args = ['--no-walk', '--format=%aN <%aE>', rev, '--']
        d = self._dovccmd('log', args, path=self.workdir)

        @d.addCallback
        def process(git_output):
            if False:
                i = 10
                return i + 15
            if not git_output:
                raise EnvironmentError('could not get commit author for rev')
            return git_output
        return d

    @defer.inlineCallbacks
    def _get_commit_committer(self, rev):
        if False:
            return 10
        args = ['--no-walk', '--format=%cN <%cE>', rev, '--']
        res = (yield self._dovccmd('log', args, path=self.workdir))
        if not res:
            raise EnvironmentError('could not get commit committer for rev')
        return res

    @defer.inlineCallbacks
    def _process_changes(self, newRev, branch):
        if False:
            print('Hello World!')
        '\n        Read changes since last change.\n\n        - Read list of commit hashes.\n        - Extract details from each commit.\n        - Add changes to database.\n        '
        if not self.lastRev:
            return
        revListArgs = ['--ignore-missing'] + ['--format=%H', f'{newRev}'] + ['^' + rev for rev in sorted(self.lastRev.values())] + ['--']
        self.changeCount = 0
        results = (yield self._dovccmd('log', revListArgs, path=self.workdir))
        revList = results.split()
        revList.reverse()
        if self.buildPushesWithNoCommits and (not revList):
            existingRev = self.lastRev.get(branch)
            if existingRev != newRev:
                revList = [newRev]
                if existingRev is None:
                    log.msg(f'gitpoller: rebuilding {newRev} for new branch "{branch}"')
                else:
                    log.msg(f'gitpoller: rebuilding {newRev} for updated branch "{branch}"')
        self.changeCount = len(revList)
        self.lastRev[branch] = newRev
        if self.changeCount:
            log.msg(f'gitpoller: processing {self.changeCount} changes: {revList} from "{self.repourl}" branch "{branch}"')
        for rev in revList:
            dl = defer.DeferredList([self._get_commit_timestamp(rev), self._get_commit_author(rev), self._get_commit_committer(rev), self._get_commit_files(rev), self._get_commit_comments(rev)], consumeErrors=True)
            results = (yield dl)
            failures = [r[1] for r in results if not r[0]]
            if failures:
                for failure in failures:
                    log.err(failure, f'while processing changes for {newRev} {branch}')
                failures[0].raiseException()
            (timestamp, author, committer, files, comments) = [r[1] for r in results]
            yield self.master.data.updates.addChange(author=author, committer=committer, revision=bytes2unicode(rev, encoding=self.encoding), files=files, comments=comments, when_timestamp=timestamp, branch=bytes2unicode(self._removeHeads(branch)), project=self.project, repository=bytes2unicode(self.repourl, encoding=self.encoding), category=self.category, src='git')

    def _isSshPrivateKeyNeededForCommand(self, command):
        if False:
            return 10
        commandsThatNeedKey = ['fetch', 'ls-remote']
        if self.sshPrivateKey is not None and command in commandsThatNeedKey:
            return True
        return False

    def _downloadSshPrivateKey(self, keyPath):
        if False:
            return 10
        writeLocalFile(keyPath, self.sshPrivateKey, mode=stat.S_IRUSR)

    def _downloadSshKnownHosts(self, path):
        if False:
            i = 10
            return i + 15
        if self.sshKnownHosts is not None:
            contents = self.sshKnownHosts
        else:
            contents = getSshKnownHostsContents(self.sshHostKey)
        writeLocalFile(path, contents)

    def _getSshPrivateKeyPath(self, ssh_data_path):
        if False:
            print('Hello World!')
        return os.path.join(ssh_data_path, 'ssh-key')

    def _getSshKnownHostsPath(self, ssh_data_path):
        if False:
            while True:
                i = 10
        return os.path.join(ssh_data_path, 'ssh-known-hosts')

    @defer.inlineCallbacks
    def _dovccmd(self, command, args, path=None):
        if False:
            print('Hello World!')
        if self._isSshPrivateKeyNeededForCommand(command):
            with private_tempdir.PrivateTemporaryDirectory(dir=self.workdir, prefix='.buildbot-ssh') as tmp_path:
                stdout = (yield self._dovccmdImpl(command, args, path, tmp_path))
        else:
            stdout = (yield self._dovccmdImpl(command, args, path, None))
        return stdout

    @defer.inlineCallbacks
    def _dovccmdImpl(self, command, args, path, ssh_workdir):
        if False:
            for i in range(10):
                print('nop')
        full_args = []
        full_env = os.environ.copy()
        if self._isSshPrivateKeyNeededForCommand(command):
            key_path = self._getSshPrivateKeyPath(ssh_workdir)
            self._downloadSshPrivateKey(key_path)
            known_hosts_path = None
            if self.sshHostKey is not None or self.sshKnownHosts is not None:
                known_hosts_path = self._getSshKnownHostsPath(ssh_workdir)
                self._downloadSshKnownHosts(known_hosts_path)
            self.adjustCommandParamsForSshPrivateKey(full_args, full_env, key_path, None, known_hosts_path)
        full_args += [command] + args
        res = (yield runprocess.run_process(self.master.reactor, [self.gitbin] + full_args, path, env=full_env))
        (code, stdout, stderr) = res
        stdout = bytes2unicode(stdout, self.encoding)
        stderr = bytes2unicode(stderr, self.encoding)
        if code != 0:
            if code == 128:
                raise GitError(f'command {full_args} in {path} on repourl {self.repourl} failed with exit code {code}: {stderr}')
            raise EnvironmentError(f'command {full_args} in {path} on repourl {self.repourl} failed with exit code {code}: {stderr}')
        return stdout.strip()