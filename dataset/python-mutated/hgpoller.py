import os
import time
from twisted.internet import defer
from twisted.python import log
from buildbot import config
from buildbot.changes import base
from buildbot.util import bytes2unicode
from buildbot.util import deferredLocked
from buildbot.util import runprocess
from buildbot.util.state import StateMixin

class HgPoller(base.ReconfigurablePollingChangeSource, StateMixin):
    """This source will poll a remote hg repo for changes and submit
    them to the change master."""
    compare_attrs = ('repourl', 'branch', 'branches', 'bookmarks', 'workdir', 'pollInterval', 'hgpoller', 'usetimestamps', 'category', 'project', 'pollAtLaunch', 'pollRandomDelayMin', 'pollRandomDelayMax')
    db_class_name = 'HgPoller'

    def __init__(self, repourl, **kwargs):
        if False:
            print('Hello World!')
        name = kwargs.get('name', None)
        if not name:
            branches = self.build_branches(kwargs.get('branch', None), kwargs.get('branches', None))
            kwargs['name'] = self.build_name(None, repourl, kwargs.get('bookmarks', None), branches)
        self.initLock = defer.DeferredLock()
        super().__init__(repourl, **kwargs)

    def checkConfig(self, repourl, branch=None, branches=None, bookmarks=None, workdir=None, pollInterval=10 * 60, hgbin='hg', usetimestamps=True, category=None, project='', pollinterval=-2, encoding='utf-8', name=None, pollAtLaunch=False, revlink=lambda branch, revision: '', pollRandomDelayMin=0, pollRandomDelayMax=0):
        if False:
            for i in range(10):
                print('nop')
        if pollinterval != -2:
            pollInterval = pollinterval
        if branch and branches:
            config.error("HgPoller: can't specify both branch and branches")
        if not callable(revlink):
            config.error('You need to provide a valid callable for revlink')
        if workdir is None:
            config.error('workdir is mandatory for now in HgPoller')
        name = self.build_name(name, repourl, bookmarks, self.build_branches(branch, branches))
        super().checkConfig(name=name, pollInterval=pollInterval, pollAtLaunch=pollAtLaunch, pollRandomDelayMin=pollRandomDelayMin, pollRandomDelayMax=pollRandomDelayMax)

    @defer.inlineCallbacks
    def reconfigService(self, repourl, branch=None, branches=None, bookmarks=None, workdir=None, pollInterval=10 * 60, hgbin='hg', usetimestamps=True, category=None, project='', pollinterval=-2, encoding='utf-8', name=None, pollAtLaunch=False, revlink=lambda branch, revision: '', pollRandomDelayMin=0, pollRandomDelayMax=0):
        if False:
            i = 10
            return i + 15
        if pollinterval != -2:
            pollInterval = pollinterval
        self.repourl = repourl
        self.branches = self.build_branches(branch, branches)
        self.bookmarks = bookmarks or []
        name = self.build_name(name, repourl, bookmarks, self.branches)
        if not self.branches and (not self.bookmarks):
            self.branches = ['default']
        self.encoding = encoding
        self.lastChange = time.time()
        self.lastPoll = time.time()
        self.hgbin = hgbin
        self.workdir = workdir
        self.usetimestamps = usetimestamps
        self.category = category if callable(category) else bytes2unicode(category)
        self.project = project
        self.lastRev = {}
        self.revlink_callable = revlink
        yield super().reconfigService(name=name, pollInterval=pollInterval, pollAtLaunch=pollAtLaunch, pollRandomDelayMin=pollRandomDelayMin, pollRandomDelayMax=pollRandomDelayMax)

    def build_name(self, name, repourl, bookmarks, branches):
        if False:
            for i in range(10):
                print('nop')
        if name is not None:
            return name
        name = repourl
        if bookmarks:
            name += '_' + '_'.join(bookmarks)
        if branches:
            name += '_' + '_'.join(branches)
        return name

    def build_branches(self, branch, branches):
        if False:
            for i in range(10):
                print('nop')
        if branch:
            return [branch]
        return branches or []

    @defer.inlineCallbacks
    def activate(self):
        if False:
            print('Hello World!')
        self.lastRev = (yield self.getState('lastRev', {}))
        super().activate()

    def describe(self):
        if False:
            return 10
        status = ''
        if not self.master:
            status = '[STOPPED - check log]'
        return f"HgPoller watching the remote Mercurial repository '{self.repourl}', branches: {', '.join(self.branches)}, in workdir '{self.workdir}' {status}"

    @deferredLocked('initLock')
    @defer.inlineCallbacks
    def poll(self):
        if False:
            for i in range(10):
                print('nop')
        yield self._getChanges()
        yield self._processChanges()

    def _absWorkdir(self):
        if False:
            i = 10
            return i + 15
        workdir = self.workdir
        if os.path.isabs(workdir):
            return workdir
        return os.path.join(self.master.basedir, workdir)

    @defer.inlineCallbacks
    def _getRevDetails(self, rev):
        if False:
            return 10
        'Return a deferred for (date, author, files, comments) of given rev.\n\n        Deferred will be in error if rev is unknown.\n        '
        command = [self.hgbin, 'log', '-r', rev, os.linesep.join(('--template={date|hgdate}', '{author}', "{files % '{file}" + os.pathsep + "'}", '{desc|strip}'))]
        (rc, output) = (yield runprocess.run_process(self.master.reactor, command, workdir=self._absWorkdir(), env=os.environ, collect_stderr=False, stderr_is_error=True))
        if rc != 0:
            msg = f'{self}: got error {rc} when getting details for revision {rev}'
            raise RuntimeError(msg)
        output = output.decode(self.encoding, 'replace')
        (date, author, files, comments) = output.split(os.linesep, 3)
        if not self.usetimestamps:
            stamp = None
        else:
            try:
                stamp = float(date.split()[0])
            except Exception:
                log.msg(f'hgpoller: caught exception converting output {repr(date)} to timestamp')
                raise
        return (stamp, author.strip(), files.split(os.pathsep)[:-1], comments.strip())

    def _isRepositoryReady(self):
        if False:
            return 10
        'Easy to patch in tests.'
        return os.path.exists(os.path.join(self._absWorkdir(), '.hg'))

    @defer.inlineCallbacks
    def _initRepository(self):
        if False:
            i = 10
            return i + 15
        'Have mercurial init the workdir as a repository (hg init) if needed.\n\n        hg init will also create all needed intermediate directories.\n        '
        if self._isRepositoryReady():
            return
        log.msg(f'hgpoller: initializing working dir from {self.repourl}')
        rc = (yield runprocess.run_process(self.master.reactor, [self.hgbin, 'init', self._absWorkdir()], env=os.environ, collect_stdout=False, collect_stderr=False))
        if rc != 0:
            self._stopOnFailure()
            raise EnvironmentError(f'{self}: repository init failed with exit code {rc}')
        log.msg(f'hgpoller: finished initializing working dir {self.workdir}')

    @defer.inlineCallbacks
    def _getChanges(self):
        if False:
            i = 10
            return i + 15
        self.lastPoll = time.time()
        yield self._initRepository()
        log.msg(f'{self}: polling hg repo at {self.repourl}')
        command = [self.hgbin, 'pull']
        for name in self.branches:
            command += ['-b', name]
        for name in self.bookmarks:
            command += ['-B', name]
        command += [self.repourl]
        yield runprocess.run_process(self.master.reactor, command, workdir=self._absWorkdir(), env=os.environ, collect_stdout=False, collect_stderr=False)

    def _getCurrentRev(self, branch='default'):
        if False:
            while True:
                i = 10
        'Return a deferred for current numeric rev in state db.\n\n        If never has been set, current rev is None.\n        '
        return self.lastRev.get(branch, None)

    def _setCurrentRev(self, rev, branch='default'):
        if False:
            return 10
        'Return a deferred to set current revision in persistent state.'
        self.lastRev[branch] = str(rev)
        return self.setState('lastRev', self.lastRev)

    @defer.inlineCallbacks
    def _getHead(self, branch):
        if False:
            while True:
                i = 10
        "Return a deferred for branch head revision or None.\n\n        We'll get an error if there is no head for this branch, which is\n        probably a good thing, since it's probably a misspelling\n        (if really buildbotting a branch that does not have any changeset\n        yet, one shouldn't be surprised to get errors)\n        "
        (rc, stdout) = (yield runprocess.run_process(self.master.reactor, [self.hgbin, 'heads', branch, '--template={rev}' + os.linesep], workdir=self._absWorkdir(), env=os.environ, collect_stderr=False, stderr_is_error=True))
        if rc != 0:
            log.err(f'{self}: could not find revision {branch} in repository {self.repourl}')
            return None
        if not stdout:
            return None
        if len(stdout.split()) > 1:
            log.err(f'{self}: caught several heads in branch {branch} from repository {self.repourl}. Staying at previous revisionYou should wait until the situation is normal again due to a merge or directly strip if remote repo gets stripped later.')
            return None
        return stdout.strip().decode(self.encoding)

    @defer.inlineCallbacks
    def _processChanges(self):
        if False:
            return 10
        "Send info about pulled changes to the master and record current.\n\n        HgPoller does the recording by moving the working dir to the head\n        of the branch.\n        We don't update the tree (unnecessary treatment and waste of space)\n        instead, we simply store the current rev number in a file.\n        Recall that hg rev numbers are local and incremental.\n        "
        for branch in self.branches + self.bookmarks:
            rev = (yield self._getHead(branch))
            if rev is None:
                continue
            yield self._processBranchChanges(rev, branch)

    @defer.inlineCallbacks
    def _getRevNodeList(self, revset):
        if False:
            while True:
                i = 10
        (rc, stdout) = (yield runprocess.run_process(self.master.reactor, [self.hgbin, 'log', '-r', revset, '--template={rev}:{node}\\n'], workdir=self._absWorkdir(), env=os.environ, collect_stdout=True, collect_stderr=False, stderr_is_error=True))
        if rc != 0:
            raise EnvironmentError(f'{self}: could not get rev node list: {rc}')
        results = stdout.decode(self.encoding)
        revNodeList = [rn.split(':', 1) for rn in results.strip().split()]
        return revNodeList

    @defer.inlineCallbacks
    def _processBranchChanges(self, new_rev, branch):
        if False:
            print('Hello World!')
        prev_rev = (yield self._getCurrentRev(branch))
        if new_rev == prev_rev:
            return
        if prev_rev is None:
            yield self._setCurrentRev(new_rev, branch)
            return
        revNodeList = (yield self._getRevNodeList(f'{prev_rev}::{new_rev}'))
        if not revNodeList:
            revNodeList = (yield self._getRevNodeList(new_rev))
        else:
            del revNodeList[0]
        log.msg(f'hgpoller: processing {len(revNodeList)} changes in branch {repr(branch)}: {repr(revNodeList)} in {repr(self._absWorkdir())}')
        for (_, node) in revNodeList:
            (timestamp, author, files, comments) = (yield self._getRevDetails(node))
            yield self.master.data.updates.addChange(author=author, committer=None, revision=str(node), revlink=self.revlink_callable(branch, str(node)), files=files, comments=comments, when_timestamp=int(timestamp) if timestamp else None, branch=bytes2unicode(branch), category=bytes2unicode(self.category), project=bytes2unicode(self.project), repository=bytes2unicode(self.repourl), src='hg')
            yield self._setCurrentRev(new_rev, branch)

    def _stopOnFailure(self, f):
        if False:
            while True:
                i = 10
        'utility method to stop the service when a failure occurs'
        if self.running:
            d = defer.maybeDeferred(self.stopService)
            d.addErrback(log.err, 'while stopping broken HgPoller service')
        return f