import base64
import json
import os
import random
import re
import shlex
import string
import sys
import time
from twisted.cred import credentials
from twisted.internet import defer
from twisted.internet import protocol
from twisted.internet import reactor
from twisted.internet import task
from twisted.internet import utils
from twisted.python import log
from twisted.python import runtime
from twisted.python.procutils import which
from twisted.spread import pb
from buildbot.process.results import SUCCESS
from buildbot.process.results import Results
from buildbot.util import bytes2unicode
from buildbot.util import now
from buildbot.util import unicode2bytes
from buildbot.util.eventual import fireEventually

class SourceStamp:

    def __init__(self, branch, revision, patch, repository=''):
        if False:
            return 10
        self.branch = branch
        self.revision = revision
        self.patch = patch
        self.repository = repository

def output(*msg):
    if False:
        for i in range(10):
            print('nop')
    print(' '.join([str(m) for m in msg]))

class SourceStampExtractor:

    def __init__(self, treetop, branch, repository):
        if False:
            print('Hello World!')
        self.treetop = treetop
        self.repository = repository
        self.branch = branch
        exes = which(self.vcexe)
        if not exes:
            output(f"Could not find executable '{self.vcexe}'.")
            sys.exit(1)
        self.exe = exes[0]

    def dovc(self, cmd):
        if False:
            i = 10
            return i + 15
        'This accepts the arguments of a command, without the actual\n        command itself.'
        env = os.environ.copy()
        env['LC_ALL'] = 'C'
        d = utils.getProcessOutputAndValue(self.exe, cmd, env=env, path=self.treetop)
        d.addCallback(self._didvc, cmd)
        return d

    def _didvc(self, res, cmd):
        if False:
            return 10
        (stdout, _, __) = res
        return stdout

    def get(self):
        if False:
            return 10
        'Return a Deferred that fires with a SourceStamp instance.'
        d = self.getBaseRevision()
        d.addCallback(self.getPatch)
        d.addCallback(self.done)
        return d

    def readPatch(self, diff, patchlevel):
        if False:
            print('Hello World!')
        if not diff:
            diff = None
        self.patch = (patchlevel, diff)

    def done(self, res):
        if False:
            while True:
                i = 10
        if not self.repository:
            self.repository = self.treetop
        ss = SourceStamp(bytes2unicode(self.branch), self.baserev, self.patch, repository=self.repository)
        return ss

class CVSExtractor(SourceStampExtractor):
    patchlevel = 0
    vcexe = 'cvs'

    def getBaseRevision(self):
        if False:
            i = 10
            return i + 15
        self.baserev = time.strftime('%Y-%m-%d %H:%M:%S +0000', time.gmtime(now()))
        return defer.succeed(None)

    def getPatch(self, res):
        if False:
            for i in range(10):
                print('nop')
        if self.branch is not None:
            output("Sorry, CVS 'try' builds don't work with branches")
            sys.exit(1)
        args = ['-q', 'diff', '-u', '-D', self.baserev]
        d = self.dovc(args)
        d.addCallback(self.readPatch, self.patchlevel)
        return d

class SVNExtractor(SourceStampExtractor):
    patchlevel = 0
    vcexe = 'svn'

    def getBaseRevision(self):
        if False:
            print('Hello World!')
        d = self.dovc(['status', '-u'])
        d.addCallback(self.parseStatus)
        return d

    def parseStatus(self, res):
        if False:
            while True:
                i = 10
        for line in res.split(b'\n'):
            m = re.search(b'^Status against revision:\\s+(\\d+)', line)
            if m:
                self.baserev = m.group(1)
                return
        output(b"Could not find 'Status against revision' in SVN output: " + res)
        sys.exit(1)

    def getPatch(self, res):
        if False:
            i = 10
            return i + 15
        d = self.dovc(['diff', f'-r{self.baserev}'])
        d.addCallback(self.readPatch, self.patchlevel)
        return d

class BzrExtractor(SourceStampExtractor):
    patchlevel = 0
    vcexe = 'bzr'

    def getBaseRevision(self):
        if False:
            while True:
                i = 10
        d = self.dovc(['revision-info', '-rsubmit:'])
        d.addCallback(self.get_revision_number)
        return d

    def get_revision_number(self, out):
        if False:
            for i in range(10):
                print('nop')
        (_, revid) = out.split()
        self.baserev = 'revid:' + revid
        return

    def getPatch(self, res):
        if False:
            for i in range(10):
                print('nop')
        d = self.dovc(['diff', f'-r{self.baserev}..'])
        d.addCallback(self.readPatch, self.patchlevel)
        return d

class MercurialExtractor(SourceStampExtractor):
    patchlevel = 1
    vcexe = 'hg'

    def _didvc(self, res, cmd):
        if False:
            return 10
        (stdout, stderr, code) = res
        if code:
            cs = ' '.join(['hg'] + cmd)
            if stderr:
                stderr = '\n' + stderr.rstrip()
            raise RuntimeError(f'{cs} returned {code} {stderr}')
        return stdout

    @defer.inlineCallbacks
    def getBaseRevision(self):
        if False:
            print('Hello World!')
        upstream = ''
        if self.repository:
            upstream = f"r'{self.repository}'"
        output = ''
        try:
            output = (yield self.dovc(['log', '--template', '{node}\\n', '-r', f'max(::. - outgoing({upstream}))']))
        except RuntimeError:
            if upstream:
                raise
            output = (yield self.dovc(['log', '--template', '{node}\\n', '-r', 'p1()']))
        m = re.search(b'^(\\w+)', output)
        if not m:
            raise RuntimeError(f'Revision {output!r} is not in the right format')
        self.baserev = m.group(0)

    def getPatch(self, res):
        if False:
            for i in range(10):
                print('nop')
        d = self.dovc(['diff', '-r', self.baserev])
        d.addCallback(self.readPatch, self.patchlevel)
        return d

class PerforceExtractor(SourceStampExtractor):
    patchlevel = 0
    vcexe = 'p4'

    def getBaseRevision(self):
        if False:
            while True:
                i = 10
        d = self.dovc(['changes', '-m1', '...'])
        d.addCallback(self.parseStatus)
        return d

    def parseStatus(self, res):
        if False:
            return 10
        m = re.search(b'Change (\\d+)', res)
        if m:
            self.baserev = m.group(1)
            return
        output(b'Could not find change number in output: ' + res)
        sys.exit(1)

    def readPatch(self, diff, patchlevel):
        if False:
            print('Hello World!')
        if not self.branch:
            output('you must specify a branch')
            sys.exit(1)
        mpatch = ''
        found = False
        for line in diff.split('\n'):
            m = re.search('==== //depot/' + self.branch + '/([\\w/\\.\\d\\-_]+)#(\\d+) -', line)
            if m:
                mpatch += f'--- {m.group(1)}#{m.group(2)}\n'
                mpatch += f'+++ {m.group(1)}\n'
                found = True
            else:
                mpatch += line
                mpatch += '\n'
        if not found:
            output(b'could not parse patch file')
            sys.exit(1)
        self.patch = (patchlevel, unicode2bytes(mpatch))

    def getPatch(self, res):
        if False:
            print('Hello World!')
        d = self.dovc(['diff'])
        d.addCallback(self.readPatch, self.patchlevel)
        return d

class DarcsExtractor(SourceStampExtractor):
    patchlevel = 1
    vcexe = 'darcs'

    def getBaseRevision(self):
        if False:
            for i in range(10):
                print('nop')
        d = self.dovc(['changes', '--context'])
        d.addCallback(self.parseStatus)
        return d

    def parseStatus(self, res):
        if False:
            return 10
        self.baserev = res

    def getPatch(self, res):
        if False:
            return 10
        d = self.dovc(['diff', '-u'])
        d.addCallback(self.readPatch, self.patchlevel)
        return d

class GitExtractor(SourceStampExtractor):
    patchlevel = 1
    vcexe = 'git'
    config = None

    def getBaseRevision(self):
        if False:
            return 10
        if self.branch:
            d = self.dovc(['rev-parse', self.branch])
            d.addCallback(self.override_baserev)
            d.addCallback(self.extractLocalBranch)
            return d
        d = self.dovc(['branch', '--no-color', '-v', '--no-abbrev'])
        d.addCallback(self.parseStatus)
        return d

    def extractLocalBranch(self, res):
        if False:
            print('Hello World!')
        if '/' in self.branch:
            d = self.dovc(['remote'])
            d.addCallback(self.fixBranch)
            return d
        return None

    def fixBranch(self, remotes):
        if False:
            for i in range(10):
                print('nop')
        for l in bytes2unicode(remotes).split('\n'):
            r = l.strip()
            if r and self.branch.startswith(r + '/'):
                self.branch = self.branch[len(r) + 1:]
                break

    def readConfig(self):
        if False:
            i = 10
            return i + 15
        if self.config:
            return defer.succeed(self.config)
        d = self.dovc(['config', '-l'])
        d.addCallback(self.parseConfig)
        return d

    def parseConfig(self, res):
        if False:
            print('Hello World!')
        self.config = {}
        for l in res.split(b'\n'):
            if l.strip():
                parts = l.strip().split(b'=', 2)
                if len(parts) < 2:
                    parts.append('true')
                self.config[parts[0]] = parts[1]
        return self.config

    def parseTrackingBranch(self, res):
        if False:
            i = 10
            return i + 15
        remote = self.config.get(b'branch.' + self.branch + b'.remote')
        ref = self.config.get(b'branch.' + self.branch + b'.merge')
        if remote and ref:
            remote_branch = ref.split(b'/', 2)[-1]
            baserev = remote + b'/' + remote_branch
        else:
            baserev = b'master'
        d = self.dovc(['rev-parse', baserev])
        d.addCallback(self.override_baserev)
        return d

    def override_baserev(self, res):
        if False:
            i = 10
            return i + 15
        self.baserev = bytes2unicode(res).strip()

    def parseStatus(self, res):
        if False:
            i = 10
            return i + 15
        m = re.search(b'^\\* (\\S+)\\s+([0-9a-f]{40})', res, re.MULTILINE)
        if m:
            self.baserev = m.group(2)
            self.branch = m.group(1)
            d = self.readConfig()
            d.addCallback(self.parseTrackingBranch)
            return d
        output(b'Could not find current GIT branch: ' + res)
        sys.exit(1)

    def getPatch(self, res):
        if False:
            for i in range(10):
                print('nop')
        d = self.dovc(['diff', '--src-prefix=a/', '--dst-prefix=b/', '--no-textconv', '--no-ext-diff', self.baserev])
        d.addCallback(self.readPatch, self.patchlevel)
        return d

class MonotoneExtractor(SourceStampExtractor):
    patchlevel = 0
    vcexe = 'mtn'

    def getBaseRevision(self):
        if False:
            for i in range(10):
                print('nop')
        d = self.dovc(['automate', 'get_base_revision_id'])
        d.addCallback(self.parseStatus)
        return d

    def parseStatus(self, output):
        if False:
            while True:
                i = 10
        hash = output.strip()
        if len(hash) != 40:
            self.baserev = None
        self.baserev = hash

    def getPatch(self, res):
        if False:
            print('Hello World!')
        d = self.dovc(['diff'])
        d.addCallback(self.readPatch, self.patchlevel)
        return d

def getSourceStamp(vctype, treetop, branch=None, repository=None):
    if False:
        return 10
    if vctype == 'cvs':
        cls = CVSExtractor
    elif vctype == 'svn':
        cls = SVNExtractor
    elif vctype == 'bzr':
        cls = BzrExtractor
    elif vctype == 'hg':
        cls = MercurialExtractor
    elif vctype == 'p4':
        cls = PerforceExtractor
    elif vctype == 'darcs':
        cls = DarcsExtractor
    elif vctype == 'git':
        cls = GitExtractor
    elif vctype == 'mtn':
        cls = MonotoneExtractor
    elif vctype == 'none':
        return defer.succeed(SourceStamp('', '', (1, ''), ''))
    else:
        output(f"unknown vctype '{vctype}'")
        sys.exit(1)
    return cls(treetop, branch, repository).get()

def ns(s):
    if False:
        for i in range(10):
            print('nop')
    return f'{len(s)}:{s},'

def createJobfile(jobid, branch, baserev, patch_level, patch_body, repository, project, who, comment, builderNames, properties):
    if False:
        return 10
    try:
        bytes2unicode(patch_body)
        version = 5
    except UnicodeDecodeError:
        version = 6
    job = ''
    job += ns(str(version))
    job_dict = {'jobid': jobid, 'branch': branch, 'baserev': str(baserev), 'patch_level': patch_level, 'repository': repository, 'project': project, 'who': who, 'comment': comment, 'builderNames': builderNames, 'properties': properties}
    if version > 5:
        job_dict['patch_body_base64'] = bytes2unicode(base64.b64encode(patch_body))
    else:
        job_dict['patch_body'] = bytes2unicode(patch_body)
    job += ns(json.dumps(job_dict))
    return job

def getTopdir(topfile, start=None):
    if False:
        for i in range(10):
            print('nop')
    'walk upwards from the current directory until we find this topfile'
    if not start:
        start = os.getcwd()
    here = start
    toomany = 20
    while toomany > 0:
        if os.path.exists(os.path.join(here, topfile)):
            return here
        next = os.path.dirname(here)
        if next == here:
            break
        here = next
        toomany -= 1
    output(f"Unable to find topfile '{topfile}' anywhere from {start} upwards")
    sys.exit(1)

class RemoteTryPP(protocol.ProcessProtocol):

    def __init__(self, job):
        if False:
            return 10
        self.job = job
        self.d = defer.Deferred()

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        self.transport.write(unicode2bytes(self.job))
        self.transport.closeStdin()

    def outReceived(self, data):
        if False:
            print('Hello World!')
        sys.stdout.write(bytes2unicode(data))

    def errReceived(self, data):
        if False:
            i = 10
            return i + 15
        sys.stderr.write(bytes2unicode(data))

    def processEnded(self, reason):
        if False:
            for i in range(10):
                print('nop')
        sig = reason.value.signal
        rc = reason.value.exitCode
        if sig is not None or rc != 0:
            self.d.errback(RuntimeError(f"remote 'buildbot tryserver' failed: sig={sig}, rc={rc}"))
            return
        self.d.callback((sig, rc))

class FakeBuildSetStatus:

    def callRemote(self, name):
        if False:
            print('Hello World!')
        if name == 'getBuildRequests':
            return defer.succeed([])
        raise NotImplementedError()

class Try(pb.Referenceable):
    buildsetStatus = None
    quiet = False
    printloop = False

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        self.config = config
        self.connect = self.getopt('connect')
        if self.connect not in ['ssh', 'pb']:
            output('you must specify a connect style: ssh or pb')
            sys.exit(1)
        self.builderNames = self.getopt('builders')
        self.project = self.getopt('project', '')
        self.who = self.getopt('who')
        self.comment = self.getopt('comment')

    def getopt(self, config_name, default=None):
        if False:
            i = 10
            return i + 15
        value = self.config.get(config_name)
        if value is None or value == []:
            value = default
        return value

    def createJob(self):
        if False:
            while True:
                i = 10
        self.bsid = f'{time.time()}-{random.randint(0, 1000000)}'
        branch = self.getopt('branch')
        difffile = self.config.get('diff')
        if difffile:
            baserev = self.config.get('baserev')
            if difffile == '-':
                diff = sys.stdin.read()
            else:
                with open(difffile, 'rb') as f:
                    diff = f.read()
            if not diff:
                diff = None
            patch = (self.config['patchlevel'], diff)
            ss = SourceStamp(branch, baserev, patch, repository=self.getopt('repository'))
            d = defer.succeed(ss)
        else:
            vc = self.getopt('vc')
            if vc in ('cvs', 'svn'):
                topdir = self.getopt('topdir')
                if topdir:
                    treedir = os.path.expanduser(topdir)
                else:
                    topfile = self.getopt('topfile')
                    if topfile:
                        treedir = getTopdir(topfile)
                    else:
                        output('Must specify topdir or topfile.')
                        sys.exit(1)
            else:
                treedir = os.getcwd()
            d = getSourceStamp(vc, treedir, branch, self.getopt('repository'))
        d.addCallback(self._createJob_1)
        return d

    def _createJob_1(self, ss):
        if False:
            for i in range(10):
                print('nop')
        self.sourcestamp = ss
        (patchlevel, diff) = ss.patch
        if diff is None:
            output('WARNING: There is no patch to try, diff is empty.')
        if self.connect == 'ssh':
            revspec = ss.revision
            if revspec is None:
                revspec = ''
            self.jobfile = createJobfile(self.bsid, ss.branch or '', revspec, patchlevel, diff, ss.repository, self.project, self.who, self.comment, self.builderNames, self.config.get('properties', {}))

    def fakeDeliverJob(self):
        if False:
            for i in range(10):
                print('nop')
        ss = self.sourcestamp
        output(f'Job:\n\tRepository: {ss.repository}\n\tProject: {self.project}\n\tBranch: {ss.branch}\n\tRevision: {ss.revision}\n\tBuilders: {self.builderNames}\n{ss.patch[1]}')
        self.buildsetStatus = FakeBuildSetStatus()
        d = defer.Deferred()
        d.callback(True)
        return d

    def deliver_job_ssh(self):
        if False:
            return 10
        tryhost = self.getopt('host')
        tryport = self.getopt('port')
        tryuser = self.getopt('username')
        trydir = self.getopt('jobdir')
        buildbotbin = self.getopt('buildbotbin')
        ssh_command = self.getopt('ssh')
        if not ssh_command:
            ssh_commands = which('ssh')
            if not ssh_commands:
                raise RuntimeError("couldn't find ssh executable, make sure it is available in the PATH")
            argv = [ssh_commands[0]]
        elif runtime.platformType == 'win32':
            argv = [string.strip(a, '"') for a in re.split('([^" ]+|"[^"]+")', ssh_command)[1::2]]
        else:
            argv = shlex.split(ssh_command)
        if tryuser:
            argv += ['-l', tryuser]
        if tryport:
            argv += ['-p', tryport]
        argv += [tryhost, buildbotbin, 'tryserver', '--jobdir', trydir]
        pp = RemoteTryPP(self.jobfile)
        reactor.spawnProcess(pp, argv[0], argv, os.environ)
        d = pp.d
        return d

    @defer.inlineCallbacks
    def deliver_job_pb(self):
        if False:
            for i in range(10):
                print('nop')
        user = self.getopt('username')
        passwd = self.getopt('passwd')
        master = self.getopt('master')
        (tryhost, tryport) = master.split(':')
        tryport = int(tryport)
        f = pb.PBClientFactory()
        d = f.login(credentials.UsernamePassword(unicode2bytes(user), unicode2bytes(passwd)))
        reactor.connectTCP(tryhost, tryport, f)
        remote = (yield d)
        ss = self.sourcestamp
        output('Delivering job; comment=', self.comment)
        self.buildsetStatus = (yield remote.callRemote('try', ss.branch, ss.revision, ss.patch, ss.repository, self.project, self.builderNames, self.who, self.comment, self.config.get('properties', {})))

    def deliverJob(self):
        if False:
            print('Hello World!')
        if self.connect == 'ssh':
            return self.deliver_job_ssh()
        if self.connect == 'pb':
            return self.deliver_job_pb()
        raise RuntimeError(f"unknown connecttype '{self.connect}', should be 'ssh' or 'pb'")

    def getStatus(self):
        if False:
            i = 10
            return i + 15
        wait = bool(self.getopt('wait'))
        if not wait:
            output('not waiting for builds to finish')
        elif self.connect == 'ssh':
            output('waiting for builds with ssh is not supported')
        else:
            self.running = defer.Deferred()
            if not self.buildsetStatus:
                output('try scheduler on the master does not have the builder configured')
                return None
            self._getStatus_1()
            if bool(self.config.get('dryrun')):
                self.statusDone()
            return self.running
        return None

    @defer.inlineCallbacks
    def _getStatus_1(self):
        if False:
            i = 10
            return i + 15
        brs = (yield self.buildsetStatus.callRemote('getBuildRequests'))
        self.builderNames = []
        self.buildRequests = {}
        self.builds = {}
        self.outstanding = []
        self.results = {}
        self.currentStep = {}
        self.ETA = {}
        for (n, br) in brs:
            self.builderNames.append(n)
            self.buildRequests[n] = br
            self.builds[n] = None
            self.outstanding.append(n)
            self.results[n] = [None, None]
            self.currentStep[n] = None
            self.ETA[n] = None
            br.callRemote('subscribe', self)
        if not self.getopt('quiet'):
            self.printloop = task.LoopingCall(self.printStatus)
            self.printloop.start(3, now=False)

    def remote_newbuild(self, bs, builderName):
        if False:
            i = 10
            return i + 15
        if self.builds[builderName]:
            self.builds[builderName].callRemote('unsubscribe', self)
        self.builds[builderName] = bs
        bs.callRemote('subscribe', self, 20)
        d = bs.callRemote('waitUntilFinished')
        d.addCallback(self._build_finished, builderName)

    def remote_stepStarted(self, buildername, build, stepname, step):
        if False:
            i = 10
            return i + 15
        self.currentStep[buildername] = stepname

    def remote_stepFinished(self, buildername, build, stepname, step, results):
        if False:
            for i in range(10):
                print('nop')
        pass

    def remote_buildETAUpdate(self, buildername, build, eta):
        if False:
            for i in range(10):
                print('nop')
        self.ETA[buildername] = now() + eta

    @defer.inlineCallbacks
    def _build_finished(self, bs, builderName):
        if False:
            while True:
                i = 10
        self.builds[builderName] = None
        self.ETA[builderName] = None
        self.currentStep[builderName] = 'finished'
        self.results[builderName][0] = (yield bs.callRemote('getResults'))
        self.results[builderName][1] = (yield bs.callRemote('getText'))
        self.outstanding.remove(builderName)
        if not self.outstanding:
            self.statusDone()

    def printStatus(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            names = sorted(self.buildRequests.keys())
            for n in names:
                if n not in self.outstanding:
                    (code, text) = self.results[n]
                    t = Results[code]
                    if text:
                        t += f" ({' '.join(text)})"
                elif self.builds[n]:
                    t = self.currentStep[n] or 'building'
                    if self.ETA[n]:
                        t += f' [ETA {self.ETA[n] - now()}s]'
                else:
                    t = 'no build'
                self.announce(f'{n}: {t}')
            self.announce('')
        except Exception:
            log.err(None, 'printing status')

    def statusDone(self):
        if False:
            return 10
        if self.printloop:
            self.printloop.stop()
            self.printloop = None
        output('All Builds Complete')
        names = sorted(self.buildRequests.keys())
        happy = True
        for n in names:
            (code, text) = self.results[n]
            t = f'{n}: {Results[code]}'
            if text:
                t += f" ({' '.join(text)})"
            output(t)
            if code != SUCCESS:
                happy = False
        if happy:
            self.exitcode = 0
        else:
            self.exitcode = 1
        self.running.callback(self.exitcode)

    @defer.inlineCallbacks
    def getAvailableBuilderNames(self):
        if False:
            print('Hello World!')
        if self.connect == 'pb':
            user = self.getopt('username')
            passwd = self.getopt('passwd')
            master = self.getopt('master')
            (tryhost, tryport) = master.split(':')
            tryport = int(tryport)
            f = pb.PBClientFactory()
            d = f.login(credentials.UsernamePassword(unicode2bytes(user), unicode2bytes(passwd)))
            reactor.connectTCP(tryhost, tryport, f)
            remote = (yield d)
            buildernames = (yield remote.callRemote('getAvailableBuilderNames'))
            output('The following builders are available for the try scheduler: ')
            for buildername in buildernames:
                output(buildername)
            yield remote.broker.transport.loseConnection()
            return
        if self.connect == 'ssh':
            output('Cannot get available builders over ssh.')
            sys.exit(1)
        raise RuntimeError(f"unknown connecttype '{self.connect}', should be 'pb'")

    def announce(self, message):
        if False:
            return 10
        if not self.quiet:
            output(message)

    @defer.inlineCallbacks
    def run_impl(self):
        if False:
            while True:
                i = 10
        output(f"using '{self.connect}' connect method")
        self.exitcode = 0
        yield fireEventually(None)
        try:
            if bool(self.config.get('get-builder-names')):
                yield self.getAvailableBuilderNames()
            else:
                yield self.createJob()
                yield self.announce('job created')
                if bool(self.config.get('dryrun')):
                    yield self.fakeDeliverJob()
                else:
                    yield self.deliverJob()
                yield self.announce('job has been delivered')
                yield self.getStatus()
            if not bool(self.config.get('dryrun')):
                yield self.cleanup()
        except SystemExit as e:
            self.exitcode = e.code
        except Exception as e:
            log.err(e)
            raise

    def run(self):
        if False:
            print('Hello World!')
        d = self.run_impl()
        d.addCallback(lambda res: reactor.stop())
        reactor.run()
        sys.exit(self.exitcode)

    def trapSystemExit(self, why):
        if False:
            for i in range(10):
                print('nop')
        why.trap(SystemExit)
        self.exitcode = why.value.code

    def cleanup(self, res=None):
        if False:
            print('Hello World!')
        if self.buildsetStatus:
            self.buildsetStatus.broker.transport.loseConnection()