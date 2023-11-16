import base64
import json
import os
from twisted.internet import defer
from twisted.protocols import basic
from twisted.python import log
from twisted.spread import pb
from buildbot import pbutil
from buildbot.process.properties import Properties
from buildbot.schedulers import base
from buildbot.util import bytes2unicode
from buildbot.util import netstrings
from buildbot.util import unicode2bytes
from buildbot.util.maildir import MaildirService

class TryBase(base.BaseScheduler):

    def filterBuilderList(self, builderNames):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make sure that C{builderNames} is a subset of the configured\n        C{self.builderNames}, returning an empty list if not.  If\n        C{builderNames} is empty, use C{self.builderNames}.\n\n        @returns: list of builder names to build on\n        '
        if builderNames:
            for b in builderNames:
                if b not in self.builderNames:
                    log.msg(f'{self} got with builder {b}')
                    log.msg(f" but that wasn't in our list: {self.builderNames}")
                    return []
        else:
            builderNames = self.builderNames
        return builderNames

class BadJobfile(Exception):
    pass

class JobdirService(MaildirService):
    name = 'JobdirService'

    def __init__(self, scheduler, basedir=None):
        if False:
            print('Hello World!')
        self.scheduler = scheduler
        super().__init__(basedir)

    def messageReceived(self, filename):
        if False:
            i = 10
            return i + 15
        with self.moveToCurDir(filename) as f:
            rv = self.scheduler.handleJobFile(filename, f)
        return rv

class Try_Jobdir(TryBase):
    compare_attrs = ('jobdir',)

    def __init__(self, name, builderNames, jobdir, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(name, builderNames, **kwargs)
        self.jobdir = jobdir
        self.watcher = JobdirService(scheduler=self)

    def addService(self, child):
        if False:
            return 10
        pass

    def removeService(self, child):
        if False:
            print('Hello World!')
        pass

    @defer.inlineCallbacks
    def activate(self):
        if False:
            i = 10
            return i + 15
        yield super().activate()
        if not self.enabled:
            return
        jobdir = os.path.join(self.master.basedir, self.jobdir)
        self.watcher.setBasedir(jobdir)
        for subdir in 'cur new tmp'.split():
            if not os.path.exists(os.path.join(jobdir, subdir)):
                os.mkdir(os.path.join(jobdir, subdir))
        self.watcher.startService()

    @defer.inlineCallbacks
    def deactivate(self):
        if False:
            while True:
                i = 10
        yield super().deactivate()
        if not self.enabled:
            return
        self.watcher.stopService()

    def parseJob(self, f):
        if False:
            return 10
        p = netstrings.NetstringParser()
        f.seek(0, 2)
        if f.tell() > basic.NetstringReceiver.MAX_LENGTH:
            raise BadJobfile('The patch size is greater that NetStringReceiver.MAX_LENGTH. Please Set this higher in the master.cfg')
        f.seek(0, 0)
        try:
            p.feed(f.read())
        except basic.NetstringParseError as e:
            raise BadJobfile('unable to parse netstrings') from e
        if not p.strings:
            raise BadJobfile('could not find any complete netstrings')
        ver = bytes2unicode(p.strings.pop(0))
        v1_keys = ['jobid', 'branch', 'baserev', 'patch_level', 'patch_body']
        v2_keys = v1_keys + ['repository', 'project']
        v3_keys = v2_keys + ['who']
        v4_keys = v3_keys + ['comment']
        keys = [v1_keys, v2_keys, v3_keys, v4_keys]
        parsed_job = {}

        def extract_netstrings(p, keys):
            if False:
                while True:
                    i = 10
            for (i, key) in enumerate(keys):
                if key == 'patch_body':
                    parsed_job[key] = p.strings[i]
                else:
                    parsed_job[key] = bytes2unicode(p.strings[i])

        def postprocess_parsed_job():
            if False:
                i = 10
                return i + 15
            parsed_job['branch'] = parsed_job['branch'] or None
            parsed_job['baserev'] = parsed_job['baserev'] or None
            parsed_job['patch_level'] = int(parsed_job['patch_level'])
            for key in 'repository project who comment'.split():
                parsed_job[key] = parsed_job.get(key, '')
            parsed_job['properties'] = parsed_job.get('properties', {})
        if ver <= '4':
            i = int(ver) - 1
            extract_netstrings(p, keys[i])
            parsed_job['builderNames'] = [bytes2unicode(s) for s in p.strings[len(keys[i]):]]
            postprocess_parsed_job()
        elif ver == '5':
            try:
                data = bytes2unicode(p.strings[0])
                parsed_job = json.loads(data)
                parsed_job['patch_body'] = unicode2bytes(parsed_job['patch_body'])
            except ValueError as e:
                raise BadJobfile('unable to parse JSON') from e
            postprocess_parsed_job()
        elif ver == '6':
            try:
                data = bytes2unicode(p.strings[0])
                parsed_job = json.loads(data)
                parsed_job['patch_body'] = base64.b64decode(parsed_job['patch_body_base64'])
                del parsed_job['patch_body_base64']
            except ValueError as e:
                raise BadJobfile('unable to parse JSON') from e
            postprocess_parsed_job()
        else:
            raise BadJobfile(f"unknown version '{ver}'")
        return parsed_job

    def handleJobFile(self, filename, f):
        if False:
            for i in range(10):
                print('nop')
        try:
            parsed_job = self.parseJob(f)
            builderNames = parsed_job['builderNames']
        except BadJobfile:
            log.msg(f'{self} reports a bad jobfile in {filename}')
            log.err()
            return defer.succeed(None)
        builderNames = self.filterBuilderList(builderNames)
        if not builderNames:
            log.msg('incoming Try job did not specify any allowed builder names')
            return defer.succeed(None)
        who = ''
        if parsed_job['who']:
            who = parsed_job['who']
        comment = ''
        if parsed_job['comment']:
            comment = parsed_job['comment']
        sourcestamp = {'branch': parsed_job['branch'], 'codebase': '', 'revision': parsed_job['baserev'], 'patch_body': parsed_job['patch_body'], 'patch_level': parsed_job['patch_level'], 'patch_author': who, 'patch_comment': comment, 'patch_subdir': '', 'project': parsed_job['project'], 'repository': parsed_job['repository']}
        reason = "'try' job"
        if parsed_job['who']:
            reason += f" by user {bytes2unicode(parsed_job['who'])}"
        properties = parsed_job['properties']
        requested_props = Properties()
        requested_props.update(properties, 'try build')
        return self.addBuildsetForSourceStamps(sourcestamps=[sourcestamp], reason=reason, external_idstring=bytes2unicode(parsed_job['jobid']), builderNames=builderNames, priority=self.priority, properties=requested_props)

class RemoteBuildSetStatus(pb.Referenceable):

    def __init__(self, master, bsid, brids):
        if False:
            for i in range(10):
                print('nop')
        self.master = master
        self.bsid = bsid
        self.brids = brids

    @defer.inlineCallbacks
    def remote_getBuildRequests(self):
        if False:
            while True:
                i = 10
        brids = {}
        for (builderid, brid) in self.brids.items():
            builderDict = (yield self.master.data.get(('builders', builderid)))
            brids[builderDict['name']] = brid
        return [(n, RemoteBuildRequest(self.master, n, brid)) for (n, brid) in brids.items()]

class RemoteBuildRequest(pb.Referenceable):

    def __init__(self, master, builderName, brid):
        if False:
            while True:
                i = 10
        self.master = master
        self.builderName = builderName
        self.brid = brid
        self.consumer = None

    @defer.inlineCallbacks
    def remote_subscribe(self, subscriber):
        if False:
            print('Hello World!')
        brdict = (yield self.master.data.get(('buildrequests', self.brid)))
        if not brdict:
            return
        builderId = brdict['builderid']
        reportedBuilds = set([])

        def gotBuild(key, msg):
            if False:
                i = 10
                return i + 15
            if msg['buildrequestid'] != self.brid or key[-1] != 'new':
                return None
            if msg['buildid'] in reportedBuilds:
                return None
            reportedBuilds.add(msg['buildid'])
            return subscriber.callRemote('newbuild', RemoteBuild(self.master, msg, self.builderName), self.builderName)
        self.consumer = (yield self.master.mq.startConsuming(gotBuild, ('builders', str(builderId), 'builds', None, None)))
        subscriber.notifyOnDisconnect(lambda _: self.remote_unsubscribe(subscriber))
        builds = (yield self.master.data.get(('buildrequests', self.brid, 'builds')))
        for build in builds:
            if build['buildid'] in reportedBuilds:
                continue
            reportedBuilds.add(build['buildid'])
            yield subscriber.callRemote('newbuild', RemoteBuild(self.master, build, self.builderName), self.builderName)

    def remote_unsubscribe(self, subscriber):
        if False:
            while True:
                i = 10
        if self.consumer:
            self.consumer.stopConsuming()
            self.consumer = None

class RemoteBuild(pb.Referenceable):

    def __init__(self, master, builddict, builderName):
        if False:
            while True:
                i = 10
        self.master = master
        self.builddict = builddict
        self.builderName = builderName
        self.consumer = None

    @defer.inlineCallbacks
    def remote_subscribe(self, subscriber, interval):
        if False:
            return 10

        def stepChanged(key, msg):
            if False:
                for i in range(10):
                    print('nop')
            if key[-1] == 'started':
                return subscriber.callRemote('stepStarted', self.builderName, self, msg['name'], None)
            elif key[-1] == 'finished':
                return subscriber.callRemote('stepFinished', self.builderName, self, msg['name'], None, msg['results'])
            return None
        self.consumer = (yield self.master.mq.startConsuming(stepChanged, ('builds', str(self.builddict['buildid']), 'steps', None, None)))
        subscriber.notifyOnDisconnect(lambda _: self.remote_unsubscribe(subscriber))

    def remote_unsubscribe(self, subscriber):
        if False:
            for i in range(10):
                print('nop')
        if self.consumer:
            self.consumer.stopConsuming()
            self.consumer = None

    @defer.inlineCallbacks
    def remote_waitUntilFinished(self):
        if False:
            while True:
                i = 10
        d = defer.Deferred()

        def buildEvent(key, msg):
            if False:
                for i in range(10):
                    print('nop')
            if key[-1] == 'finished':
                d.callback(None)
        consumer = (yield self.master.mq.startConsuming(buildEvent, ('builds', str(self.builddict['buildid']), None)))
        yield d
        consumer.stopConsuming()
        return self

    @defer.inlineCallbacks
    def remote_getResults(self):
        if False:
            print('Hello World!')
        buildid = self.builddict['buildid']
        builddict = (yield self.master.data.get(('builds', buildid)))
        return builddict['results']

    @defer.inlineCallbacks
    def remote_getText(self):
        if False:
            i = 10
            return i + 15
        buildid = self.builddict['buildid']
        builddict = (yield self.master.data.get(('builds', buildid)))
        return [builddict['state_string']]

class Try_Userpass_Perspective(pbutil.NewCredPerspective):

    def __init__(self, scheduler, username):
        if False:
            i = 10
            return i + 15
        self.scheduler = scheduler
        self.username = username

    @defer.inlineCallbacks
    def perspective_try(self, branch, revision, patch, repository, project, builderNames, who='', comment='', properties=None):
        if False:
            while True:
                i = 10
        log.msg(f'user {self.username} requesting build on builders {builderNames}')
        if properties is None:
            properties = {}
        builderNames = self.scheduler.filterBuilderList(builderNames)
        if not builderNames:
            return None
        branch = bytes2unicode(branch)
        revision = bytes2unicode(revision)
        patch_level = patch[0]
        patch_body = unicode2bytes(patch[1])
        repository = bytes2unicode(repository)
        project = bytes2unicode(project)
        who = bytes2unicode(who)
        comment = bytes2unicode(comment)
        reason = "'try' job"
        if who:
            reason += f' by user {bytes2unicode(who)}'
        if comment:
            reason += f' ({bytes2unicode(comment)})'
        sourcestamp = {'branch': branch, 'revision': revision, 'repository': repository, 'project': project, 'patch_level': patch_level, 'patch_body': patch_body, 'patch_subdir': '', 'patch_author': who or '', 'patch_comment': comment or '', 'codebase': ''}
        requested_props = Properties()
        requested_props.update(properties, 'try build')
        (bsid, brids) = (yield self.scheduler.addBuildsetForSourceStamps(sourcestamps=[sourcestamp], reason=reason, properties=requested_props, builderNames=builderNames))
        bss = RemoteBuildSetStatus(self.scheduler.master, bsid, brids)
        return bss

    def perspective_getAvailableBuilderNames(self):
        if False:
            return 10
        return self.scheduler.listBuilderNames()

class Try_Userpass(TryBase):
    compare_attrs = ('name', 'builderNames', 'port', 'userpass', 'properties')

    def __init__(self, name, builderNames, port, userpass, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(name, builderNames, **kwargs)
        self.port = port
        self.userpass = userpass
        self.registrations = []

    @defer.inlineCallbacks
    def activate(self):
        if False:
            while True:
                i = 10
        yield super().activate()
        if not self.enabled:
            return

        def factory(mind, username):
            if False:
                i = 10
                return i + 15
            return Try_Userpass_Perspective(self, username)
        for (user, passwd) in self.userpass:
            reg = (yield self.master.pbmanager.register(self.port, user, passwd, factory))
            self.registrations.append(reg)

    @defer.inlineCallbacks
    def deactivate(self):
        if False:
            print('Hello World!')
        yield super().deactivate()
        if not self.enabled:
            return
        yield defer.gatherResults([reg.unregister() for reg in self.registrations])