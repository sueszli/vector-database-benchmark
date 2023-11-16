import json
from twisted.internet import defer
from twisted.python import failure
from buildbot.data import connector
from buildbot.data import resultspec
from buildbot.db.buildrequests import AlreadyClaimedError
from buildbot.test.util import validation
from buildbot.util import service

class FakeUpdates(service.AsyncService):

    def __init__(self, testcase):
        if False:
            for i in range(10):
                print('nop')
        self.testcase = testcase
        self.changesAdded = []
        self.changesourceIds = {}
        self.buildsetsAdded = []
        self.maybeBuildsetCompleteCalls = 0
        self.masterStateChanges = []
        self.schedulerIds = {}
        self.builderIds = {}
        self.schedulerMasters = {}
        self.changesourceMasters = {}
        self.workerIds = {}
        self.logs = {}
        self.claimedBuildRequests = set([])
        self.stepStateString = {}
        self.stepUrls = {}
        self.properties = []
        self.missingWorkers = []

    def assertProperties(self, sourced, properties):
        if False:
            return 10
        self.testcase.assertIsInstance(properties, dict)
        for (k, v) in properties.items():
            self.testcase.assertIsInstance(k, str)
            if sourced:
                self.testcase.assertIsInstance(v, tuple)
                self.testcase.assertEqual(len(v), 2)
                (propval, propsrc) = v
                self.testcase.assertIsInstance(propsrc, str)
            else:
                propval = v
            try:
                json.dumps(propval)
            except (TypeError, ValueError):
                self.testcase.fail(f'value for {k} is not JSON-able')

    def addChange(self, files=None, comments=None, author=None, committer=None, revision=None, when_timestamp=None, branch=None, category=None, revlink='', properties=None, repository='', codebase=None, project='', src=None):
        if False:
            i = 10
            return i + 15
        if properties is None:
            properties = {}
        if files is not None:
            self.testcase.assertIsInstance(files, list)
            map(lambda f: self.testcase.assertIsInstance(f, str), files)
        self.testcase.assertIsInstance(comments, (type(None), str))
        self.testcase.assertIsInstance(author, (type(None), str))
        self.testcase.assertIsInstance(committer, (type(None), str))
        self.testcase.assertIsInstance(revision, (type(None), str))
        self.testcase.assertIsInstance(when_timestamp, (type(None), int))
        self.testcase.assertIsInstance(branch, (type(None), str))
        if callable(category):
            pre_change = self.master.config.preChangeGenerator(author=author, committer=committer, files=files, comments=comments, revision=revision, when_timestamp=when_timestamp, branch=branch, revlink=revlink, properties=properties, repository=repository, project=project)
            category = category(pre_change)
        self.testcase.assertIsInstance(category, (type(None), str))
        self.testcase.assertIsInstance(revlink, (type(None), str))
        self.assertProperties(sourced=False, properties=properties)
        self.testcase.assertIsInstance(repository, str)
        self.testcase.assertIsInstance(codebase, (type(None), str))
        self.testcase.assertIsInstance(project, str)
        self.testcase.assertIsInstance(src, (type(None), str))
        self.changesAdded.append(locals())
        self.changesAdded[-1].pop('self')
        return defer.succeed(len(self.changesAdded))

    def masterActive(self, name, masterid):
        if False:
            return 10
        self.testcase.assertIsInstance(name, str)
        self.testcase.assertIsInstance(masterid, int)
        if masterid:
            self.testcase.assertEqual(masterid, 1)
        self.thisMasterActive = True
        return defer.succeed(None)

    def masterStopped(self, name, masterid):
        if False:
            for i in range(10):
                print('nop')
        self.testcase.assertIsInstance(name, str)
        self.testcase.assertEqual(masterid, 1)
        self.thisMasterActive = False
        return defer.succeed(None)

    def expireMasters(self, forceHouseKeeping=False):
        if False:
            for i in range(10):
                print('nop')
        return defer.succeed(None)

    @defer.inlineCallbacks
    def addBuildset(self, waited_for, scheduler=None, sourcestamps=None, reason='', properties=None, builderids=None, external_idstring=None, parent_buildid=None, parent_relationship=None, priority=0):
        if False:
            for i in range(10):
                print('nop')
        if sourcestamps is None:
            sourcestamps = []
        if properties is None:
            properties = {}
        if builderids is None:
            builderids = []
        self.testcase.assertIsInstance(scheduler, str)
        self.testcase.assertIsInstance(sourcestamps, list)
        for ss in sourcestamps:
            if not isinstance(ss, int) and (not isinstance(ss, dict)):
                self.testcase.fail(f'{ss} ({type(ss)}) is not an integer or a dictionary')
            del ss
        self.testcase.assertIsInstance(reason, str)
        self.assertProperties(sourced=True, properties=properties)
        self.testcase.assertIsInstance(builderids, list)
        self.testcase.assertIsInstance(external_idstring, (type(None), str))
        self.buildsetsAdded.append(locals())
        self.buildsetsAdded[-1].pop('self')
        (bsid, brids) = (yield self.master.db.buildsets.addBuildset(sourcestamps=sourcestamps, reason=reason, properties=properties, builderids=builderids, waited_for=waited_for, external_idstring=external_idstring, parent_buildid=parent_buildid, parent_relationship=parent_relationship))
        return (bsid, brids)

    def maybeBuildsetComplete(self, bsid):
        if False:
            return 10
        self.maybeBuildsetCompleteCalls += 1
        return defer.succeed(None)

    @defer.inlineCallbacks
    def claimBuildRequests(self, brids, claimed_at=None):
        if False:
            return 10
        validation.verifyType(self.testcase, 'brids', brids, validation.ListValidator(validation.IntValidator()))
        validation.verifyType(self.testcase, 'claimed_at', claimed_at, validation.NoneOk(validation.DateTimeValidator()))
        if not brids:
            return True
        try:
            yield self.master.db.buildrequests.claimBuildRequests(brids=brids, claimed_at=claimed_at)
        except AlreadyClaimedError:
            return False
        self.claimedBuildRequests.update(set(brids))
        return True

    @defer.inlineCallbacks
    def unclaimBuildRequests(self, brids):
        if False:
            for i in range(10):
                print('nop')
        validation.verifyType(self.testcase, 'brids', brids, validation.ListValidator(validation.IntValidator()))
        self.claimedBuildRequests.difference_update(set(brids))
        if brids:
            yield self.master.db.buildrequests.unclaimBuildRequests(brids)

    def completeBuildRequests(self, brids, results, complete_at=None):
        if False:
            i = 10
            return i + 15
        validation.verifyType(self.testcase, 'brids', brids, validation.ListValidator(validation.IntValidator()))
        validation.verifyType(self.testcase, 'results', results, validation.IntValidator())
        validation.verifyType(self.testcase, 'complete_at', complete_at, validation.NoneOk(validation.DateTimeValidator()))
        return defer.succeed(True)

    def rebuildBuildrequest(self, buildrequest):
        if False:
            print('Hello World!')
        return defer.succeed(None)

    @defer.inlineCallbacks
    def update_project_info(self, projectid, slug, description, description_format, description_html):
        if False:
            return 10
        yield self.master.db.projects.update_project_info(projectid, slug, description, description_format, description_html)

    def find_project_id(self, name):
        if False:
            for i in range(10):
                print('nop')
        validation.verifyType(self.testcase, 'project name', name, validation.StringValidator())
        return self.master.db.projects.find_project_id(name)

    def updateBuilderList(self, masterid, builderNames):
        if False:
            while True:
                i = 10
        self.testcase.assertEqual(masterid, self.master.masterid)
        for n in builderNames:
            self.testcase.assertIsInstance(n, str)
        self.builderNames = builderNames
        return defer.succeed(None)

    @defer.inlineCallbacks
    def updateBuilderInfo(self, builderid, description, description_format, description_html, projectid, tags):
        if False:
            i = 10
            return i + 15
        yield self.master.db.builders.updateBuilderInfo(builderid, description, description_format, description_html, projectid, tags)

    def masterDeactivated(self, masterid):
        if False:
            i = 10
            return i + 15
        return defer.succeed(None)

    def findSchedulerId(self, name):
        if False:
            while True:
                i = 10
        return self.master.db.schedulers.findSchedulerId(name)

    def forget_about_it(self, name):
        if False:
            while True:
                i = 10
        validation.verifyType(self.testcase, 'scheduler name', name, validation.StringValidator())
        if name not in self.schedulerIds:
            self.schedulerIds[name] = max([0] + list(self.schedulerIds.values())) + 1
        return defer.succeed(self.schedulerIds[name])

    def findChangeSourceId(self, name):
        if False:
            return 10
        validation.verifyType(self.testcase, 'changesource name', name, validation.StringValidator())
        if name not in self.changesourceIds:
            self.changesourceIds[name] = max([0] + list(self.changesourceIds.values())) + 1
        return defer.succeed(self.changesourceIds[name])

    def findBuilderId(self, name):
        if False:
            while True:
                i = 10
        validation.verifyType(self.testcase, 'builder name', name, validation.StringValidator())
        return self.master.db.builders.findBuilderId(name)

    def trySetSchedulerMaster(self, schedulerid, masterid):
        if False:
            return 10
        currentMasterid = self.schedulerMasters.get(schedulerid)
        if isinstance(currentMasterid, Exception):
            return defer.fail(failure.Failure(currentMasterid))
        if currentMasterid and masterid is not None:
            return defer.succeed(False)
        self.schedulerMasters[schedulerid] = masterid
        return defer.succeed(True)

    def trySetChangeSourceMaster(self, changesourceid, masterid):
        if False:
            return 10
        currentMasterid = self.changesourceMasters.get(changesourceid)
        if isinstance(currentMasterid, Exception):
            return defer.fail(failure.Failure(currentMasterid))
        if currentMasterid and masterid is not None:
            return defer.succeed(False)
        self.changesourceMasters[changesourceid] = masterid
        return defer.succeed(True)

    def addBuild(self, builderid, buildrequestid, workerid):
        if False:
            return 10
        validation.verifyType(self.testcase, 'builderid', builderid, validation.IntValidator())
        validation.verifyType(self.testcase, 'buildrequestid', buildrequestid, validation.IntValidator())
        validation.verifyType(self.testcase, 'workerid', workerid, validation.IntValidator())
        return defer.succeed((10, 1))

    def generateNewBuildEvent(self, buildid):
        if False:
            print('Hello World!')
        validation.verifyType(self.testcase, 'buildid', buildid, validation.IntValidator())
        return defer.succeed(None)

    def setBuildStateString(self, buildid, state_string):
        if False:
            for i in range(10):
                print('nop')
        validation.verifyType(self.testcase, 'buildid', buildid, validation.IntValidator())
        validation.verifyType(self.testcase, 'state_string', state_string, validation.StringValidator())
        return defer.succeed(None)

    def finishBuild(self, buildid, results):
        if False:
            while True:
                i = 10
        validation.verifyType(self.testcase, 'buildid', buildid, validation.IntValidator())
        validation.verifyType(self.testcase, 'results', results, validation.IntValidator())
        return defer.succeed(None)

    def setBuildProperty(self, buildid, name, value, source):
        if False:
            for i in range(10):
                print('nop')
        validation.verifyType(self.testcase, 'buildid', buildid, validation.IntValidator())
        validation.verifyType(self.testcase, 'name', name, validation.StringValidator())
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            self.testcase.fail(f'Value for {name} is not JSON-able')
        validation.verifyType(self.testcase, 'source', source, validation.StringValidator())
        return defer.succeed(None)

    @defer.inlineCallbacks
    def setBuildProperties(self, buildid, properties):
        if False:
            print('Hello World!')
        for (k, v, s) in properties.getProperties().asList():
            self.properties.append((buildid, k, v, s))
            yield self.setBuildProperty(buildid, k, v, s)

    def addStep(self, buildid, name):
        if False:
            print('Hello World!')
        validation.verifyType(self.testcase, 'buildid', buildid, validation.IntValidator())
        validation.verifyType(self.testcase, 'name', name, validation.IdentifierValidator(50))
        return defer.succeed((10, 1, name))

    def addStepURL(self, stepid, name, url):
        if False:
            print('Hello World!')
        validation.verifyType(self.testcase, 'stepid', stepid, validation.IntValidator())
        validation.verifyType(self.testcase, 'name', name, validation.StringValidator())
        validation.verifyType(self.testcase, 'url', url, validation.StringValidator())
        self.stepUrls.setdefault(stepid, []).append((name, url))
        return defer.succeed(None)

    def startStep(self, stepid):
        if False:
            return 10
        validation.verifyType(self.testcase, 'stepid', stepid, validation.IntValidator())
        return defer.succeed(None)

    def set_step_locks_acquired_at(self, stepid):
        if False:
            return 10
        validation.verifyType(self.testcase, 'stepid', stepid, validation.IntValidator())
        return defer.succeed(None)

    def setStepStateString(self, stepid, state_string):
        if False:
            i = 10
            return i + 15
        validation.verifyType(self.testcase, 'stepid', stepid, validation.IntValidator())
        validation.verifyType(self.testcase, 'state_string', state_string, validation.StringValidator())
        self.stepStateString[stepid] = state_string
        return defer.succeed(None)

    def finishStep(self, stepid, results, hidden):
        if False:
            return 10
        validation.verifyType(self.testcase, 'stepid', stepid, validation.IntValidator())
        validation.verifyType(self.testcase, 'results', results, validation.IntValidator())
        validation.verifyType(self.testcase, 'hidden', hidden, validation.BooleanValidator())
        return defer.succeed(None)

    def addLog(self, stepid, name, type):
        if False:
            return 10
        validation.verifyType(self.testcase, 'stepid', stepid, validation.IntValidator())
        validation.verifyType(self.testcase, 'name', name, validation.StringValidator())
        validation.verifyType(self.testcase, 'type', type, validation.IdentifierValidator(1))
        logid = max([0] + list(self.logs)) + 1
        self.logs[logid] = {'name': name, 'type': type, 'content': [], 'finished': False}
        return defer.succeed(logid)

    def finishLog(self, logid):
        if False:
            while True:
                i = 10
        validation.verifyType(self.testcase, 'logid', logid, validation.IntValidator())
        self.logs[logid]['finished'] = True
        return defer.succeed(None)

    def compressLog(self, logid):
        if False:
            while True:
                i = 10
        validation.verifyType(self.testcase, 'logid', logid, validation.IntValidator())
        return defer.succeed(None)

    def appendLog(self, logid, content):
        if False:
            return 10
        validation.verifyType(self.testcase, 'logid', logid, validation.IntValidator())
        validation.verifyType(self.testcase, 'content', content, validation.StringValidator())
        self.testcase.assertEqual(content[-1], '\n')
        self.logs[logid]['content'].append(content)
        return defer.succeed(None)

    def findWorkerId(self, name):
        if False:
            print('Hello World!')
        validation.verifyType(self.testcase, 'worker name', name, validation.IdentifierValidator(50))
        return self.master.db.workers.findWorkerId(name)

    def workerConnected(self, workerid, masterid, workerinfo):
        if False:
            i = 10
            return i + 15
        return self.master.db.workers.workerConnected(workerid=workerid, masterid=masterid, workerinfo=workerinfo)

    def workerConfigured(self, workerid, masterid, builderids):
        if False:
            for i in range(10):
                print('nop')
        return self.master.db.workers.workerConfigured(workerid=workerid, masterid=masterid, builderids=builderids)

    def workerDisconnected(self, workerid, masterid):
        if False:
            return 10
        return self.master.db.workers.workerDisconnected(workerid=workerid, masterid=masterid)

    def deconfigureAllWorkersForMaster(self, masterid):
        if False:
            while True:
                i = 10
        return self.master.db.workers.deconfigureAllWorkersForMaster(masterid=masterid)

    def workerMissing(self, workerid, masterid, last_connection, notify):
        if False:
            for i in range(10):
                print('nop')
        self.missingWorkers.append((workerid, masterid, last_connection, notify))

    def schedulerEnable(self, schedulerid, v):
        if False:
            i = 10
            return i + 15
        return self.master.db.schedulers.enable(schedulerid, v)

    @defer.inlineCallbacks
    def setWorkerState(self, workerid, paused, graceful):
        if False:
            i = 10
            return i + 15
        yield self.master.db.workers.set_worker_paused(workerid=workerid, paused=paused)
        yield self.master.db.workers.set_worker_graceful(workerid=workerid, graceful=graceful)

    def set_worker_paused(self, workerid, paused, pause_reason=None):
        if False:
            while True:
                i = 10
        return self.master.db.workers.set_worker_paused(workerid, paused, pause_reason=pause_reason)

    def set_worker_graceful(self, workerid, graceful):
        if False:
            return 10
        return self.master.db.workers.set_worker_graceful(workerid, graceful)

    @defer.inlineCallbacks
    def setBuildData(self, buildid, name, value, source):
        if False:
            return 10
        validation.verifyType(self.testcase, 'buildid', buildid, validation.IntValidator())
        validation.verifyType(self.testcase, 'name', name, validation.StringValidator())
        validation.verifyType(self.testcase, 'value', value, validation.BinaryValidator())
        validation.verifyType(self.testcase, 'source', source, validation.StringValidator())
        yield self.master.db.build_data.setBuildData(buildid, name, value, source)

    @defer.inlineCallbacks
    def addTestResultSet(self, builderid, buildid, stepid, description, category, value_unit):
        if False:
            print('Hello World!')
        validation.verifyType(self.testcase, 'builderid', builderid, validation.IntValidator())
        validation.verifyType(self.testcase, 'buildid', buildid, validation.IntValidator())
        validation.verifyType(self.testcase, 'stepid', stepid, validation.IntValidator())
        validation.verifyType(self.testcase, 'description', description, validation.StringValidator())
        validation.verifyType(self.testcase, 'category', category, validation.StringValidator())
        validation.verifyType(self.testcase, 'value_unit', value_unit, validation.StringValidator())
        test_result_setid = (yield self.master.db.test_result_sets.addTestResultSet(builderid, buildid, stepid, description, category, value_unit))
        return test_result_setid

    @defer.inlineCallbacks
    def completeTestResultSet(self, test_result_setid, tests_passed=None, tests_failed=None):
        if False:
            i = 10
            return i + 15
        validation.verifyType(self.testcase, 'test_result_setid', test_result_setid, validation.IntValidator())
        validation.verifyType(self.testcase, 'tests_passed', tests_passed, validation.NoneOk(validation.IntValidator()))
        validation.verifyType(self.testcase, 'tests_failed', tests_failed, validation.NoneOk(validation.IntValidator()))
        yield self.master.db.test_result_sets.completeTestResultSet(test_result_setid, tests_passed, tests_failed)

    @defer.inlineCallbacks
    def addTestResults(self, builderid, test_result_setid, result_values):
        if False:
            i = 10
            return i + 15
        yield self.master.db.test_results.addTestResults(builderid, test_result_setid, result_values)

class FakeDataConnector(service.AsyncMultiService):

    def __init__(self, master, testcase):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.setServiceParent(master)
        self.updates = FakeUpdates(testcase)
        self.updates.setServiceParent(self)
        self.realConnector = connector.DataConnector()
        self.realConnector.setServiceParent(self)
        self.rtypes = self.realConnector.rtypes
        self.plural_rtypes = self.realConnector.plural_rtypes

    def _scanModule(self, mod):
        if False:
            for i in range(10):
                print('nop')
        return self.realConnector._scanModule(mod)

    def getEndpoint(self, path):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(path, tuple):
            raise TypeError('path must be a tuple')
        return self.realConnector.getEndpoint(path)

    def getResourceType(self, name):
        if False:
            i = 10
            return i + 15
        return getattr(self.rtypes, name)

    def get(self, path, filters=None, fields=None, order=None, limit=None, offset=None):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(path, tuple):
            raise TypeError('path must be a tuple')
        return self.realConnector.get(path, filters=filters, fields=fields, order=order, limit=limit, offset=offset)

    def get_with_resultspec(self, path, rspec):
        if False:
            print('Hello World!')
        if not isinstance(path, tuple):
            raise TypeError('path must be a tuple')
        if not isinstance(rspec, resultspec.ResultSpec):
            raise TypeError('rspec must be ResultSpec')
        return self.realConnector.get_with_resultspec(path, rspec)

    def control(self, action, args, path):
        if False:
            return 10
        if not isinstance(path, tuple):
            raise TypeError('path must be a tuple')
        return self.realConnector.control(action, args, path)

    def resultspec_from_jsonapi(self, args, entityType, is_collection):
        if False:
            print('Hello World!')
        return self.realConnector.resultspec_from_jsonapi(args, entityType, is_collection)

    def getResourceTypeForGraphQlType(self, type):
        if False:
            return 10
        return self.realConnector.getResourceTypeForGraphQlType(type)