import warnings
from unittest.mock import Mock
from unittest.mock import call
from packaging.version import parse as parse_version
from twisted.internet import defer
from twisted.internet import error
from twisted.internet import reactor
from twisted.python import failure
from twisted.trial import unittest
from buildbot.process.results import FAILURE
from buildbot.process.results import RETRY
from buildbot.process.results import SUCCESS
from buildbot.reporters import utils
from buildbot.reporters.gerrit import GERRIT_LABEL_REVIEWED
from buildbot.reporters.gerrit import GERRIT_LABEL_VERIFIED
from buildbot.reporters.gerrit import GerritStatusPush
from buildbot.reporters.gerrit import defaultReviewCB
from buildbot.reporters.gerrit import defaultSummaryCB
from buildbot.reporters.gerrit import makeReviewResult
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util.reporter import ReporterTestMixin
warnings.filterwarnings('error', message='.*Gerrit status')

def sampleReviewCB(builderName, build, result, status, arg):
    if False:
        for i in range(10):
            print('nop')
    verified = 1 if result == SUCCESS else -1
    return makeReviewResult(str({'name': builderName, 'result': result}), (GERRIT_LABEL_VERIFIED, verified))

@defer.inlineCallbacks
def sampleReviewCBDeferred(builderName, build, result, status, arg):
    if False:
        print('Hello World!')
    verified = 1 if result == SUCCESS else -1
    result = (yield makeReviewResult(str({'name': builderName, 'result': result}), (GERRIT_LABEL_VERIFIED, verified)))
    return result

def sampleStartCB(builderName, build, arg):
    if False:
        print('Hello World!')
    return makeReviewResult(str({'name': builderName}), (GERRIT_LABEL_REVIEWED, 0))

@defer.inlineCallbacks
def sampleStartCBDeferred(builderName, build, arg):
    if False:
        i = 10
        return i + 15
    result = (yield makeReviewResult(str({'name': builderName}), (GERRIT_LABEL_REVIEWED, 0)))
    return result

def sampleSummaryCB(buildInfoList, results, status, arg):
    if False:
        for i in range(10):
            print('nop')
    success = False
    failure = False
    for buildInfo in buildInfoList:
        if buildInfo['result'] == SUCCESS:
            success = True
        else:
            failure = True
    if failure:
        verified = -1
    elif success:
        verified = 1
    else:
        verified = 0
    return makeReviewResult(str(buildInfoList), (GERRIT_LABEL_VERIFIED, verified))

@defer.inlineCallbacks
def sampleSummaryCBDeferred(buildInfoList, results, master, arg):
    if False:
        print('Hello World!')
    success = False
    failure = False
    for buildInfo in buildInfoList:
        if buildInfo['result'] == SUCCESS:
            success = True
        else:
            failure = True
    if failure:
        verified = -1
    elif success:
        verified = 1
    else:
        verified = 0
    result = (yield makeReviewResult(str(buildInfoList), (GERRIT_LABEL_VERIFIED, verified)))
    return result

def legacyTestReviewCB(builderName, build, result, status, arg):
    if False:
        while True:
            i = 10
    msg = str({'name': builderName, 'result': result})
    return (msg, 1 if result == SUCCESS else -1, 0)

def legacyTestSummaryCB(buildInfoList, results, status, arg):
    if False:
        for i in range(10):
            print('nop')
    success = False
    failure = False
    for buildInfo in buildInfoList:
        if buildInfo['result'] == SUCCESS:
            success = True
        else:
            failure = True
    if failure:
        verified = -1
    elif success:
        verified = 1
    else:
        verified = 0
    return (str(buildInfoList), verified, 0)

class TestGerritStatusPush(TestReactorMixin, unittest.TestCase, ReporterTestMixin):

    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        self.setup_reporter_test()
        self.master = fakemaster.make_master(self, wantData=True, wantDb=True, wantMq=True)

    @defer.inlineCallbacks
    def setupGerritStatusPushSimple(self, *args, **kwargs):
        if False:
            return 10
        serv = kwargs.pop('server', 'serv')
        username = kwargs.pop('username', 'user')
        gsp = GerritStatusPush(serv, username, *args, **kwargs)
        yield gsp.setServiceParent(self.master)
        yield gsp.startService()
        return gsp

    @defer.inlineCallbacks
    def setupGerritStatusPush(self, *args, **kwargs):
        if False:
            print('Hello World!')
        gsp = (yield self.setupGerritStatusPushSimple(*args, **kwargs))
        gsp.sendCodeReview = Mock()
        return gsp

    @defer.inlineCallbacks
    def setupBuildResults(self, buildResults, finalResult):
        if False:
            while True:
                i = 10
        self.insert_test_data(buildResults, finalResult)
        res = (yield utils.getDetailsForBuildset(self.master, 98, want_properties=True))
        builds = res['builds']
        buildset = res['buildset']

        @defer.inlineCallbacks
        def getChangesForBuild(buildid):
            if False:
                return 10
            assert buildid == 20
            ch = (yield self.master.db.changes.getChange(13))
            return [ch]
        self.master.db.changes.getChangesForBuild = getChangesForBuild
        return (buildset, builds)

    def makeBuildInfo(self, buildResults, resultText, builds):
        if False:
            for i in range(10):
                print('nop')
        info = []
        for (i, buildResult) in enumerate(buildResults):
            info.append({'name': f'Builder{i}', 'result': buildResult, 'resultText': resultText[i], 'text': 'buildText', 'url': f'http://localhost:8080/#/builders/{79 + i}/builds/{i}', 'build': builds[i]})
        return info

    @defer.inlineCallbacks
    def run_fake_summary_build(self, gsp, buildResults, finalResult, resultText, expWarning=False):
        if False:
            return 10
        (buildset, builds) = (yield self.setupBuildResults(buildResults, finalResult))
        yield gsp.buildsetComplete('buildset.98.complete'.split('.'), buildset)
        info = self.makeBuildInfo(buildResults, resultText, builds)
        if expWarning:
            self.assertEqual([w['message'] for w in self.flushWarnings()], ['The Gerrit status callback uses the old way to communicate results.  The outcome might be not what is expected.'])
        return str(info)

    @defer.inlineCallbacks
    def check_summary_build_deferred(self, buildResults, finalResult, resultText, verifiedScore):
        if False:
            i = 10
            return i + 15
        gsp = (yield self.setupGerritStatusPush(summaryCB=sampleSummaryCBDeferred))
        msg = (yield self.run_fake_summary_build(gsp, buildResults, finalResult, resultText))
        result = makeReviewResult(msg, (GERRIT_LABEL_VERIFIED, verifiedScore))
        gsp.sendCodeReview.assert_called_once_with(self.reporter_test_project, self.reporter_test_revision, result)

    @defer.inlineCallbacks
    def check_summary_build(self, buildResults, finalResult, resultText, verifiedScore):
        if False:
            return 10
        gsp = (yield self.setupGerritStatusPush(summaryCB=sampleSummaryCB))
        msg = (yield self.run_fake_summary_build(gsp, buildResults, finalResult, resultText))
        result = makeReviewResult(msg, (GERRIT_LABEL_VERIFIED, verifiedScore))
        gsp.sendCodeReview.assert_called_once_with(self.reporter_test_project, self.reporter_test_revision, result)

    @defer.inlineCallbacks
    def check_summary_build_legacy(self, buildResults, finalResult, resultText, verifiedScore):
        if False:
            while True:
                i = 10
        gsp = (yield self.setupGerritStatusPush(summaryCB=legacyTestSummaryCB))
        msg = (yield self.run_fake_summary_build(gsp, buildResults, finalResult, resultText, expWarning=True))
        result = makeReviewResult(msg, (GERRIT_LABEL_VERIFIED, verifiedScore), (GERRIT_LABEL_REVIEWED, 0))
        gsp.sendCodeReview.assert_called_once_with(self.reporter_test_project, self.reporter_test_revision, result)

    @defer.inlineCallbacks
    def test_gerrit_ssh_cmd(self):
        if False:
            for i in range(10):
                print('nop')
        kwargs = {'server': 'example.com', 'username': 'buildbot'}
        without_identity = (yield self.setupGerritStatusPush(**kwargs))
        expected1 = ['ssh', '-o', 'BatchMode=yes', 'buildbot@example.com', '-p', '29418', 'gerrit', 'foo']
        self.assertEqual(expected1, without_identity._gerritCmd('foo'))
        yield without_identity.disownServiceParent()
        with_identity = (yield self.setupGerritStatusPush(identity_file='/path/to/id_rsa', **kwargs))
        expected2 = ['ssh', '-o', 'BatchMode=yes', '-i', '/path/to/id_rsa', 'buildbot@example.com', '-p', '29418', 'gerrit', 'foo']
        self.assertEqual(expected2, with_identity._gerritCmd('foo'))

    def test_buildsetComplete_success_sends_summary_review_deferred(self):
        if False:
            return 10
        d = self.check_summary_build_deferred(buildResults=[SUCCESS, SUCCESS], finalResult=SUCCESS, resultText=['succeeded', 'succeeded'], verifiedScore=1)
        return d

    def test_buildsetComplete_success_sends_summary_review(self):
        if False:
            while True:
                i = 10
        d = self.check_summary_build(buildResults=[SUCCESS, SUCCESS], finalResult=SUCCESS, resultText=['succeeded', 'succeeded'], verifiedScore=1)
        return d

    def test_buildsetComplete_failure_sends_summary_review(self):
        if False:
            print('Hello World!')
        d = self.check_summary_build(buildResults=[FAILURE, FAILURE], finalResult=FAILURE, resultText=['failed', 'failed'], verifiedScore=-1)
        return d

    def test_buildsetComplete_mixed_sends_summary_review(self):
        if False:
            while True:
                i = 10
        d = self.check_summary_build(buildResults=[SUCCESS, FAILURE], finalResult=FAILURE, resultText=['succeeded', 'failed'], verifiedScore=-1)
        return d

    def test_buildsetComplete_success_sends_summary_review_legacy(self):
        if False:
            return 10
        d = self.check_summary_build_legacy(buildResults=[SUCCESS, SUCCESS], finalResult=SUCCESS, resultText=['succeeded', 'succeeded'], verifiedScore=1)
        return d

    def test_buildsetComplete_failure_sends_summary_review_legacy(self):
        if False:
            print('Hello World!')
        d = self.check_summary_build_legacy(buildResults=[FAILURE, FAILURE], finalResult=FAILURE, resultText=['failed', 'failed'], verifiedScore=-1)
        return d

    def test_buildsetComplete_mixed_sends_summary_review_legacy(self):
        if False:
            while True:
                i = 10
        d = self.check_summary_build_legacy(buildResults=[SUCCESS, FAILURE], finalResult=FAILURE, resultText=['succeeded', 'failed'], verifiedScore=-1)
        return d

    @defer.inlineCallbacks
    def test_buildsetComplete_filtered_builder(self):
        if False:
            print('Hello World!')
        gsp = (yield self.setupGerritStatusPush(summaryCB=sampleSummaryCB))
        gsp.builders = ['foo']
        yield self.run_fake_summary_build(gsp, [FAILURE, FAILURE], FAILURE, ['failed', 'failed'])
        self.assertFalse(gsp.sendCodeReview.called, 'sendCodeReview should not be called')

    @defer.inlineCallbacks
    def test_buildsetComplete_filtered_matching_builder(self):
        if False:
            print('Hello World!')
        gsp = (yield self.setupGerritStatusPush(summaryCB=sampleSummaryCB))
        gsp.builders = ['Builder1']
        yield self.run_fake_summary_build(gsp, [FAILURE, FAILURE], FAILURE, ['failed', 'failed'])
        self.assertTrue(gsp.sendCodeReview.called, 'sendCodeReview should be called')

    @defer.inlineCallbacks
    def run_fake_single_build(self, gsp, buildResult, expWarning=False):
        if False:
            i = 10
            return i + 15
        (_, builds) = (yield self.setupBuildResults([buildResult], buildResult))
        yield gsp._got_event(('builds', builds[0]['buildid'], 'new'), builds[0])
        yield gsp._got_event(('builds', builds[0]['buildid'], 'finished'), builds[0])
        if expWarning:
            self.assertEqual([w['message'] for w in self.flushWarnings()], ['The Gerrit status callback uses the old way to communicate results.  The outcome might be not what is expected.'])
        return str({'name': 'Builder0', 'result': buildResult})

    @defer.inlineCallbacks
    def check_single_build(self, buildResult, verifiedScore):
        if False:
            for i in range(10):
                print('nop')
        gsp = (yield self.setupGerritStatusPush(reviewCB=sampleReviewCB, startCB=sampleStartCB))
        msg = (yield self.run_fake_single_build(gsp, buildResult))
        start = makeReviewResult(str({'name': self.reporter_test_builder_name}), (GERRIT_LABEL_REVIEWED, 0))
        result = makeReviewResult(msg, (GERRIT_LABEL_VERIFIED, verifiedScore))
        calls = [call(self.reporter_test_project, self.reporter_test_revision, start), call(self.reporter_test_project, self.reporter_test_revision, result)]
        gsp.sendCodeReview.assert_has_calls(calls)

    @defer.inlineCallbacks
    def check_single_build_deferred(self, buildResult, verifiedScore):
        if False:
            i = 10
            return i + 15
        gsp = (yield self.setupGerritStatusPush(reviewCB=sampleReviewCBDeferred, startCB=sampleStartCBDeferred))
        msg = (yield self.run_fake_single_build(gsp, buildResult))
        start = makeReviewResult(str({'name': self.reporter_test_builder_name}), (GERRIT_LABEL_REVIEWED, 0))
        result = makeReviewResult(msg, (GERRIT_LABEL_VERIFIED, verifiedScore))
        calls = [call(self.reporter_test_project, self.reporter_test_revision, start), call(self.reporter_test_project, self.reporter_test_revision, result)]
        gsp.sendCodeReview.assert_has_calls(calls)

    @defer.inlineCallbacks
    def check_single_build_legacy(self, buildResult, verifiedScore):
        if False:
            for i in range(10):
                print('nop')
        gsp = (yield self.setupGerritStatusPush(reviewCB=legacyTestReviewCB, startCB=sampleStartCB))
        msg = (yield self.run_fake_single_build(gsp, buildResult, expWarning=True))
        start = makeReviewResult(str({'name': self.reporter_test_builder_name}), (GERRIT_LABEL_REVIEWED, 0))
        result = makeReviewResult(msg, (GERRIT_LABEL_VERIFIED, verifiedScore), (GERRIT_LABEL_REVIEWED, 0))
        calls = [call(self.reporter_test_project, self.reporter_test_revision, start), call(self.reporter_test_project, self.reporter_test_revision, result)]
        gsp.sendCodeReview.assert_has_calls(calls)

    def test_buildComplete_success_sends_review(self):
        if False:
            for i in range(10):
                print('nop')
        return self.check_single_build(SUCCESS, 1)

    def test_buildComplete_failure_sends_review(self):
        if False:
            for i in range(10):
                print('nop')
        return self.check_single_build(FAILURE, -1)

    def test_buildComplete_success_sends_review_legacy(self):
        if False:
            i = 10
            return i + 15
        return self.check_single_build_legacy(SUCCESS, 1)

    def test_buildComplete_failure_sends_review_legacy(self):
        if False:
            return 10
        return self.check_single_build_legacy(FAILURE, -1)

    @defer.inlineCallbacks
    def test_single_build_filtered(self):
        if False:
            print('Hello World!')
        gsp = (yield self.setupGerritStatusPush(reviewCB=sampleReviewCB, startCB=sampleStartCB))
        gsp.builders = ['Builder0']
        yield self.run_fake_single_build(gsp, SUCCESS)
        self.assertTrue(gsp.sendCodeReview.called, 'sendCodeReview should be called')
        gsp.sendCodeReview = Mock()
        gsp.builders = ['foo']
        yield self.run_fake_single_build(gsp, SUCCESS)
        self.assertFalse(gsp.sendCodeReview.called, 'sendCodeReview should not be called')

    def test_defaultReviewCBSuccess(self):
        if False:
            for i in range(10):
                print('nop')
        res = defaultReviewCB('builderName', {}, SUCCESS, None, None)
        self.assertEqual(res['labels'], {'Verified': 1})
        res = defaultReviewCB('builderName', {}, RETRY, None, None)
        self.assertEqual(res['labels'], {})

    def test_defaultSummaryCB(self):
        if False:
            while True:
                i = 10
        info = self.makeBuildInfo([SUCCESS, FAILURE], ['yes', 'no'], [None, None])
        res = defaultSummaryCB(info, SUCCESS, None, None)
        self.assertEqual(res['labels'], {'Verified': -1})
        info = self.makeBuildInfo([SUCCESS, SUCCESS], ['yes', 'yes'], [None, None])
        res = defaultSummaryCB(info, SUCCESS, None, None)
        self.assertEqual(res['labels'], {'Verified': 1})

    @defer.inlineCallbacks
    def testBuildGerritCommand(self):
        if False:
            i = 10
            return i + 15
        gsp = (yield self.setupGerritStatusPushSimple())
        spawnSkipFirstArg = Mock()
        gsp.spawnProcess = lambda _, *a, **k: spawnSkipFirstArg(*a, **k)
        yield gsp.sendCodeReview('project', 'revision', {'message': 'bla', 'labels': {'Verified': 1}})
        spawnSkipFirstArg.assert_called_once_with('ssh', ['ssh', '-o', 'BatchMode=yes', 'user@serv', '-p', '29418', 'gerrit', 'version'], env=None)
        gsp.processVersion(parse_version('2.6'), lambda : None)
        spawnSkipFirstArg = Mock()
        yield gsp.sendCodeReview('project', 'revision', {'message': 'bla', 'labels': {'Verified': 1}})
        spawnSkipFirstArg.assert_called_once_with('ssh', ['ssh', '-o', 'BatchMode=yes', 'user@serv', '-p', '29418', 'gerrit', 'review', '--project project', "--message 'bla'", '--label Verified=1', 'revision'], env=None)
        gsp.processVersion(parse_version('2.4'), lambda : None)
        spawnSkipFirstArg = Mock()
        yield gsp.sendCodeReview('project', 'revision', {'message': 'bla', 'labels': {'Verified': 1}})
        spawnSkipFirstArg.assert_called_once_with('ssh', ['ssh', '-o', 'BatchMode=yes', 'user@serv', '-p', '29418', 'gerrit', 'review', '--project project', "--message 'bla'", '--verified 1', 'revision'], env=None)
        gsp._gerrit_notify = 'OWNER'
        gsp.processVersion(parse_version('2.6'), lambda : None)
        spawnSkipFirstArg = Mock()
        yield gsp.sendCodeReview('project', 'revision', {'message': 'bla', 'labels': {'Verified': 1}})
        spawnSkipFirstArg.assert_called_once_with('ssh', ['ssh', '-o', 'BatchMode=yes', 'user@serv', '-p', '29418', 'gerrit', 'review', '--project project', '--notify OWNER', "--message 'bla'", '--label Verified=1', 'revision'], env=None)
        gsp.processVersion(parse_version('2.4'), lambda : None)
        spawnSkipFirstArg = Mock()
        yield gsp.sendCodeReview('project', 'revision', {'message': 'bla', 'labels': {'Verified': 1}})
        spawnSkipFirstArg.assert_called_once_with('ssh', ['ssh', '-o', 'BatchMode=yes', 'user@serv', '-p', '29418', 'gerrit', 'review', '--project project', '--notify OWNER', "--message 'bla'", '--verified 1', 'revision'], env=None)
        gsp.processVersion(parse_version('2.13'), lambda : None)
        spawnSkipFirstArg = Mock()
        yield gsp.sendCodeReview('project', 'revision', {'message': 'bla', 'labels': {'Verified': 1}})
        spawnSkipFirstArg.assert_called_once_with('ssh', ['ssh', '-o', 'BatchMode=yes', 'user@serv', '-p', '29418', 'gerrit', 'review', '--project project', '--tag autogenerated:buildbot', '--notify OWNER', "--message 'bla'", '--label Verified=1', 'revision'], env=None)

    @defer.inlineCallbacks
    def test_callWithVersion_bytes_output(self):
        if False:
            print('Hello World!')
        gsp = (yield self.setupGerritStatusPushSimple())
        exp_argv = ['ssh', '-o', 'BatchMode=yes', 'user@serv', '-p', '29418', 'gerrit', 'version']

        def spawnProcess(pp, cmd, argv, env):
            if False:
                return 10
            self.assertEqual([cmd, argv], [exp_argv[0], exp_argv])
            pp.errReceived(b'test stderr\n')
            pp.outReceived(b'gerrit version 2.14\n')
            pp.outReceived(b'(garbage that should not cause a crash)\n')
            so = error.ProcessDone(None)
            pp.processEnded(failure.Failure(so))
        self.patch(reactor, 'spawnProcess', spawnProcess)
        gsp.callWithVersion(lambda : self.assertEqual(gsp.gerrit_version, parse_version('2.14')))

    def test_name_as_class_attribute(self):
        if False:
            i = 10
            return i + 15

        class FooStatusPush(GerritStatusPush):
            name = 'foo'
        reporter = FooStatusPush('gerrit.server.com', 'password')
        self.assertEqual(reporter.name, 'foo')

    def test_name_as_kwarg(self):
        if False:
            return 10
        reporter = GerritStatusPush('gerrit.server.com', 'password', name='foo')
        self.assertEqual(reporter.name, 'foo')

    def test_default_name(self):
        if False:
            while True:
                i = 10
        reporter = GerritStatusPush('gerrit.server.com', 'password')
        self.assertEqual(reporter.name, 'GerritStatusPush')