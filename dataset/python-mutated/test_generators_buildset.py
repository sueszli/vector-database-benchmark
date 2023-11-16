from unittest.mock import Mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.process.results import SUCCESS
from buildbot.reporters import utils
from buildbot.reporters.generators.buildset import BuildSetStatusGenerator
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util.config import ConfigErrorsMixin
from buildbot.test.util.reporter import ReporterTestMixin
from buildbot.test.util.warnings import assertProducesWarning
from buildbot.warnings import DeprecatedApiWarning

class TestBuildSetGenerator(ConfigErrorsMixin, TestReactorMixin, ReporterTestMixin, unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.setup_test_reactor()
        self.setup_reporter_test()
        self.master = fakemaster.make_master(self, wantData=True, wantDb=True, wantMq=True)

    @defer.inlineCallbacks
    def insert_build_finished_get_props(self, results, **kwargs):
        if False:
            while True:
                i = 10
        build = (yield self.insert_build_finished(results, **kwargs))
        yield utils.getDetailsForBuild(self.master, build, want_properties=True)
        return build

    @defer.inlineCallbacks
    def setup_generator(self, results=SUCCESS, message=None, db_args=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if message is None:
            message = {'body': 'body', 'type': 'text', 'subject': 'subject'}
        if db_args is None:
            db_args = {}
        build = (yield self.insert_build_finished_get_props(results, **db_args))
        buildset = (yield self.master.data.get(('buildsets', 98)))
        g = BuildSetStatusGenerator(**kwargs)
        g.formatter = Mock(spec=g.formatter)
        g.formatter.format_message_for_build.return_value = message
        g.formatter.want_logs = False
        g.formatter.want_logs_content = False
        g.formatter.want_steps = False
        return (g, build, buildset)

    @defer.inlineCallbacks
    def buildset_message(self, g, builds, results=SUCCESS):
        if False:
            print('Hello World!')
        reporter = Mock()
        reporter.getResponsibleUsersForBuild.return_value = []
        report = (yield g.buildset_message(g.formatter, self.master, reporter, builds, results))
        return report

    @defer.inlineCallbacks
    def generate(self, g, key, build):
        if False:
            for i in range(10):
                print('nop')
        reporter = Mock()
        reporter.getResponsibleUsersForBuild.return_value = []
        report = (yield g.generate(self.master, reporter, key, build))
        return report

    @defer.inlineCallbacks
    def test_buildset_message_nominal(self):
        if False:
            return 10
        (g, build, _) = (yield self.setup_generator(mode=('change',)))
        report = (yield self.buildset_message(g, [build]))
        g.formatter.format_message_for_build.assert_called_with(self.master, build, is_buildset=True, mode=('change',), users=[])
        self.assertEqual(report, {'body': 'body', 'subject': 'subject', 'type': 'text', 'results': SUCCESS, 'builds': [build], 'users': [], 'patches': [], 'logs': []})

    @defer.inlineCallbacks
    def test_buildset_message_no_result(self):
        if False:
            while True:
                i = 10
        (g, build, _) = (yield self.setup_generator(results=None, mode=('change',)))
        report = (yield self.buildset_message(g, [build], results=None))
        g.formatter.format_message_for_build.assert_called_with(self.master, build, is_buildset=True, mode=('change',), users=[])
        self.assertEqual(report, {'body': 'body', 'subject': 'subject', 'type': 'text', 'results': None, 'builds': [build], 'users': [], 'patches': [], 'logs': []})

    @defer.inlineCallbacks
    def test_buildset_subject_deprecated(self):
        if False:
            return 10
        with assertProducesWarning(DeprecatedApiWarning, 'subject parameter'):
            yield self.setup_generator(subject='subject')

    @defer.inlineCallbacks
    def test_buildset_message_no_result_formatter_no_subject(self):
        if False:
            while True:
                i = 10
        message = {'body': 'body', 'type': 'text', 'subject': None}
        (g, build, _) = (yield self.setup_generator(results=None, message=message, mode=('change',)))
        report = (yield self.buildset_message(g, [build], results=None))
        g.formatter.format_message_for_build.assert_called_with(self.master, build, is_buildset=True, mode=('change',), users=[])
        self.assertEqual(report, {'body': 'body', 'subject': 'Buildbot not finished in Buildbot on whole buildset', 'type': 'text', 'results': None, 'builds': [build], 'users': [], 'patches': [], 'logs': []})

    @defer.inlineCallbacks
    def test_generate_complete(self):
        if False:
            return 10
        (g, build, buildset) = (yield self.setup_generator())
        report = (yield self.generate(g, ('buildsets', 98, 'complete'), buildset))
        del build['buildrequest']
        del build['parentbuild']
        del build['parentbuilder']
        self.assertEqual(report, {'body': 'body', 'subject': 'subject', 'type': 'text', 'results': SUCCESS, 'builds': [build], 'users': [], 'patches': [], 'logs': []})

    @defer.inlineCallbacks
    def test_generate_complete_non_matching_builder(self):
        if False:
            i = 10
            return i + 15
        (g, _, buildset) = (yield self.setup_generator(builders=['non-matched']))
        report = (yield self.generate(g, ('buildsets', 98, 'complete'), buildset))
        self.assertIsNone(report)

    @defer.inlineCallbacks
    def test_generate_complete_non_matching_result(self):
        if False:
            return 10
        (g, _, buildset) = (yield self.setup_generator(mode=('failing',)))
        report = (yield self.generate(g, ('buildsets', 98, 'complete'), buildset))
        self.assertIsNone(report)