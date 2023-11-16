from parameterized import parameterized
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.reporters.generators.worker import WorkerMissingGenerator
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util.config import ConfigErrorsMixin

class TestWorkerMissingGenerator(ConfigErrorsMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantData=True, wantDb=True, wantMq=True)

    def _get_worker_dict(self, worker_name):
        if False:
            return 10
        return {'name': worker_name, 'notify': ['workeradmin@example.org'], 'workerinfo': {'admin': 'myadmin'}, 'last_connection': 'yesterday'}

    @parameterized.expand([(['myworker'],), ('all',)])
    @defer.inlineCallbacks
    def test_report_matched_worker(self, worker_filter):
        if False:
            return 10
        g = WorkerMissingGenerator(workers=worker_filter)
        report = (yield g.generate(self.master, None, 'worker.98.complete', self._get_worker_dict('myworker')))
        self.assertEqual(report['users'], ['workeradmin@example.org'])
        self.assertIn(b'worker named myworker went away', report['body'])

    @defer.inlineCallbacks
    def test_report_not_matched_worker(self):
        if False:
            print('Hello World!')
        g = WorkerMissingGenerator(workers=['other'])
        report = (yield g.generate(self.master, None, 'worker.98.complete', self._get_worker_dict('myworker')))
        self.assertIsNone(report)

    def test_unsupported_workers(self):
        if False:
            return 10
        g = WorkerMissingGenerator(workers='string worker')
        with self.assertRaisesConfigError("workers must be 'all', or list of worker names"):
            g.check()