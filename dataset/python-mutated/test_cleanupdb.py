import os
import textwrap
import sqlalchemy as sa
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.scripts import cleanupdb
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.unit.db import test_logs
from buildbot.test.util import db
from buildbot.test.util import dirs
from buildbot.test.util import misc
try:
    import lz4
    [lz4]
    hasLz4 = True
except ImportError:
    hasLz4 = False

def mkconfig(**kwargs):
    if False:
        return 10
    config = {'quiet': False, 'basedir': os.path.abspath('basedir'), 'force': True}
    config.update(kwargs)
    return config

def patch_environ(case, key, value):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add an environment variable for the duration of a test.\n    '
    old_environ = os.environ.copy()

    def cleanup():
        if False:
            for i in range(10):
                print('nop')
        os.environ.clear()
        os.environ.update(old_environ)
    os.environ[key] = value
    case.addCleanup(cleanup)

class TestCleanupDb(misc.StdoutAssertionsMixin, dirs.DirsMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.setUpDirs('basedir')
        with open(os.path.join('basedir', 'buildbot.tac'), 'wt', encoding='utf-8') as f:
            f.write(textwrap.dedent("\n                from twisted.application import service\n                application = service.Application('buildmaster')\n            "))
        self.setUpStdoutAssertions()
        self.ensureNoSqliteMemory()

    def tearDown(self):
        if False:
            return 10
        self.tearDownDirs()

    def ensureNoSqliteMemory(self):
        if False:
            i = 10
            return i + 15
        envkey = 'BUILDBOT_TEST_DB_URL'
        if envkey not in os.environ or os.environ[envkey] == 'sqlite://':
            patch_environ(self, envkey, 'sqlite:///' + os.path.abspath(os.path.join('basedir', 'state.sqlite')))

    def createMasterCfg(self, extraconfig=''):
        if False:
            return 10
        db_url = db.resolve_test_index_in_db_url(os.environ['BUILDBOT_TEST_DB_URL'])
        with open(os.path.join('basedir', 'master.cfg'), 'wt', encoding='utf-8') as f:
            f.write(textwrap.dedent(f"\n                from buildbot.plugins import *\n                c = BuildmasterConfig = dict()\n                c['db_url'] = {repr(db_url)}\n                c['buildbotNetUsageData'] = None\n                c['multiMaster'] = True  # don't complain for no builders\n                {extraconfig}\n            "))

    @defer.inlineCallbacks
    def test_cleanup_not_basedir(self):
        if False:
            while True:
                i = 10
        res = (yield cleanupdb._cleanupDatabase(mkconfig(basedir='doesntexist')))
        self.assertEqual(res, 1)
        self.assertInStdout('invalid buildmaster directory')

    @defer.inlineCallbacks
    def test_cleanup_bad_config(self):
        if False:
            while True:
                i = 10
        res = (yield cleanupdb._cleanupDatabase(mkconfig(basedir='basedir')))
        self.assertEqual(res, 1)
        self.assertInStdout("master.cfg' does not exist")

    @defer.inlineCallbacks
    def test_cleanup_bad_config2(self):
        if False:
            print('Hello World!')
        self.createMasterCfg(extraconfig='++++ # syntaxerror')
        res = (yield cleanupdb._cleanupDatabase(mkconfig(basedir='basedir')))
        self.assertEqual(res, 1)
        self.assertInStdout('encountered a SyntaxError while parsing config file:')
        self.flushLoggedErrors()

    def assertDictAlmostEqual(self, d1, d2):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(d1), len(d2))
        for k in d2.keys():
            self.assertApproximates(d1[k], d2[k], 10)

class TestCleanupDbRealDb(db.RealDatabaseWithConnectorMixin, TestCleanupDb):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            print('Hello World!')
        yield super().setUp()
        table_names = ['logs', 'logchunks', 'steps', 'builds', 'projects', 'builders', 'masters', 'buildrequests', 'buildsets', 'workers']
        self.master = fakemaster.make_master(self, wantRealReactor=True)
        yield self.setUpRealDatabaseWithConnector(self.master, table_names=table_names)

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.tearDownRealDatabaseWithConnector()

    @defer.inlineCallbacks
    def test_cleanup(self):
        if False:
            i = 10
            return i + 15
        yield self.insert_test_data(test_logs.Tests.backgroundData)
        LOGDATA = 'xx\n' * 2000
        logid = (yield self.master.db.logs.addLog(102, 'x', 'x', 's'))
        yield self.master.db.logs.appendLog(logid, LOGDATA)
        lengths = {}
        for mode in self.master.db.logs.COMPRESSION_MODE:
            if mode == 'lz4' and (not hasLz4):
                lengths['lz4'] = 40
                continue
            self.createMasterCfg(f"c['logCompressionMethod'] = '{mode}'")
            res = (yield cleanupdb._cleanupDatabase(mkconfig(basedir='basedir')))
            self.assertEqual(res, 0)
            res = (yield self.master.db.logs.getLogLines(logid, 0, 2000))
            self.assertEqual(res, LOGDATA)

            def thd(conn):
                if False:
                    for i in range(10):
                        print('nop')
                tbl = self.master.db.model.logchunks
                q = sa.select([sa.func.sum(sa.func.length(tbl.c.content))])
                q = q.where(tbl.c.logid == logid)
                return conn.execute(q).fetchone()[0]
            lengths[mode] = (yield self.master.db.pool.do(thd))
        self.assertDictAlmostEqual(lengths, {'raw': 5999, 'bz2': 44, 'lz4': 40, 'gz': 31})