import json
from unittest import mock
from twisted.internet import defer
from twisted.internet import protocol
from twisted.internet import reactor
from twisted.trial import unittest
from twisted.web import client
from buildbot.data import connector as dataconnector
from buildbot.db import connector as dbconnector
from buildbot.mq import connector as mqconnector
from buildbot.test import fakedb
from buildbot.test.fake import fakemaster
from buildbot.test.util import db
from buildbot.test.util import www
from buildbot.util import bytes2unicode
from buildbot.util import unicode2bytes
from buildbot.www import auth
from buildbot.www import authz
from buildbot.www import service as wwwservice
SOMETIME = 1348971992
OTHERTIME = 1008971992

class BodyReader(protocol.Protocol):

    def __init__(self, finishedDeferred):
        if False:
            print('Hello World!')
        self.body = []
        self.finishedDeferred = finishedDeferred

    def dataReceived(self, data):
        if False:
            print('Hello World!')
        self.body.append(data)

    def connectionLost(self, reason):
        if False:
            print('Hello World!')
        if reason.check(client.ResponseDone):
            self.finishedDeferred.callback(b''.join(self.body))
        else:
            self.finishedDeferred.errback(reason)

class Www(db.RealDatabaseMixin, www.RequiresWwwMixin, unittest.TestCase):
    master = None

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            while True:
                i = 10
        yield self.setUpRealDatabase(table_names=['masters', 'objects', 'object_state'], sqlite_memory=False)
        master = fakemaster.FakeMaster(reactor)
        master.config.db = {'db_url': self.db_url}
        master.db = dbconnector.DBConnector('basedir')
        yield master.db.setServiceParent(master)
        yield master.db.setup(check_version=False)
        master.config.mq = {'type': 'simple'}
        master.mq = mqconnector.MQConnector()
        yield master.mq.setServiceParent(master)
        yield master.mq.setup()
        master.data = dataconnector.DataConnector()
        yield master.data.setServiceParent(master)
        master.config.www = {'port': 'tcp:0:interface=127.0.0.1', 'debug': True, 'auth': auth.NoAuth(), 'authz': authz.Authz(), 'avatar_methods': [], 'logfileName': 'http.log'}
        master.www = wwwservice.WWWService()
        yield master.www.setServiceParent(master)
        yield master.www.startService()
        yield master.www.reconfigServiceWithBuildbotConfig(master.config)
        session = mock.Mock()
        session.uid = '0'
        master.www.site.sessionFactory = mock.Mock(return_value=session)
        self.url = f'http://127.0.0.1:{master.www.getPortnum()}/'
        self.url = unicode2bytes(self.url)
        master.config.buildbotURL = self.url
        yield master.www.reconfigServiceWithBuildbotConfig(master.config)
        self.master = master
        if hasattr(client, 'HTTPConnectionPool'):
            self.pool = client.HTTPConnectionPool(reactor)
            self.agent = client.Agent(reactor, pool=self.pool)
        else:
            self.pool = None
            self.agent = client.Agent(reactor)

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            i = 10
            return i + 15
        if self.pool:
            yield self.pool.closeCachedConnections()
        if self.master:
            yield self.master.www.stopService()
        yield self.tearDownRealDatabase()

    @defer.inlineCallbacks
    def apiGet(self, url, expect200=True):
        if False:
            print('Hello World!')
        pg = (yield self.agent.request(b'GET', url))
        d = defer.Deferred()
        bodyReader = BodyReader(d)
        pg.deliverBody(bodyReader)
        body = (yield d)
        if expect200 and pg.code != 200:
            self.fail(f"did not get 200 response for '{url}'")
        return json.loads(bytes2unicode(body))

    def link(self, suffix):
        if False:
            print('Hello World!')
        return self.url + b'api/v2/' + suffix

    @defer.inlineCallbacks
    def test_masters(self):
        if False:
            while True:
                i = 10
        yield self.insert_test_data([fakedb.Master(id=7, name='some:master', active=0, last_active=SOMETIME), fakedb.Master(id=8, name='other:master', active=1, last_active=OTHERTIME)])
        res = (yield self.apiGet(self.link(b'masters')))
        self.assertEqual(res, {'masters': [{'active': False, 'masterid': 7, 'name': 'some:master', 'last_active': SOMETIME}, {'active': True, 'masterid': 8, 'name': 'other:master', 'last_active': OTHERTIME}], 'meta': {'total': 2}})
        res = (yield self.apiGet(self.link(b'masters/7')))
        self.assertEqual(res, {'masters': [{'active': False, 'masterid': 7, 'name': 'some:master', 'last_active': SOMETIME}], 'meta': {}})