import types
from twisted.internet import defer
from buildbot.db import model
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util import db

class FakeDBConnector:
    pass

class ConnectorComponentMixin(TestReactorMixin, db.RealDatabaseMixin):
    """
    Implements a mock DBConnector object, replete with a thread pool and a DB
    model.  This includes a RealDatabaseMixin, so subclasses should not
    instantiate that class directly.  The connector appears at C{self.db}, and
    the component should be attached to it as an attribute.

    @ivar db: fake database connector
    @ivar db.pool: DB thread pool
    @ivar db.model: DB model
    """

    @defer.inlineCallbacks
    def setUpConnectorComponent(self, table_names=None, basedir='basedir', dialect_name='sqlite'):
        if False:
            print('Hello World!')
        'Set up C{self.db}, using the given db_url and basedir.'
        self.setup_test_reactor()
        if table_names is None:
            table_names = []
        yield self.setUpRealDatabase(table_names=table_names, basedir=basedir)
        self.db = FakeDBConnector()
        self.db.pool = self.db_pool
        self.db.master = fakemaster.make_master(self)
        self.db.model = model.Model(self.db)
        self.db._engine = types.SimpleNamespace(dialect=types.SimpleNamespace(name=dialect_name))

    @defer.inlineCallbacks
    def tearDownConnectorComponent(self):
        if False:
            while True:
                i = 10
        yield self.tearDownRealDatabase()
        del self.db.pool
        del self.db.model
        del self.db

class FakeConnectorComponentMixin(TestReactorMixin):

    def setUpConnectorComponent(self):
        if False:
            while True:
                i = 10
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantDb=True)
        self.db = self.master.db
        self.db.checkForeignKeys = True
        self.insert_test_data = self.db.insert_test_data
        return defer.succeed(None)