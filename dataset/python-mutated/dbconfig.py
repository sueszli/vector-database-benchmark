from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import ProgrammingError
from buildbot.config.master import MasterConfig
from buildbot.db import enginestrategy
from buildbot.db import model
from buildbot.db import state

class FakeDBConnector:
    pass

class FakeCacheManager:

    def get_cache(self, cache_name, miss_fn):
        if False:
            i = 10
            return i + 15
        return None

class FakeMaster:
    pass

class FakePool:
    pass

class DbConfig:

    def __init__(self, BuildmasterConfig, basedir, name='config'):
        if False:
            while True:
                i = 10
        self.db_url = MasterConfig.getDbUrlFromConfig(BuildmasterConfig, throwErrors=False)
        self.basedir = basedir
        self.name = name

    def getDb(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            db_engine = enginestrategy.create_engine(self.db_url, basedir=self.basedir)
        except Exception:
            return None
        db = FakeDBConnector()
        db.master = FakeMaster()
        db.pool = FakePool()
        db.pool.engine = db_engine
        db.master.caches = FakeCacheManager()
        db.model = model.Model(db)
        db.state = state.StateConnectorComponent(db)
        try:
            self.objectid = db.state.thdGetObjectId(db_engine, self.name, 'DbConfig')['id']
        except (ProgrammingError, OperationalError):
            db.pool.engine.dispose()
            return None
        return db

    def get(self, name, default=state.StateConnectorComponent.Thunk):
        if False:
            while True:
                i = 10
        db = self.getDb()
        if db is not None:
            ret = db.state.thdGetState(db.pool.engine, self.objectid, name, default=default)
            db.pool.engine.dispose()
        else:
            if default is not state.StateConnectorComponent.Thunk:
                return default
            raise KeyError('Db not yet initialized')
        return ret

    def set(self, name, value):
        if False:
            print('Hello World!')
        db = self.getDb()
        if db is not None:
            db.state.thdSetState(db.pool.engine, self.objectid, name, value)
            db.pool.engine.dispose()