from datetime import datetime
from pickle import dumps, loads
from unittest.mock import Mock
import pytest
from celery import states
from celery.exceptions import ImproperlyConfigured
from celery.utils.objects import Bunch
CASSANDRA_MODULES = ['cassandra', 'cassandra.auth', 'cassandra.cluster', 'cassandra.query']

class test_CassandraBackend:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.app.conf.update(cassandra_servers=['example.com'], cassandra_keyspace='celery', cassandra_table='task_results')

    @pytest.mark.patched_module(*CASSANDRA_MODULES)
    def test_init_no_cassandra(self, module):
        if False:
            print('Hello World!')
        from celery.backends import cassandra as mod
        (prev, mod.cassandra) = (mod.cassandra, None)
        try:
            with pytest.raises(ImproperlyConfigured):
                mod.CassandraBackend(app=self.app)
        finally:
            mod.cassandra = prev

    @pytest.mark.patched_module(*CASSANDRA_MODULES)
    def test_init_with_and_without_LOCAL_QUROM(self, module):
        if False:
            return 10
        from celery.backends import cassandra as mod
        mod.cassandra = Mock()
        cons = mod.cassandra.ConsistencyLevel = Bunch(LOCAL_QUORUM='foo')
        self.app.conf.cassandra_read_consistency = 'LOCAL_FOO'
        self.app.conf.cassandra_write_consistency = 'LOCAL_FOO'
        mod.CassandraBackend(app=self.app)
        cons.LOCAL_FOO = 'bar'
        mod.CassandraBackend(app=self.app)
        with pytest.raises(ImproperlyConfigured):
            self.app.conf.cassandra_servers = None
            self.app.conf.cassandra_secure_bundle_path = None
            mod.CassandraBackend(app=self.app, keyspace='b', column_family='c')
        with pytest.raises(ImproperlyConfigured):
            self.app.conf.cassandra_servers = ['localhost']
            self.app.conf.cassandra_secure_bundle_path = '/home/user/secure-connect-bundle.zip'
            mod.CassandraBackend(app=self.app, keyspace='b', column_family='c')

    def test_init_with_cloud(self):
        if False:
            i = 10
            return i + 15
        from celery.backends import cassandra as mod

        class DummyClusterWithBundle:

            def __init__(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                if args != ():
                    raise ValueError('I should be created with kwargs only')
                pass

            def connect(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                return Mock()
        mod.cassandra = Mock()
        mod.cassandra.cluster = Mock()
        mod.cassandra.cluster.Cluster = DummyClusterWithBundle
        self.app.conf.cassandra_secure_bundle_path = '/path/to/bundle.zip'
        self.app.conf.cassandra_servers = None
        x = mod.CassandraBackend(app=self.app)
        x._get_connection()
        assert isinstance(x._cluster, DummyClusterWithBundle)

    @pytest.mark.patched_module(*CASSANDRA_MODULES)
    @pytest.mark.usefixtures('depends_on_current_app')
    def test_reduce(self, module):
        if False:
            print('Hello World!')
        from celery.backends.cassandra import CassandraBackend
        assert loads(dumps(CassandraBackend(app=self.app)))

    @pytest.mark.patched_module(*CASSANDRA_MODULES)
    def test_get_task_meta_for(self, module):
        if False:
            while True:
                i = 10
        from celery.backends import cassandra as mod
        mod.cassandra = Mock()
        x = mod.CassandraBackend(app=self.app)
        session = x._session = Mock()
        execute = session.execute = Mock()
        result_set = Mock()
        result_set.one.return_value = [states.SUCCESS, '1', datetime.now(), b'', b'']
        execute.return_value = result_set
        x.decode = Mock()
        meta = x._get_task_meta_for('task_id')
        assert meta['status'] == states.SUCCESS
        result_set.one.return_value = []
        x._session.execute.return_value = result_set
        meta = x._get_task_meta_for('task_id')
        assert meta['status'] == states.PENDING

    def test_as_uri(self):
        if False:
            i = 10
            return i + 15
        from celery.backends import cassandra as mod
        mod.cassandra = Mock()
        x = mod.CassandraBackend(app=self.app)
        x.as_uri()
        x.as_uri(include_password=False)

    @pytest.mark.patched_module(*CASSANDRA_MODULES)
    def test_store_result(self, module):
        if False:
            for i in range(10):
                print('nop')
        from celery.backends import cassandra as mod
        mod.cassandra = Mock()
        x = mod.CassandraBackend(app=self.app)
        session = x._session = Mock()
        session.execute = Mock()
        x._store_result('task_id', 'result', states.SUCCESS)

    def test_timeouting_cluster(self):
        if False:
            print('Hello World!')
        from celery.backends import cassandra as mod

        class OTOExc(Exception):
            pass

        class VeryFaultyCluster:

            def __init__(self, *args, **kwargs):
                if False:
                    return 10
                pass

            def connect(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                raise OTOExc()

            def shutdown(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        mod.cassandra = Mock()
        mod.cassandra.OperationTimedOut = OTOExc
        mod.cassandra.cluster = Mock()
        mod.cassandra.cluster.Cluster = VeryFaultyCluster
        x = mod.CassandraBackend(app=self.app)
        with pytest.raises(OTOExc):
            x._store_result('task_id', 'result', states.SUCCESS)
        assert x._cluster is None
        assert x._session is None

    def test_create_result_table(self):
        if False:
            return 10
        from celery.backends import cassandra as mod

        class OTOExc(Exception):
            pass

        class FaultySession:

            def __init__(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def execute(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                raise OTOExc()

        class DummyCluster:

            def __init__(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                pass

            def connect(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                return FaultySession()
        mod.cassandra = Mock()
        mod.cassandra.cluster = Mock()
        mod.cassandra.cluster.Cluster = DummyCluster
        mod.cassandra.AlreadyExists = OTOExc
        x = mod.CassandraBackend(app=self.app)
        x._get_connection(write=True)
        assert x._session is not None

    def test_init_session(self):
        if False:
            return 10
        from celery.backends import cassandra as mod

        class DummyCluster:

            def __init__(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def connect(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                return Mock()
        mod.cassandra = Mock()
        mod.cassandra.cluster = Mock()
        mod.cassandra.cluster.Cluster = DummyCluster
        x = mod.CassandraBackend(app=self.app)
        assert x._session is None
        x._get_connection(write=True)
        assert x._session is not None
        s = x._session
        x._get_connection()
        assert s is x._session

    def test_auth_provider(self):
        if False:
            return 10
        from celery.backends import cassandra as mod

        class DummyAuth:
            ValidAuthProvider = Mock()
        mod.cassandra = Mock()
        mod.cassandra.auth = DummyAuth
        self.app.conf.cassandra_auth_provider = 'ValidAuthProvider'
        self.app.conf.cassandra_auth_kwargs = {'username': 'stuff'}
        mod.CassandraBackend(app=self.app)
        self.app.conf.cassandra_auth_provider = 'SpiderManAuth'
        self.app.conf.cassandra_auth_kwargs = {'username': 'Jack'}
        with pytest.raises(ImproperlyConfigured):
            mod.CassandraBackend(app=self.app)

    def test_options(self):
        if False:
            return 10
        from celery.backends import cassandra as mod
        mod.cassandra = Mock()
        self.app.conf.cassandra_options = {'cql_version': '3.2.1', 'protocol_version': 3}
        mod.CassandraBackend(app=self.app)