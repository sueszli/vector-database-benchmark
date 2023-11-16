from unittest.mock import MagicMock, Mock, sentinel
import pytest
from celery import states
from celery.app import backends
from celery.backends import couchdb as module
from celery.backends.couchdb import CouchBackend
from celery.exceptions import ImproperlyConfigured
try:
    import pycouchdb
except ImportError:
    pycouchdb = None
COUCHDB_CONTAINER = 'celery_container'
pytest.importorskip('pycouchdb')

class test_CouchBackend:

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.Server = self.patching('pycouchdb.Server')
        self.backend = CouchBackend(app=self.app)

    def test_init_no_pycouchdb(self):
        if False:
            for i in range(10):
                print('nop')
        'test init no pycouchdb raises'
        (prev, module.pycouchdb) = (module.pycouchdb, None)
        try:
            with pytest.raises(ImproperlyConfigured):
                CouchBackend(app=self.app)
        finally:
            module.pycouchdb = prev

    def test_get_container_exists(self):
        if False:
            print('Hello World!')
        self.backend._connection = sentinel._connection
        connection = self.backend.connection
        assert connection is sentinel._connection
        self.Server.assert_not_called()

    def test_get(self):
        if False:
            i = 10
            return i + 15
        'test_get\n\n        CouchBackend.get should return  and take two params\n        db conn to couchdb is mocked.\n        '
        x = CouchBackend(app=self.app)
        x._connection = Mock()
        get = x._connection.get = MagicMock()
        assert x.get('1f3fab') == get.return_value['value']
        x._connection.get.assert_called_once_with('1f3fab')

    def test_get_non_existent_key(self):
        if False:
            i = 10
            return i + 15
        x = CouchBackend(app=self.app)
        x._connection = Mock()
        get = x._connection.get = MagicMock()
        get.side_effect = pycouchdb.exceptions.NotFound
        assert x.get('1f3fab') is None
        x._connection.get.assert_called_once_with('1f3fab')

    @pytest.mark.parametrize('key', ['1f3fab', b'1f3fab'])
    def test_set(self, key):
        if False:
            i = 10
            return i + 15
        x = CouchBackend(app=self.app)
        x._connection = Mock()
        x._set_with_state(key, 'value', states.SUCCESS)
        x._connection.save.assert_called_once_with({'_id': '1f3fab', 'value': 'value'})

    @pytest.mark.parametrize('key', ['1f3fab', b'1f3fab'])
    def test_set_with_conflict(self, key):
        if False:
            return 10
        x = CouchBackend(app=self.app)
        x._connection = Mock()
        x._connection.save.side_effect = (pycouchdb.exceptions.Conflict, None)
        get = x._connection.get = MagicMock()
        x._set_with_state(key, 'value', states.SUCCESS)
        x._connection.get.assert_called_once_with('1f3fab')
        x._connection.get('1f3fab').__setitem__.assert_called_once_with('value', 'value')
        x._connection.save.assert_called_with(get('1f3fab'))
        assert x._connection.save.call_count == 2

    def test_delete(self):
        if False:
            return 10
        'test_delete\n\n        CouchBackend.delete should return and take two params\n        db conn to pycouchdb is mocked.\n        TODO Should test on key not exists\n\n        '
        x = CouchBackend(app=self.app)
        x._connection = Mock()
        mocked_delete = x._connection.delete = Mock()
        mocked_delete.return_value = None
        assert x.delete('1f3fab') is None
        x._connection.delete.assert_called_once_with('1f3fab')

    def test_backend_by_url(self, url='couchdb://myhost/mycoolcontainer'):
        if False:
            while True:
                i = 10
        from celery.backends.couchdb import CouchBackend
        (backend, url_) = backends.by_url(url, self.app.loader)
        assert backend is CouchBackend
        assert url_ == url

    def test_backend_params_by_url(self):
        if False:
            for i in range(10):
                print('nop')
        url = 'couchdb://johndoe:mysecret@myhost:123/mycoolcontainer'
        with self.Celery(backend=url) as app:
            x = app.backend
            assert x.container == 'mycoolcontainer'
            assert x.host == 'myhost'
            assert x.username == 'johndoe'
            assert x.password == 'mysecret'
            assert x.port == 123