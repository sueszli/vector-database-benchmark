import pytest
from mock import Mock
from golem.apps.rpc import ClientAppProvider
from golem.apps.manager import AppManager
from golem.apps.default import BlenderAppDefinition
from golem.testutils import pytest_database_fixture

class TestClientAppProvider:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self._app_manger = Mock(spec=AppManager)
        self._handler = ClientAppProvider(self._app_manger)

    def test_list(self):
        if False:
            print('Hello World!')
        mocked_apps = [(BlenderAppDefinition.id, BlenderAppDefinition)]
        self._app_manger.apps = Mock(return_value=mocked_apps)
        result = self._handler.apps_list()
        assert len(result) == len(mocked_apps), 'count of result does not match input count'
        assert result[0]['id'] == mocked_apps[0][0], 'the first returned app id does not match input'
        assert self._app_manger.apps.called_once_with()

    def test_update(self):
        if False:
            i = 10
            return i + 15
        app_id = 'a'
        enabled = True
        result = self._handler.apps_update(app_id, enabled)
        self._app_manger.registered.called_once_with(app_id)
        self._app_manger.set_enabled.called_once_with(app_id, enabled)
        assert result == 'App state updated.'

    def test_update_not_registered(self):
        if False:
            i = 10
            return i + 15
        app_id = 'a'
        enabled = True
        self._app_manger.registered.return_value = False
        with pytest.raises(Exception):
            self._handler.apps_update(app_id, enabled)
        self._app_manger.registered.called_once_with(app_id)
        self._app_manger.set_enabled.assert_not_called()

    def test_delete(self):
        if False:
            i = 10
            return i + 15
        app_id = 'a'
        result = self._handler.apps_delete(app_id)
        self._app_manger.delete.called_once_with(app_id)
        assert result == 'App deleted with success.'

    def test_delete_failed(self):
        if False:
            print('Hello World!')
        app_id = 'a'
        self._app_manger.delete.return_value = False
        with pytest.raises(Exception):
            self._handler.apps_delete(app_id)
        self._app_manger.delete.called_once_with(app_id)