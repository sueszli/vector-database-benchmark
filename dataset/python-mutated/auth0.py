import requests
from api_app.analyzers_manager import classes
from tests.mock_utils import MockUpResponse, if_mock_connections, patch

class Auth0(classes.ObservableAnalyzer):
    name: str = 'Auth0'
    base_url: str = 'https://signals.api.auth0.com/v2.0/ip'
    _api_key_name: str

    def run(self):
        if False:
            return 10
        headers = {'X-Auth-Token': self._api_key_name}
        url = f'{self.base_url}/{self.observable_name}'
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        json_response = response.json()
        return json_response

    @classmethod
    def _monkeypatch(cls):
        if False:
            return 10
        patches = [if_mock_connections(patch('requests.get', return_value=MockUpResponse({}, 200)))]
        return super()._monkeypatch(patches=patches)