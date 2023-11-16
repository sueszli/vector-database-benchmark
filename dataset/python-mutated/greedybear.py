import requests
from api_app.analyzers_manager.classes import ObservableAnalyzer
from tests.mock_utils import MockUpResponse, if_mock_connections, patch

class GreedyBear(ObservableAnalyzer):
    _api_key_name: str
    url: str

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        headers = {'Authorization': 'Token ' + self._api_key_name, 'Accept': 'application/json'}
        params_ = {'query': self.observable_name}
        uri = '/api/enrichment'
        response = requests.get(self.url + uri, params=params_, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result

    @classmethod
    def _monkeypatch(cls):
        if False:
            i = 10
            return i + 15
        patches = [if_mock_connections(patch('requests.get', return_value=MockUpResponse({}, 200)))]
        return super()._monkeypatch(patches=patches)