import requests
from api_app.analyzers_manager import classes
from tests.mock_utils import MockUpResponse, if_mock_connections, patch

class Hunter_Io(classes.ObservableAnalyzer):
    base_url: str = 'https://api.hunter.io/v2/domain-search?'
    _api_key_name: str

    def run(self):
        if False:
            while True:
                i = 10
        url = f'{self.base_url}domain={self.observable_name}&api_key={self._api_key_name}'
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    @classmethod
    def _monkeypatch(cls):
        if False:
            i = 10
            return i + 15
        patches = [if_mock_connections(patch('requests.get', return_value=MockUpResponse({}, 200)))]
        return super()._monkeypatch(patches=patches)