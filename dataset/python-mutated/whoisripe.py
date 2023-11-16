import requests
from api_app.analyzers_manager import classes
from tests.mock_utils import MockUpResponse, if_mock_connections, patch

class WhoIsRipeAPI(classes.ObservableAnalyzer):
    url: str = 'https://rest.db.ripe.net/search.json'

    def run(self):
        if False:
            return 10
        params = {'query-string': self.observable_name}
        response = requests.get(self.url, params=params)
        response.raise_for_status()
        return response.json()

    @classmethod
    def _monkeypatch(cls):
        if False:
            while True:
                i = 10
        patches = [if_mock_connections(patch('requests.get', return_value=MockUpResponse({}, 200)))]
        return super()._monkeypatch(patches=patches)