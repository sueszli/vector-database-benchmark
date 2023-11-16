import requests
from api_app.analyzers_manager import classes
from tests.mock_utils import MockUpResponse, if_mock_connections, patch

class BitcoinAbuseAPI(classes.ObservableAnalyzer):
    url: str = 'https://www.bitcoinabuse.com/api/reports/check'
    _api_key_name: str

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'address': self.observable_name, 'api_token': self._api_key_name}
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