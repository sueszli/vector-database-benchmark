from urllib.parse import urlparse
import requests
from api_app.analyzers_manager import classes
from tests.mock_utils import MockUpResponse, if_mock_connections, patch

class Tranco(classes.ObservableAnalyzer):
    base_url: str = 'https://tranco-list.eu/api/ranks/domain/'

    def run(self):
        if False:
            i = 10
            return i + 15
        observable_to_analyze = self.observable_name
        if self.observable_classification == self.ObservableTypes.URL:
            observable_to_analyze = urlparse(self.observable_name).hostname
        url = self.base_url + observable_to_analyze
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    @classmethod
    def _monkeypatch(cls):
        if False:
            print('Hello World!')
        patches = [if_mock_connections(patch('requests.get', return_value=MockUpResponse({}, 200)))]
        return super()._monkeypatch(patches=patches)