import requests
from api_app.analyzers_manager import classes
from api_app.analyzers_manager.exceptions import AnalyzerRunException
from tests.mock_utils import MockUpResponse, if_mock_connections, patch

class CRXcavator(classes.ObservableAnalyzer):
    name: str = 'CRXcavator'
    base_url: str = 'https://api.crxcavator.io/v1/report/'

    def run(self):
        if False:
            while True:
                i = 10
        try:
            response = requests.get(self.base_url + self.observable_name)
            response.raise_for_status()
        except requests.RequestException as e:
            raise AnalyzerRunException(e)
        result = response.json()
        return result

    @classmethod
    def _monkeypatch(cls):
        if False:
            print('Hello World!')
        patches = [if_mock_connections(patch('requests.get', return_value=MockUpResponse({}, 200)))]
        return super()._monkeypatch(patches=patches)