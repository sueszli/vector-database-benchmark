import requests
from api_app.analyzers_manager import classes
from api_app.analyzers_manager.exceptions import AnalyzerRunException
from tests.mock_utils import MockUpResponse, if_mock_connections, patch

class EmailRep(classes.ObservableAnalyzer):
    base_url: str = 'https://emailrep.io/{}'
    _api_key_name: str

    def run(self):
        if False:
            print('Hello World!')
        "\n        API key is not mandatory, emailrep supports requests with no key:\n        a valid key let you to do more requests per day.\n        therefore we're not checking if a key has been configured.\n        "
        headers = {'User-Agent': 'IntelOwl', 'Key': self._api_key_name, 'Accept': 'application/json'}
        if self.observable_classification not in [self.ObservableTypes.GENERIC]:
            raise AnalyzerRunException(f'not supported observable type {self.observable_classification}. Supported: generic')
        url = self.base_url.format(self.observable_name)
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
        except requests.RequestException as e:
            raise AnalyzerRunException(e)
        return response.json()

    @classmethod
    def _monkeypatch(cls):
        if False:
            i = 10
            return i + 15
        patches = [if_mock_connections(patch('requests.get', return_value=MockUpResponse({}, 200)))]
        return super()._monkeypatch(patches=patches)