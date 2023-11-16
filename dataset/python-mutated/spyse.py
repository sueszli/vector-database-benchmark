import re
import requests
from api_app.analyzers_manager import classes
from api_app.analyzers_manager.exceptions import AnalyzerRunException
from intel_owl.consts import REGEX_CVE, REGEX_EMAIL
from tests.mock_utils import MockUpResponse, if_mock_connections, patch

class Spyse(classes.ObservableAnalyzer):
    base_url: str = 'https://api.spyse.com/v4/data/'
    _api_key_name: str

    def __build_spyse_api_uri(self) -> str:
        if False:
            print('Hello World!')
        if self.observable_classification == self.ObservableTypes.DOMAIN:
            endpoint = 'domain'
        elif self.observable_classification == self.ObservableTypes.IP:
            endpoint = 'ip'
        elif self.observable_classification == self.ObservableTypes.GENERIC:
            if re.match(REGEX_EMAIL, self.observable_name):
                endpoint = 'email'
            elif re.match(REGEX_CVE, self.observable_name):
                endpoint = 'cve'
            else:
                raise AnalyzerRunException(f'{self.analyzer_name} with `generic` supports email and CVE only.')
        else:
            raise AnalyzerRunException(f'{self.observable_classification} not supported.Supported are: IP, domain and generic.')
        return f'{self.base_url}/{endpoint}/{self.observable_name}'

    def run(self):
        if False:
            i = 10
            return i + 15
        headers = {'Accept': 'application/json', 'Authorization': f'Bearer {self._api_key_name}'}
        api_uri = self.__build_spyse_api_uri()
        try:
            response = requests.get(api_uri, headers=headers)
            response.raise_for_status()
        except requests.RequestException as e:
            raise AnalyzerRunException(e)
        result = response.json()
        return result

    @classmethod
    def _monkeypatch(cls):
        if False:
            return 10
        patches = [if_mock_connections(patch('requests.get', return_value=MockUpResponse({}, 200)))]
        return super()._monkeypatch(patches=patches)