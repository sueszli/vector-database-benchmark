import requests
from api_app.analyzers_manager.classes import ObservableAnalyzer
from api_app.analyzers_manager.exceptions import AnalyzerRunException
from tests.mock_utils import MockUpResponse, if_mock_connections, patch

class VirusheeCheckHash(ObservableAnalyzer):
    base_url: str = 'https://api.virushee.com/file/hash/{input}'
    _api_key_name: str

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.__session = requests.Session()
        if hasattr(self, '_api_key_name'):
            self.__session.headers['X-API-Key'] = self._api_key_name
        url = self.base_url.format(input=self.observable_name)
        try:
            response = self.__session.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            raise AnalyzerRunException(e)
        result = response.json()
        return result

    @classmethod
    def _monkeypatch(cls):
        if False:
            i = 10
            return i + 15
        patches = [if_mock_connections(patch('requests.Session.get', return_value=MockUpResponse({'success': True}, 200)))]
        return super()._monkeypatch(patches=patches)