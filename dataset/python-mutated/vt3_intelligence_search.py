import requests
from api_app.analyzers_manager.classes import ObservableAnalyzer
from tests.mock_utils import MockUpResponse, if_mock_connections, patch
from ...exceptions import AnalyzerRunException
from .vt3_base import VirusTotalv3AnalyzerMixin

class VirusTotalv3Intelligence(ObservableAnalyzer, VirusTotalv3AnalyzerMixin):
    base_url = 'https://www.virustotal.com/api/v3/intelligence'
    limit: int
    order_by: str

    def config(self):
        if False:
            return 10
        super().config()
        if self.limit > 300:
            self.limit = 300

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'query': self.observable_name, 'limit': self.limit}
        if self.order_by:
            params['order'] = self.order_by
        try:
            response = requests.get(self.base_url + '/search', params=params, headers=self.headers)
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