import requests
from api_app.analyzers_manager import classes
from api_app.analyzers_manager.exceptions import AnalyzerRunException
from tests.mock_utils import MockUpResponse, if_mock_connections, patch

class IPApi(classes.ObservableAnalyzer):
    batch_url = 'http://ip-api.com/batch'
    dns_url = 'http://edns.ip-api.com/json'
    fields: str
    lang: str

    def config(self):
        if False:
            return 10
        super().config()
        self.IP = [{'query': self.observable_name, 'fields': self.fields, 'lang': self.lang}]

    def run(self):
        if False:
            while True:
                i = 10
        try:
            response_batch = requests.post(self.batch_url, json=self.IP)
            response_batch.raise_for_status()
            response_dns = requests.get(self.dns_url)
            response_dns.raise_for_status()
        except requests.RequestException as e:
            raise AnalyzerRunException(e)
        response = {'ip_info': response_batch.json(), 'dns_info': response_dns.json()}
        return response

    @classmethod
    def _monkeypatch(cls):
        if False:
            while True:
                i = 10
        patches = [if_mock_connections(patch('requests.get', return_value=MockUpResponse({}, 200)))]
        return super()._monkeypatch(patches=patches)