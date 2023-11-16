"""Google DNS resolutions"""
import logging
from urllib.parse import urlparse
import requests
from api_app.analyzers_manager import classes
from api_app.analyzers_manager.exceptions import AnalyzerRunException
from tests.mock_utils import MockUpResponse, if_mock_connections, patch
from ..dns_responses import dns_resolver_response
logger = logging.getLogger(__name__)

class GoogleDNSResolver(classes.ObservableAnalyzer):
    """Resolve a DNS query with Google"""
    query_type: str

    def run(self):
        if False:
            return 10
        try:
            observable = self.observable_name
            if self.observable_classification == self.ObservableTypes.URL:
                observable = urlparse(self.observable_name).hostname
            params = {'name': observable, 'type': self.query_type}
            response = requests.get('https://dns.google.com/resolve', params=params)
            response.raise_for_status()
            data = response.json()
            resolutions = data.get('Answer', None)
        except requests.exceptions.RequestException:
            raise AnalyzerRunException('an error occurred during the connection to Google')
        return dns_resolver_response(self.observable_name, resolutions)

    @classmethod
    def _monkeypatch(cls):
        if False:
            while True:
                i = 10
        patches = [if_mock_connections(patch('requests.get', return_value=MockUpResponse({'Answer': ['test1', 'test2']}, 200)))]
        return super()._monkeypatch(patches=patches)