"""Check if the domains is reported as malicious in DNS0.eu database"""
import logging
from ipaddress import AddressValueError, IPv4Address
from urllib.parse import urlparse
import requests
from api_app.analyzers_manager import classes
from api_app.analyzers_manager.exceptions import AnalyzerRunException
from tests.mock_utils import MockUpResponse, if_mock_connections, patch
from ..dns_responses import malicious_detector_response
logger = logging.getLogger(__name__)

class DNS0EUMaliciousDetector(classes.ObservableAnalyzer):

    class NotADomain(Exception):
        pass

    def run(self):
        if False:
            return 10
        observable = self.observable_name
        is_malicious = False
        try:
            if self.observable_classification == self.ObservableTypes.URL:
                observable = urlparse(self.observable_name).hostname
                try:
                    IPv4Address(observable)
                except AddressValueError:
                    pass
                else:
                    raise self.NotADomain()
            params = {'name': observable, 'type': 'A'}
            headers = {'accept': 'application/dns-json'}
            response = requests.get('https://zero.dns0.eu', params=params, headers=headers)
            response.raise_for_status()
            response_dict = response.json()
            response_answer = response_dict.get('Authority', [])
            if response_answer:
                resolution = response_answer[0].get('data', '')
                if 'negative-caching.dns0.eu' in resolution:
                    is_malicious = True
        except requests.exceptions.RequestException:
            raise AnalyzerRunException('Connection to DNS0 failed')
        except self.NotADomain:
            logger.info(f'not analyzing {observable} because not a domain')
        return malicious_detector_response(self.observable_name, is_malicious)

    @classmethod
    def _monkeypatch(cls):
        if False:
            for i in range(10):
                print('nop')
        patches = [if_mock_connections(patch('requests.get', return_value=MockUpResponse({'Answer': [{'data': 'negative-caching.dns0.eu'}]}, 200)))]
        return super()._monkeypatch(patches=patches)