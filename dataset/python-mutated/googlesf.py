"""Check if the domains is reported as malicious for GoogleSafeBrowsing"""
from typing import Dict, List
import pysafebrowsing
from api_app.analyzers_manager import classes
from api_app.analyzers_manager.exceptions import AnalyzerRunException
from tests.mock_utils import if_mock_connections, patch
from ..dns_responses import malicious_detector_response

class MockUpSafeBrowsing:

    @staticmethod
    def lookup_urls(urls: List[str]) -> Dict:
        if False:
            print('Hello World!')
        return {url: {'malicious': True, 'cache': 'test', 'threats': 'test', 'platforms': 'test'} for url in urls}

class GoogleSF(classes.ObservableAnalyzer):
    """Check if observable analyzed is marked as malicious for Google SafeBrowsing"""
    _api_key_name: str

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        sb_instance = pysafebrowsing.SafeBrowsing(self._api_key_name)
        response = sb_instance.lookup_urls([self.observable_name])
        if self.observable_name in response and isinstance(response[self.observable_name], dict):
            result = response[self.observable_name]
        else:
            raise AnalyzerRunException(f'result not expected: {response}')
        malicious = result['malicious']
        googlesb_result = malicious_detector_response(self.observable_name, malicious)
        if malicious:
            googlesb_result['cache'] = result['cache']
            googlesb_result['threats'] = result['threats']
            googlesb_result['platforms'] = result['platforms']
        return googlesb_result

    @classmethod
    def _monkeypatch(cls):
        if False:
            i = 10
            return i + 15
        patches = [if_mock_connections(patch('pysafebrowsing.SafeBrowsing', return_value=MockUpSafeBrowsing()))]
        return super()._monkeypatch(patches=patches)