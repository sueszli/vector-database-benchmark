import json
import logging
import time
import requests
from api_app.analyzers_manager.classes import FileAnalyzer
from api_app.analyzers_manager.exceptions import AnalyzerConfigurationException, AnalyzerRunException
from api_app.analyzers_manager.observable_analyzers.yaraify import YARAify
from tests.mock_utils import MockUpResponse, if_mock_connections, patch
logger = logging.getLogger(__name__)

class YARAifyFileScan(FileAnalyzer, YARAify):
    _api_key_identifier: str
    clamav_scan: bool
    unpack: bool
    share_file: bool
    skip_noisy: bool
    skip_known: bool

    def config(self):
        if False:
            print('Hello World!')
        self.query = 'lookup_hash'
        YARAify.config(self)
        self.search_term = self.md5
        self.max_tries = 200
        self.poll_distance = 3
        self.send_file = self._job.tlp == self._job.TLP.CLEAR.value
        if self.send_file and (not hasattr(self, '_api_key_identifier')):
            raise AnalyzerConfigurationException('Unable to send file without having api_key_identifier set')

    def run(self):
        if False:
            return 10
        name_to_send = self.filename if self.filename else self.md5
        file = self.read_file_bytes()
        logger.info(f'checking hash: {self.md5}')
        hash_scan = YARAify.run(self)
        query_status = hash_scan.get('query_status')
        logger.info(f'query_status={query_status!r} for hash {self.md5}')
        if query_status == 'ok':
            logger.info(f'found YARAify hash scan and returning it. {self.md5}')
            return hash_scan
        result = hash_scan
        if self.send_file:
            data = {'identifier': self._api_key_identifier, 'clamav_scan': int(self.clamav_scan), 'unpack': int(self.unpack), 'share_file': int(self.share_file), 'skip_noisy': int(self.skip_noisy), 'skip_known': int(self.skip_known)}
            files_ = {'json_data': (None, json.dumps(data), 'application/json'), 'file': (name_to_send, file)}
            logger.info(f'yara file scan md5 {self.md5} sending sample for analysis')
            response = requests.post(self.url, files=files_)
            response.raise_for_status()
            scan_response = response.json()
            scan_query_status = scan_response.get('query_status')
            if scan_query_status == 'queued':
                task_id = scan_response.get('data', {}).get('task_id', '')
                if not task_id:
                    raise AnalyzerRunException(f'task_id value is unexpected: {task_id}.Analysis was requested for md5 {self.md5}')
                for _try in range(self.max_tries):
                    try:
                        logger.info(f'yara file scan md5 {self.md5} polling for result try #{_try + 1}.task_id: {task_id}')
                        data = {'query': 'get_results', 'task_id': task_id}
                        response = requests.post(self.url, json=data)
                        response.raise_for_status()
                        task_response = response.json()
                        logger.debug(task_response)
                        data_results = task_response.get('data')
                        if isinstance(data_results, dict) and data_results:
                            logger.info(f'found scan result for {self.md5}')
                            break
                    except requests.RequestException as e:
                        logger.warning(e, stack_info=True)
                    time.sleep(self.poll_distance)
            else:
                raise AnalyzerRunException(f'query_status value is unexpected: {scan_query_status}.Analysis was requested for md5 {self.md5}')
            result = response.json()
        return result

    @classmethod
    def _monkeypatch(cls):
        if False:
            print('Hello World!')
        patches = [if_mock_connections(patch('requests.post', side_effect=[MockUpResponse({'query_status': 'not-available'}, 200), MockUpResponse({'query_status': 'queued', 'data': {'task_id': 123}}, 200), MockUpResponse({'query_status': 'ok', 'data': {'static_results': []}}, 200)]))]
        return FileAnalyzer._monkeypatch(patches=patches)