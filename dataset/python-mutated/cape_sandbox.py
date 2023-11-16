import logging
import time
from tempfile import NamedTemporaryFile
import requests
from api_app.analyzers_manager.classes import FileAnalyzer
from api_app.analyzers_manager.exceptions import AnalyzerRunException
from tests.mock_utils import MockUpResponse, if_mock_connections, patch
logger = logging.getLogger(__name__)

class CAPEsandbox(FileAnalyzer):

    class ContinuePolling(Exception):
        pass
    options: str
    package: str
    timeout: int
    priority: int
    machine: str
    platform: str
    memory: bool
    enforce_timeout: bool
    custom: str
    tags: str
    route: str
    max_tries: int
    poll_distance: int
    requests_timeout: int
    _api_key_name: str
    _url_key_name: str
    _certificate: str

    @staticmethod
    def _clean_certificate(cert):
        if False:
            for i in range(10):
                print('nop')
        return cert.replace('-----BEGIN CERTIFICATE-----', '-----BEGIN_CERTIFICATE-----').replace('-----END CERTIFICATE-----', '-----END_CERTIFICATE-----').replace(' ', '\n').replace('-----BEGIN_CERTIFICATE-----', '-----BEGIN CERTIFICATE-----').replace('-----END_CERTIFICATE-----', '-----END CERTIFICATE-----')

    def config(self):
        if False:
            i = 10
            return i + 15
        super().config()
        self.__cert_file = NamedTemporaryFile(mode='w')
        self.__cert_file.write(self._clean_certificate(self._certificate))
        self.__cert_file.flush()
        self.__session = requests.Session()
        self.__session.verify = self.__cert_file.name
        self.__session.headers = {'Authorization': f'Token {self._api_key_name}'}

    def run(self):
        if False:
            i = 10
            return i + 15
        api_url: str = self._url_key_name + '/apiv2/tasks/create/file/'
        to_respond = {}
        logger.info(f'Job: {self.job_id} -> Starting file upload.')
        cape_params_name = ['options', 'package', 'timeout', 'priority', 'machine', 'platform', 'memory', 'enforce_timeout', 'custom', 'tags', 'route']
        data = {name: getattr(self, name) for name in cape_params_name if getattr(self, name, None) is not None}
        try:
            response = self.__session.post(api_url, files={'file': (self.filename, self.read_file_bytes())}, data=data, timeout=self.requests_timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise AnalyzerRunException(e)
        response_json = response.json()
        logger.debug(f'response received: {response_json}')
        response_error = response_json.get('error', False)
        if not response_error:
            task_id = response_json.get('data').get('task_ids')[0]
            result = self.__poll_for_result(task_id=task_id)
            to_respond['result_url'] = self._url_key_name + f'/submit/status/{task_id}/'
            to_respond['response'] = result
            logger.info(f'Job: {self.job_id} -> File uploaded successfully without any errors.')
        else:
            response_errors = response_json.get('errors', [])
            if response_errors:
                values = list(response_errors[0].values())
                if values and values[0] == 'Not unique, as unique option set on submit or in conf/web.conf':
                    logger.info(f"Job: {self.job_id} -> File uploaded is already present in the database. Querying its information through it's md5 hash..")
                    status_id = self.__search_by_md5()
                    gui_report_url = self._url_key_name + '/submit/status/' + status_id
                    report_url = self._url_key_name + '/apiv2/tasks/get/report/' + status_id + '/litereport'
                    to_respond['result_url'] = gui_report_url
                    try:
                        final_request = self.__session.get(report_url, timeout=self.requests_timeout)
                    except requests.RequestException as e:
                        raise AnalyzerRunException(e)
                    to_respond['response'] = final_request.json()
        return to_respond

    def __search_by_md5(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        db_search_url = self._url_key_name + '/apiv2/tasks/search/md5/' + self.md5
        try:
            q = self.__session.get(db_search_url, timeout=self.requests_timeout)
            q.raise_for_status()
        except requests.RequestException as e:
            raise AnalyzerRunException(e)
        data_list = q.json().get('data')
        if not data_list:
            raise AnalyzerRunException("'data' key in response isn't populated in __search_by_md5 as expected")
        status_id_int = data_list[0].get('id')
        status_id = str(status_id_int)
        return status_id

    def __single_poll(self, url, polling=True):
        if False:
            print('Hello World!')
        try:
            response = self.__session.get(url, timeout=self.requests_timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            if polling:
                raise self.ContinuePolling(f'RequestException {e}')
            else:
                raise AnalyzerRunException(e)
        return response

    def __poll_for_result(self, task_id) -> dict:
        if False:
            for i in range(10):
                print('nop')
        timeout_attempts = [30] + [60] * (self.timeout // 60) + [self.timeout % 60] + [50] + [self.poll_distance] * self.max_tries + [60] * 3
        tot_time = sum(timeout_attempts)
        if tot_time > 600:
            logger.warning(f' Job: {self.job_id} -> Broken soft time limit!! The analysis in the worst case will last {tot_time} seconds')
        results = None
        status_api = self._url_key_name + '/apiv2/tasks/status/' + str(task_id)
        is_pending = True
        while is_pending:
            is_pending = False
            for (try_, curr_timeout) in enumerate(timeout_attempts):
                attempt = try_ + 1
                try:
                    logger.info(f' Job: {self.job_id} -> Starting poll number #{attempt}/{len(timeout_attempts)}')
                    request = self.__single_poll(status_api)
                    responded_json = request.json()
                    error = responded_json.get('error')
                    data = responded_json.get('data')
                    logger.info(f'Job: {self.job_id} -> Status of the CAPESandbox task: {data}')
                    if error:
                        raise AnalyzerRunException(error)
                    if data == 'pending':
                        is_pending = True
                        logger.info(f' Job: {self.job_id} -> Waiting for the pending status to end, sleeping for 15 seconds...')
                        time.sleep(15)
                        break
                    if data in ('running', 'processing'):
                        raise self.ContinuePolling(f'Task still {data}')
                    if data in ('reported', 'completed'):
                        report_url = self._url_key_name + '/apiv2/tasks/get/report/' + str(task_id) + '/litereport'
                        results = self.__single_poll(report_url, polling=False).json()
                        if 'error' in results and results['error'] and (results['error_value'] == 'Task is still being analyzed'):
                            raise self.ContinuePolling('Task still processing')
                        logger.info(f' Job: {self.job_id} ->Poll number #{attempt}/{len(timeout_attempts)} fetched the results of the analysis. stopping polling..')
                        break
                    else:
                        raise AnalyzerRunException(f'status {data} was unexpected. Check the code')
                except self.ContinuePolling as e:
                    logger.info(f'Job: {self.job_id} -> Continuing the poll at attempt number: #{attempt}/{len(timeout_attempts)}. {e}. Sleeping for {curr_timeout} seconds.')
                    if try_ != self.max_tries - 1:
                        time.sleep(curr_timeout)
        if not results:
            raise AnalyzerRunException(f'{self.job_id} poll ended without results')
        return results

    @classmethod
    def _monkeypatch(cls):
        if False:
            print('Hello World!')
        patches = [if_mock_connections(patch('requests.Session.get', return_value=MockUpResponse({'error': False, 'data': 'completed'}, 200)), patch('requests.Session.post', return_value=MockUpResponse({'error': False, 'data': {'task_ids': [1234]}, 'errors': [], 'url': ['http://fake_url.com/submit/status/1234/']}, 200)))]
        return super()._monkeypatch(patches=patches)