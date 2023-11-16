import logging
import time
import requests
from django.utils.functional import cached_property
from api_app.analyzers_manager.classes import ObservableAnalyzer
from api_app.analyzers_manager.exceptions import AnalyzerConfigurationException, AnalyzerRunException
from tests.mock_utils import MockUpResponse, if_mock_connections, patch
logger = logging.getLogger(__name__)

class IntelX(ObservableAnalyzer):
    """
    Analyzer Name: `IntelX`

    Refer to: https://github.com/IntelligenceX/SDK
    Requires API Key
    """
    base_url: str = 'https://2.intelx.io'
    _api_key_name: str
    query_type: str
    rows_limit: int
    max_tries: int
    poll_distance: int
    timeout: int
    datefrom: str
    dateto: str

    def config(self):
        if False:
            while True:
                i = 10
        super().config()
        if self.query_type not in ['phonebook', 'intelligent']:
            raise AnalyzerConfigurationException(f'{self.query_type} not supported')
        self.url = self.base_url + f'/{self.query_type}/search'

    @cached_property
    def _session(self):
        if False:
            return 10
        session = requests.Session()
        session.headers.update({'x-key': self._api_key_name, 'User-Agent': 'IntelOwl'})
        return session

    def _poll_for_results(self, search_id):
        if False:
            return 10
        json_data = {}
        for chance in range(self.max_tries):
            logger.info(f'Result Polling. Try #{chance + 1}. Starting the query...<-- {self.__repr__()}')
            try:
                r = self._session.get(f'{self.url}/result?id={search_id}&limit={self.rows_limit}&offset=-1')
                r.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f'request failed: {e}')
            else:
                if r.status_code == 200:
                    json_data = r.json()
                    break
            time.sleep(self.poll_distance)
        if not json_data:
            raise AnalyzerRunException(f'reached max tries for IntelX analysis, observable {self.observable_name}')
        if self.query_type == 'phonebook':
            selectors = json_data['selectors']
            parsed_selectors = self.__pb_search_results(selectors)
            result = {'id': search_id, **parsed_selectors}
        else:
            result = json_data
        return result

    def run(self):
        if False:
            while True:
                i = 10
        params = {'term': self.observable_name, 'buckets': [], 'lookuplevel': 0, 'maxresults': self.rows_limit, 'timeout': self.timeout, 'sort': 4, 'media': 0, 'terminate': []}
        if self.query_type == 'phonebook':
            params['target'] = 0
        elif self.query_type == 'intelligent':
            params['datefrom'] = self.datefrom
            params['dateto'] = self.dateto
        logger.info(f'starting {self.query_type} request for observable {self.observable_name}')
        r = self._session.post(self.url, json=params)
        r.raise_for_status()
        search_id = r.json().get('id', None)
        if not search_id:
            raise AnalyzerRunException(f'Failed to request search. Status code: {r.status_code}.')
        result = self._poll_for_results(search_id)
        return result

    @staticmethod
    def __pb_search_results(selectors):
        if False:
            while True:
                i = 10
        '\n        https://github.com/zeropwn/intelx.py/blob/master/cli/intelx.py#L89\n        '
        result = {}
        for block in selectors:
            selectortypeh = block['selectortypeh']
            if selectortypeh not in result:
                result[selectortypeh] = []
            result[selectortypeh].append(block['selectorvalue'])
        return result

    @classmethod
    def _monkeypatch(cls):
        if False:
            while True:
                i = 10
        patches = [if_mock_connections(patch('requests.Session.post', return_value=MockUpResponse({'id': 1}, 200)), patch('requests.Session.get', return_value=MockUpResponse({'selectors': []}, 200)))]
        return super()._monkeypatch(patches=patches)