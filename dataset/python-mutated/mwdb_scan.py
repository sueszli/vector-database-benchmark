import logging
import time
import mwdblib
from requests import HTTPError
from api_app.analyzers_manager.classes import FileAnalyzer
from api_app.analyzers_manager.exceptions import AnalyzerRunException
from tests.mock_utils import MagicMock, if_mock_connections, patch
logger = logging.getLogger(__name__)

def mocked_mwdb_response(*args, **kwargs):
    if False:
        while True:
            i = 10
    attrs = {'data': {'id': 'id_test', 'children': [], 'parents': []}, 'attributes': {'karton': 'test_analysis'}}
    fileInfo = MagicMock()
    fileInfo.configure_mock(**attrs)
    QueryResponse = MagicMock()
    attrs = {'query_file.return_value': fileInfo}
    QueryResponse.configure_mock(**attrs)
    Response = MagicMock(return_value=QueryResponse)
    return Response.return_value

class MockUpUploadObject:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.data = {'id': 'id_test'}

    def flush(self):
        if False:
            return 10
        return

class MockUpQueryObject:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.attributes = {'karton': 'test'}
        self.data = {'children': [], 'parents': []}

class MockUpMWDB:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    def upload_file(*args, **kwargs):
        if False:
            print('Hello World!')
        return MockUpUploadObject()

    @staticmethod
    def query_file(*args, **kwargs):
        if False:
            print('Hello World!')
        return MockUpQueryObject()

class MWDB_Scan(FileAnalyzer):
    _api_key_name: str
    private: bool
    max_tries: int

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        super().config()
        self.public = not self.private
        self.poll_distance = 5
        self.upload_file = self._job.tlp == self._job.TLP.CLEAR.value

    def adjust_relations(self, base, key, recursive=True):
        if False:
            for i in range(10):
                print('nop')
        new_relation = []
        for relation in base[key]:
            if relation['type'] == 'file':
                new_relation.append(self.mwdb.query_file(relation['id']).data)
            elif relation['type'] == 'static_config':
                new_relation.append(self.mwdb.query_config(relation['id']).data)
        base[key] = new_relation
        if recursive:
            for new_base in base[key]:
                if base['type'] == 'file':
                    if key == 'parents':
                        self.adjust_relations(new_base, key='parents', recursive=True)
                        self.adjust_relations(new_base, key='children', recursive=False)
                    elif key == 'children':
                        self.adjust_relations(new_base, key='parents', recursive=True)
                        self.adjust_relations(new_base, key='children', recursive=False)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        result = {}
        binary = self.read_file_bytes()
        query = self._job.sha256
        self.mwdb = mwdblib.MWDB(api_key=self._api_key_name)
        if self.upload_file:
            logger.info(f'mwdb_scan uploading sample: {self.md5}')
            file_object = self.mwdb.upload_file(query, binary, private=self.private, public=self.public)
            file_object.flush()
            for _try in range(self.max_tries):
                logger.info(f'mwdb_scan sample: {self.md5} polling for result try #{_try + 1}')
                file_info = self.mwdb.query_file(file_object.data['id'])
                if 'karton' in file_info.attributes.keys():
                    break
                time.sleep(self.poll_distance)
            else:
                raise AnalyzerRunException('max retry attempts exceeded')
        else:
            try:
                file_info = self.mwdb.query_file(query)
            except HTTPError:
                result['not_found'] = True
                return result
            else:
                result['not_found'] = False
        self.adjust_relations(file_info.data, 'parents', True)
        self.adjust_relations(file_info.data, 'children', True)
        result.update(data=file_info.data, permalink=f'https://mwdb.cert.pl/file/{query}')
        return result

    @classmethod
    def _monkeypatch(cls):
        if False:
            print('Hello World!')
        patches = [if_mock_connections(patch('mwdblib.MWDB', return_value=MockUpMWDB()))]
        return super()._monkeypatch(patches=patches)