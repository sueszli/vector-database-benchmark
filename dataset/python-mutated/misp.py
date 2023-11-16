from typing import List
import pymisp
from django.conf import settings
from api_app import helpers
from api_app.analyzers_manager.constants import ObservableTypes
from api_app.connectors_manager.classes import Connector
from tests.mock_utils import if_mock_connections, patch
INTELOWL_MISP_TYPE_MAP = {ObservableTypes.IP: 'ip-src', ObservableTypes.DOMAIN: 'domain', ObservableTypes.URL: 'url', ObservableTypes.GENERIC: 'text', 'file': 'filename|md5'}

def create_misp_attribute(misp_type, misp_value) -> pymisp.MISPAttribute:
    if False:
        print('Hello World!')
    obj = pymisp.MISPAttribute()
    obj.type = misp_type
    obj.value = misp_value
    return obj

class MISP(Connector):
    tlp: str
    ssl_check: bool
    self_signed_certificate: str
    debug: bool
    _api_key_name: str
    _url_key_name: str

    @property
    def _event_obj(self) -> pymisp.MISPEvent:
        if False:
            print('Hello World!')
        obj = pymisp.MISPEvent()
        obj.info = f'Intelowl Job-{self.job_id}'
        obj.distribution = 0
        obj.threat_level_id = 4
        obj.analysis = 2
        obj.add_tag('source:intelowl')
        obj.add_tag(f'tlp:{self.tlp}')
        for tag in self._job.tags.all():
            obj.add_tag(f'intelowl-tag:{tag.label}')
        return obj

    @property
    def _base_attr_obj(self) -> pymisp.MISPAttribute:
        if False:
            return 10
        if self._job.is_sample:
            _type = INTELOWL_MISP_TYPE_MAP['file']
            value = f'{self._job.file_name}|{self._job.md5}'
        else:
            _type = self._job.observable_classification
            value = self._job.observable_name
            if _type == ObservableTypes.HASH:
                matched_type = helpers.get_hash_type(value)
                matched_type.replace('-', '')
                _type = matched_type if matched_type is not None else 'text'
            else:
                _type = INTELOWL_MISP_TYPE_MAP[_type]
        obj = create_misp_attribute(_type, value)
        analyzers_names = self._job.analyzers_to_execute.all().values_list('name', flat=True)
        obj.comment = f"Analyzers Executed: {', '.join(analyzers_names)}"
        return obj

    @property
    def _secondary_attr_objs(self) -> List[pymisp.MISPAttribute]:
        if False:
            print('Hello World!')
        obj_list = []
        if self._job.is_sample:
            obj_list.append(create_misp_attribute('mime-type', self._job.file_mimetype))
        return obj_list

    @property
    def _link_attr_obj(self) -> pymisp.MISPAttribute:
        if False:
            print('Hello World!')
        '\n        Returns attribute linking analysis on IntelOwl instance\n        '
        obj = pymisp.MISPAttribute()
        obj.type = 'link'
        obj.value = f'{settings.WEB_CLIENT_URL}/pages/scan/result/{self.job_id}'
        obj.comment = 'View Analysis on IntelOwl'
        obj.disable_correlation = True
        return obj

    def run(self):
        if False:
            print('Hello World!')
        ssl_param = f'{settings.PROJECT_LOCATION}/configuration/misp_ssl.crt' if self.ssl_check and self.self_signed_certificate else self.ssl_check
        misp_instance = pymisp.PyMISP(url=self._url_key_name, key=self._api_key_name, ssl=ssl_param, debug=self.debug, timeout=5)
        event = self._event_obj
        attributes = [self._base_attr_obj, *self._secondary_attr_objs, self._link_attr_obj]
        event.info += f': {self._base_attr_obj.value}'
        misp_event = misp_instance.add_event(event, pythonify=True)
        for attr in attributes:
            misp_instance.add_attribute(misp_event.id, attr)
        return misp_instance.get_event(misp_event.id)

    @classmethod
    def _monkeypatch(cls):
        if False:
            i = 10
            return i + 15
        patches = [if_mock_connections(patch('pymisp.PyMISP', side_effect=MockPyMISP))]
        return super()._monkeypatch(patches=patches)

class MockUpMISPElement:
    """
    Mock element(event/attribute) for testing
    """
    id: int = 1

class MockPyMISP:
    """
    Mock PyMISP instance for testing
     methods which require connection to a MISP instance
    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    def add_event(*args, **kwargs) -> MockUpMISPElement:
        if False:
            i = 10
            return i + 15
        return MockUpMISPElement()

    @staticmethod
    def add_attribute(*args, **kwargs) -> MockUpMISPElement:
        if False:
            i = 10
            return i + 15
        return MockUpMISPElement()

    @staticmethod
    def get_event(event_id) -> dict:
        if False:
            return 10
        return {'Event': {'id': event_id}}