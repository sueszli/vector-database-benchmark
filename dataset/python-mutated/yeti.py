import requests
from django.conf import settings
from api_app.connectors_manager import classes
from api_app.connectors_manager.exceptions import ConnectorRunException
from tests.mock_utils import MockUpResponse, if_mock_connections, patch

class YETI(classes.Connector):
    verify_ssl: bool
    _url_key_name: str
    _api_key_name: str

    def run(self):
        if False:
            while True:
                i = 10
        if self._job.is_sample:
            obs_value = self._job.md5
            obs_type = 'file'
        else:
            obs_value = self._job.observable_name
            obs_type = self._job.observable_classification
        context = {'source': 'IntelOwl', 'report': f'{settings.WEB_CLIENT_URL}/pages/scan/result/{self.job_id}', 'status': 'analyzed', 'date': str(self._job.finished_analysis_time), 'description': f"IntelOwl's analysis report for Job: {self.job_id} | {obs_value} | {obs_type}", 'analyzers executed': ', '.join(list(self._job.analyzers_to_execute.all().values_list('name', flat=True)))}
        tags = list(self._job.tags.all().values_list('label', flat=True))
        payload = {'value': obs_value, 'source': 'IntelOwl', 'tags': tags, 'context': context}
        headers = {'Accept': 'application/json', 'X-Api-Key': self._api_key_name}
        if self._url_key_name and self._url_key_name.endswith('/'):
            self._url_key_name = self._url_key_name[:-1]
        url = f'{self._url_key_name}/observable/'
        try:
            resp = requests.post(url=url, headers=headers, json=payload, verify=self.verify_ssl)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise ConnectorRunException(e)
        return resp.json()

    @classmethod
    def _monkeypatch(cls):
        if False:
            print('Hello World!')
        patches = [if_mock_connections(patch('requests.post', return_value=MockUpResponse({}, 200)))]
        return super()._monkeypatch(patches=patches)