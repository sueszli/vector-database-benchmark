from __future__ import annotations
import json
from airflow.models import Connection
OSS_PROJECT_ID_HOOK_UNIT_TEST = 'example-project'

def mock_oss_hook_default_project_id(self, oss_conn_id='mock_oss_default', region='mock_region'):
    if False:
        for i in range(10):
            print('nop')
    self.oss_conn_id = oss_conn_id
    self.oss_conn = Connection(extra=json.dumps({'auth_type': 'AK', 'access_key_id': 'mock_access_key_id', 'access_key_secret': 'mock_access_key_secret', 'region': 'mock_region'}))
    self.region = region