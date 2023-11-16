from __future__ import annotations
import json
from airflow.models import Connection
ANALYTICDB_SPARK_PROJECT_ID_HOOK_UNIT_TEST = 'example-project'

def mock_adb_spark_hook_default_project_id(self, adb_spark_conn_id='mock_adb_spark_default', region='mock_region'):
    if False:
        while True:
            i = 10
    self.adb_spark_conn_id = adb_spark_conn_id
    self.adb_spark_conn = Connection(extra=json.dumps({'auth_type': 'AK', 'access_key_id': 'mock_access_key_id', 'access_key_secret': 'mock_access_key_secret', 'region': 'mock_region'}))
    self.region = region