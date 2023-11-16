from __future__ import annotations
import json
import os
from unittest import mock
from airflow.models import Connection
GCP_PROJECT_ID_HOOK_UNIT_TEST = 'example-project'

def mock_base_gcp_hook_default_project_id(self, gcp_conn_id='google_cloud_default', impersonation_chain=None, delegate_to=None):
    if False:
        i = 10
        return i + 15
    self.extras_list = {'project': GCP_PROJECT_ID_HOOK_UNIT_TEST}
    self._conn = gcp_conn_id
    self.impersonation_chain = impersonation_chain
    self._client = None
    self._conn = None
    self._cached_credentials = None
    self._cached_project_id = None
    self.delegate_to = delegate_to

def mock_base_gcp_hook_no_default_project_id(self, gcp_conn_id='google_cloud_default', impersonation_chain=None, delegate_to=None):
    if False:
        i = 10
        return i + 15
    self.extras_list = {}
    self._conn = gcp_conn_id
    self.impersonation_chain = impersonation_chain
    self._client = None
    self._conn = None
    self._cached_credentials = None
    self._cached_project_id = None
    self.delegate_to = delegate_to
if os.environ.get('_AIRFLOW_SKIP_DB_TESTS') == 'true':
    GCP_CONNECTION_WITH_PROJECT_ID = None
    GCP_CONNECTION_WITHOUT_PROJECT_ID = None
else:
    GCP_CONNECTION_WITH_PROJECT_ID = Connection(extra=json.dumps({'project': GCP_PROJECT_ID_HOOK_UNIT_TEST}))
    GCP_CONNECTION_WITHOUT_PROJECT_ID = Connection(extra=json.dumps({}))

def get_open_mock():
    if False:
        for i in range(10):
            print('nop')
    mck = mock.mock_open()
    open_module = 'builtins'
    return (mck, open_module)