from __future__ import annotations
import json
import os
from contextlib import contextmanager
from airflow.exceptions import AirflowException
from airflow.models import Connection
from airflow.utils.process_utils import patch_environ
CONFIG_REQUIRED_FIELDS = ['host', 'login', 'password', 'security_token']
SALESFORCE_CONNECTION_ID = os.environ.get('SALESFORCE_CONNECTION_ID', 'salesforce_default')
CONNECTION_TYPE = os.environ.get('CONNECTION_TYPE', 'http')

@contextmanager
def provide_salesforce_connection(key_file_path: str):
    if False:
        i = 10
        return i + 15
    '\n    Context manager that provides a temporary value of SALESFORCE_DEFAULT connection.\n\n    :param key_file_path: Path to file with SALESFORCE credentials .json file.\n    '
    if not key_file_path.endswith('.json'):
        raise AirflowException('Use a JSON key file.')
    with open(key_file_path) as credentials:
        creds = json.load(credentials)
    missing_keys = CONFIG_REQUIRED_FIELDS - creds.keys()
    if missing_keys:
        message = f'{missing_keys} fields are missing'
        raise AirflowException(message)
    conn = Connection(conn_id=SALESFORCE_CONNECTION_ID, conn_type=CONNECTION_TYPE, host=creds['host'], login=creds['login'], password=creds['password'], extra=json.dumps({'security_token': creds['security_token']}))
    with patch_environ({f'AIRFLOW_CONN_{conn.conn_id.upper()}': conn.get_uri()}):
        yield