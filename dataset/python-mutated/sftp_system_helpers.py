from __future__ import annotations
import json
import os
from contextlib import contextmanager
from airflow.exceptions import AirflowException
from airflow.models import Connection
from airflow.utils.process_utils import patch_environ
SFTP_CONNECTION_ID = os.environ.get('SFTP_CONNECTION_ID', 'sftp_default')

@contextmanager
def provide_sftp_default_connection(key_file_path: str):
    if False:
        i = 10
        return i + 15
    '\n    Context manager to provide a temporary value for sftp_default connection\n\n    :param key_file_path: Path to file with sftp_default credentials .json file.\n    '
    if not key_file_path.endswith('.json'):
        raise AirflowException('Use a JSON key file.')
    with open(key_file_path) as credentials:
        creds = json.load(credentials)
    conn = Connection(conn_id=SFTP_CONNECTION_ID, conn_type='ssh', port=creds.get('port', None), host=creds.get('host', None), login=creds.get('login', None), password=creds.get('password', None), extra=json.dumps(creds.get('extra', None)))
    with patch_environ({f'AIRFLOW_CONN_{conn.conn_id.upper()}': conn.get_uri()}):
        yield