"""Hook for Cloudant."""
from __future__ import annotations
from typing import Any
from cloudant import cloudant
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook

class CloudantHook(BaseHook):
    """
    Interact with Cloudant. This class is a thin wrapper around the cloudant python library.

    .. seealso:: the latest documentation `here <https://python-cloudant.readthedocs.io/en/latest/>`_.

    :param cloudant_conn_id: The connection id to authenticate and get a session object from cloudant.
    """
    conn_name_attr = 'cloudant_conn_id'
    default_conn_name = 'cloudant_default'
    conn_type = 'cloudant'
    hook_name = 'Cloudant'

    @staticmethod
    def get_ui_field_behaviour() -> dict[str, Any]:
        if False:
            while True:
                i = 10
        'Return custom field behaviour.'
        return {'hidden_fields': ['port', 'extra'], 'relabeling': {'host': 'Account', 'login': 'Username (or API Key)', 'schema': 'Database'}}

    def __init__(self, cloudant_conn_id: str=default_conn_name) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.cloudant_conn_id = cloudant_conn_id

    def get_conn(self) -> cloudant:
        if False:
            print('Hello World!')
        "\n        Open a connection to the cloudant service and close it automatically if used as context manager.\n\n        .. note::\n            In the connection form:\n            - 'host' equals the 'Account' (optional)\n            - 'login' equals the 'Username (or API Key)' (required)\n            - 'password' equals the 'Password' (required)\n\n        :return: an authorized cloudant session context manager object.\n        "
        conn = self.get_connection(self.cloudant_conn_id)
        self._validate_connection(conn)
        cloudant_session = cloudant(user=conn.login, passwd=conn.password, account=conn.host)
        return cloudant_session

    def _validate_connection(self, conn: cloudant) -> None:
        if False:
            i = 10
            return i + 15
        for conn_param in ['login', 'password']:
            if not getattr(conn, conn_param):
                raise AirflowException(f'missing connection parameter {conn_param}')