from __future__ import annotations
from typing import TYPE_CHECKING
from impala.dbapi import connect
from airflow.providers.common.sql.hooks.sql import DbApiHook
if TYPE_CHECKING:
    from impala.interface import Connection

class ImpalaHook(DbApiHook):
    """Interact with Apache Impala through impyla."""
    conn_name_attr = 'impala_conn_id'
    default_conn_name = 'impala_default'
    conn_type = 'impala'
    hook_name = 'Impala'

    def get_conn(self) -> Connection:
        if False:
            print('Hello World!')
        connection = self.get_connection(self.impala_conn_id)
        return connect(host=connection.host, port=connection.port, user=connection.login, password=connection.password, database=connection.schema, **connection.extra_dejson)