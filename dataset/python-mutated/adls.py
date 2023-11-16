from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.hooks.base import BaseHook
from airflow.providers.microsoft.azure.utils import get_field
if TYPE_CHECKING:
    from fsspec import AbstractFileSystem
schemes = ['abfs', 'abfss', 'adl']

def get_fs(conn_id: str | None) -> AbstractFileSystem:
    if False:
        i = 10
        return i + 15
    from adlfs import AzureBlobFileSystem
    if conn_id is None:
        return AzureBlobFileSystem()
    conn = BaseHook.get_connection(conn_id)
    extras = conn.extra_dejson
    connection_string = get_field(conn_id=conn_id, conn_type='azure_data_lake', extras=extras, field_name='connection_string')
    account_name = get_field(conn_id=conn_id, conn_type='azure_data_lake', extras=extras, field_name='account_name')
    account_key = get_field(conn_id=conn_id, conn_type='azure_data_lake', extras=extras, field_name='account_key')
    sas_token = get_field(conn_id=conn_id, conn_type='azure_data_lake', extras=extras, field_name='sas_token')
    tenant = get_field(conn_id=conn_id, conn_type='azure_data_lake', extras=extras, field_name='tenant')
    return AzureBlobFileSystem(connection_string=connection_string, account_name=account_name, account_key=account_key, sas_token=sas_token, tenant_id=tenant, client_id=conn.login, client_secret=conn.password)