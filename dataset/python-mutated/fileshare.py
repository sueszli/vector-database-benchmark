from __future__ import annotations
from typing import IO, Any
from azure.storage.fileshare import FileProperties, ShareDirectoryClient, ShareFileClient, ShareServiceClient
from airflow.hooks.base import BaseHook
from airflow.providers.microsoft.azure.utils import add_managed_identity_connection_widgets, get_sync_default_azure_credential

class AzureFileShareHook(BaseHook):
    """
    Interacts with Azure FileShare Storage.

    :param azure_fileshare_conn_id: Reference to the
        :ref:`Azure FileShare connection id<howto/connection:azure_fileshare>`
        of an Azure account of which file share should be used.
    """
    conn_name_attr = 'azure_fileshare_conn_id'
    default_conn_name = 'azure_fileshare_default'
    conn_type = 'azure_fileshare'
    hook_name = 'Azure FileShare'

    @staticmethod
    @add_managed_identity_connection_widgets
    def get_connection_form_widgets() -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Returns connection widgets to add to connection form.'
        from flask_appbuilder.fieldwidgets import BS3PasswordFieldWidget, BS3TextFieldWidget
        from flask_babel import lazy_gettext
        from wtforms import PasswordField, StringField
        return {'sas_token': PasswordField(lazy_gettext('SAS Token (optional)'), widget=BS3PasswordFieldWidget()), 'connection_string': StringField(lazy_gettext('Connection String (optional)'), widget=BS3TextFieldWidget())}

    @staticmethod
    def get_ui_field_behaviour() -> dict[str, Any]:
        if False:
            return 10
        'Returns custom field behaviour.'
        return {'hidden_fields': ['schema', 'port', 'host', 'extra'], 'relabeling': {'login': 'Blob Storage Login (optional)', 'password': 'Blob Storage Key (optional)'}, 'placeholders': {'login': 'account name or account url', 'password': 'secret', 'sas_token': 'account url or token (optional)', 'connection_string': 'account url or token (optional)'}}

    def __init__(self, share_name: str | None=None, file_path: str | None=None, directory_path: str | None=None, azure_fileshare_conn_id: str='azure_fileshare_default') -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._conn_id = azure_fileshare_conn_id
        self.share_name = share_name
        self.file_path = file_path
        self.directory_path = directory_path
        self._account_url: str | None = None
        self._connection_string: str | None = None
        self._account_access_key: str | None = None
        self._sas_token: str | None = None

    def get_conn(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        conn = self.get_connection(self._conn_id)
        extras = conn.extra_dejson
        self._connection_string = extras.get('connection_string')
        if conn.login:
            self._account_url = self._parse_account_url(conn.login)
        self._sas_token = extras.get('sas_token')
        self._account_access_key = conn.password

    @staticmethod
    def _parse_account_url(account_url: str) -> str:
        if False:
            i = 10
            return i + 15
        if not account_url.lower().startswith('https'):
            return f'https://{account_url}.file.core.windows.net'
        return account_url

    def _get_sync_default_azure_credential(self):
        if False:
            for i in range(10):
                print('nop')
        conn = self.get_connection(self._conn_id)
        extras = conn.extra_dejson
        managed_identity_client_id = extras.get('managed_identity_client_id')
        workload_identity_tenant_id = extras.get('workload_identity_tenant_id')
        return get_sync_default_azure_credential(managed_identity_client_id=managed_identity_client_id, workload_identity_tenant_id=workload_identity_tenant_id)

    @property
    def share_service_client(self):
        if False:
            i = 10
            return i + 15
        self.get_conn()
        if self._connection_string:
            return ShareServiceClient.from_connection_string(conn_str=self._connection_string)
        elif self._account_url and (self._sas_token or self._account_access_key):
            credential = self._sas_token or self._account_access_key
            return ShareServiceClient(account_url=self._account_url, credential=credential)
        else:
            return ShareServiceClient(account_url=self._account_url, credential=self._get_sync_default_azure_credential(), token_intent='backup')

    @property
    def share_directory_client(self):
        if False:
            return 10
        if self._connection_string:
            return ShareDirectoryClient.from_connection_string(conn_str=self._connection_string, share_name=self.share_name, directory_path=self.directory_path)
        elif self._account_url and (self._sas_token or self._account_access_key):
            credential = self._sas_token or self._account_access_key
            return ShareDirectoryClient(account_url=self._account_url, share_name=self.share_name, directory_path=self.directory_path, credential=credential)
        else:
            return ShareDirectoryClient(account_url=self._account_url, share_name=self.share_name, directory_path=self.directory_path, credential=self._get_sync_default_azure_credential(), token_intent='backup')

    @property
    def share_file_client(self):
        if False:
            i = 10
            return i + 15
        if self._connection_string:
            return ShareFileClient.from_connection_string(conn_str=self._connection_string, share_name=self.share_name, file_path=self.file_path)
        elif self._account_url and (self._sas_token or self._account_access_key):
            credential = self._sas_token or self._account_access_key
            return ShareFileClient(account_url=self._account_url, share_name=self.share_name, file_path=self.file_path, credential=credential)
        else:
            return ShareFileClient(account_url=self._account_url, share_name=self.share_name, file_path=self.file_path, credential=self._get_sync_default_azure_credential(), token_intent='backup')

    def check_for_directory(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if a directory exists on Azure File Share.'
        return self.share_directory_client.exists()

    def list_directories_and_files(self) -> list:
        if False:
            print('Hello World!')
        'Return the list of directories and files stored on a Azure File Share.'
        return list(self.share_directory_client.list_directories_and_files())

    def list_files(self) -> list[str]:
        if False:
            print('Hello World!')
        'Return the list of files stored on a Azure File Share.'
        return [obj.name for obj in self.list_directories_and_files() if isinstance(obj, FileProperties)]

    def create_share(self, share_name: str, **kwargs) -> bool:
        if False:
            while True:
                i = 10
        '\n        Create new Azure File Share.\n\n        :param share_name: Name of the share.\n        :return: True if share is created, False if share already exists.\n        '
        try:
            self.share_service_client.create_share(share_name, **kwargs)
        except Exception as e:
            self.log.warning(e)
            return False
        return True

    def delete_share(self, share_name: str, **kwargs) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete existing Azure File Share.\n\n        :param share_name: Name of the share.\n        :return: True if share is deleted, False if share does not exist.\n        '
        try:
            self.share_service_client.delete_share(share_name, **kwargs)
        except Exception as e:
            self.log.warning(e)
            return False
        return True

    def create_directory(self, **kwargs) -> Any:
        if False:
            return 10
        'Create a new directory on a Azure File Share.'
        return self.share_directory_client.create_directory(**kwargs)

    def get_file(self, file_path: str, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Download a file from Azure File Share.\n\n        :param file_path: Where to store the file.\n        '
        with open(file_path, 'wb') as file_handle:
            data = self.share_file_client.download_file(**kwargs)
            data.readinto(file_handle)

    def get_file_to_stream(self, stream: IO, **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Download a file from Azure File Share.\n\n        :param stream: A filehandle to store the file to.\n        '
        data = self.share_file_client.download_file(**kwargs)
        data.readinto(stream)

    def load_file(self, file_path: str, **kwargs) -> None:
        if False:
            while True:
                i = 10
        '\n        Upload a file to Azure File Share.\n\n        :param file_path: Path to the file to load.\n        '
        with open(file_path, 'rb') as source_file:
            self.share_file_client.upload_file(source_file, **kwargs)

    def load_data(self, string_data: bytes | str | IO, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Upload a string to Azure File Share.\n\n        :param string_data: String/Stream to load.\n        '
        self.share_file_client.upload_file(string_data, **kwargs)

    def test_connection(self):
        if False:
            print('Hello World!')
        'Test Azure FileShare connection.'
        success = (True, 'Successfully connected to Azure File Share.')
        try:
            next(iter(self.share_service_client.list_shares()))
            return success
        except StopIteration:
            return success
        except Exception as e:
            return (False, str(e))