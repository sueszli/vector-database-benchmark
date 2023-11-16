from __future__ import annotations
from functools import cached_property
from typing import Any, Union
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.datalake.store import core, lib, multithread
from azure.identity import ClientSecretCredential
from azure.storage.filedatalake import DataLakeDirectoryClient, DataLakeFileClient, DataLakeServiceClient, DirectoryProperties, FileSystemClient, FileSystemProperties
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.providers.microsoft.azure.utils import AzureIdentityCredentialAdapter, add_managed_identity_connection_widgets, get_field
Credentials = Union[ClientSecretCredential, AzureIdentityCredentialAdapter]

class AzureDataLakeHook(BaseHook):
    """Integration with Azure Data Lake.

    AzureDataLakeHook communicates via a REST API compatible with WebHDFS. Make
    sure that a Airflow connection of type ``azure_data_lake`` exists.
    Authorization can be done by supplying a *login* (=Client ID), *password*
    (=Client Secret), and extra fields *tenant* (Tenant) and *account_name*
    (Account Name). See connection ``azure_data_lake_default`` for an example.

    Client ID and secret should be in user and password parameters.
    Tenant and account name should be extra field as
    ``{"tenant": "<TENANT>", "account_name": "ACCOUNT_NAME"}``.

    :param azure_data_lake_conn_id: Reference to
        :ref:`Azure Data Lake connection<howto/connection:adl>`.
    """
    conn_name_attr = 'azure_data_lake_conn_id'
    default_conn_name = 'azure_data_lake_default'
    conn_type = 'azure_data_lake'
    hook_name = 'Azure Data Lake'

    @staticmethod
    @add_managed_identity_connection_widgets
    def get_connection_form_widgets() -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Returns connection widgets to add to connection form.'
        from flask_appbuilder.fieldwidgets import BS3TextFieldWidget
        from flask_babel import lazy_gettext
        from wtforms import StringField
        return {'tenant': StringField(lazy_gettext('Azure Tenant ID'), widget=BS3TextFieldWidget()), 'account_name': StringField(lazy_gettext('Azure DataLake Store Name'), widget=BS3TextFieldWidget())}

    @staticmethod
    def get_ui_field_behaviour() -> dict[str, Any]:
        if False:
            while True:
                i = 10
        'Returns custom field behaviour.'
        return {'hidden_fields': ['schema', 'port', 'host', 'extra'], 'relabeling': {'login': 'Azure Client ID', 'password': 'Azure Client Secret'}, 'placeholders': {'login': 'client id', 'password': 'secret', 'tenant': 'tenant id', 'account_name': 'datalake store'}}

    def __init__(self, azure_data_lake_conn_id: str=default_conn_name) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.conn_id = azure_data_lake_conn_id
        self._conn: core.AzureDLFileSystem | None = None
        self.account_name: str | None = None

    def _get_field(self, extras, name):
        if False:
            print('Hello World!')
        return get_field(conn_id=self.conn_id, conn_type=self.conn_type, extras=extras, field_name=name)

    def get_conn(self) -> core.AzureDLFileSystem:
        if False:
            for i in range(10):
                print('nop')
        'Return a AzureDLFileSystem object.'
        if not self._conn:
            conn = self.get_connection(self.conn_id)
            extras = conn.extra_dejson
            self.account_name = self._get_field(extras, 'account_name')
            credential: Credentials
            tenant = self._get_field(extras, 'tenant')
            if tenant:
                credential = lib.auth(tenant_id=tenant, client_secret=conn.password, client_id=conn.login)
            else:
                managed_identity_client_id = self._get_field(extras, 'managed_identity_client_id')
                workload_identity_tenant_id = self._get_field(extras, 'workload_identity_tenant_id')
                credential = AzureIdentityCredentialAdapter(managed_identity_client_id=managed_identity_client_id, workload_identity_tenant_id=workload_identity_tenant_id)
            self._conn = core.AzureDLFileSystem(credential, store_name=self.account_name)
            self._conn.connect()
        return self._conn

    def check_for_file(self, file_path: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Check if a file exists on Azure Data Lake.\n\n        :param file_path: Path and name of the file.\n        :return: True if the file exists, False otherwise.\n        '
        try:
            files = self.get_conn().glob(file_path, details=False, invalidate_cache=True)
            return len(files) == 1
        except FileNotFoundError:
            return False

    def upload_file(self, local_path: str, remote_path: str, nthreads: int=64, overwrite: bool=True, buffersize: int=4194304, blocksize: int=4194304, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Upload a file to Azure Data Lake.\n\n        :param local_path: local path. Can be single file, directory (in which case,\n            upload recursively) or glob pattern. Recursive glob patterns using `**`\n            are not supported.\n        :param remote_path: Remote path to upload to; if multiple files, this is the\n            directory root to write within.\n        :param nthreads: Number of threads to use. If None, uses the number of cores.\n        :param overwrite: Whether to forcibly overwrite existing files/directories.\n            If False and remote path is a directory, will quit regardless if any files\n            would be overwritten or not. If True, only matching filenames are actually\n            overwritten.\n        :param buffersize: int [2**22]\n            Number of bytes for internal buffer. This block cannot be bigger than\n            a chunk and cannot be smaller than a block.\n        :param blocksize: int [2**22]\n            Number of bytes for a block. Within each chunk, we write a smaller\n            block for each API call. This block cannot be bigger than a chunk.\n        '
        multithread.ADLUploader(self.get_conn(), lpath=local_path, rpath=remote_path, nthreads=nthreads, overwrite=overwrite, buffersize=buffersize, blocksize=blocksize, **kwargs)

    def download_file(self, local_path: str, remote_path: str, nthreads: int=64, overwrite: bool=True, buffersize: int=4194304, blocksize: int=4194304, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        'Download a file from Azure Blob Storage.\n\n        :param local_path: local path. If downloading a single file, will write to this\n            specific file, unless it is an existing directory, in which case a file is\n            created within it. If downloading multiple files, this is the root\n            directory to write within. Will create directories as required.\n        :param remote_path: remote path/globstring to use to find remote files.\n            Recursive glob patterns using `**` are not supported.\n        :param nthreads: Number of threads to use. If None, uses the number of cores.\n        :param overwrite: Whether to forcibly overwrite existing files/directories.\n            If False and remote path is a directory, will quit regardless if any files\n            would be overwritten or not. If True, only matching filenames are actually\n            overwritten.\n        :param buffersize: int [2**22]\n            Number of bytes for internal buffer. This block cannot be bigger than\n            a chunk and cannot be smaller than a block.\n        :param blocksize: int [2**22]\n            Number of bytes for a block. Within each chunk, we write a smaller\n            block for each API call. This block cannot be bigger than a chunk.\n        '
        multithread.ADLDownloader(self.get_conn(), lpath=local_path, rpath=remote_path, nthreads=nthreads, overwrite=overwrite, buffersize=buffersize, blocksize=blocksize, **kwargs)

    def list(self, path: str) -> list:
        if False:
            return 10
        'List files in Azure Data Lake Storage.\n\n        :param path: full path/globstring to use to list files in ADLS\n        '
        if '*' in path:
            return self.get_conn().glob(path)
        else:
            return self.get_conn().walk(path)

    def remove(self, path: str, recursive: bool=False, ignore_not_found: bool=True) -> None:
        if False:
            return 10
        'Remove files in Azure Data Lake Storage.\n\n        :param path: A directory or file to remove in ADLS\n        :param recursive: Whether to loop into directories in the location and remove the files\n        :param ignore_not_found: Whether to raise error if file to delete is not found\n        '
        try:
            self.get_conn().remove(path=path, recursive=recursive)
        except FileNotFoundError:
            if ignore_not_found:
                self.log.info('File %s not found', path)
            else:
                raise AirflowException(f'File {path} not found')

class AzureDataLakeStorageV2Hook(BaseHook):
    """Interact with a ADLS gen2 storage account.

    It mainly helps to create and manage directories and files in storage
    accounts that have a hierarchical namespace. Using Adls_v2 connection
    details create DataLakeServiceClient object.

    Due to Wasb is marked as legacy and retirement of the (ADLS1), it would
    be nice to implement ADLS gen2 hook for interacting with the storage account.

    .. seealso::
        https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-directory-file-acl-python

    :param adls_conn_id: Reference to the :ref:`adls connection <howto/connection:adls>`.
    :param public_read: Whether an anonymous public read access should be used. default is False
    """
    conn_name_attr = 'adls_conn_id'
    default_conn_name = 'adls_default'
    conn_type = 'adls'
    hook_name = 'Azure Date Lake Storage V2'

    @classmethod
    @add_managed_identity_connection_widgets
    def get_connection_form_widgets(cls) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Returns connection widgets to add to connection form.'
        from flask_appbuilder.fieldwidgets import BS3PasswordFieldWidget, BS3TextFieldWidget
        from flask_babel import lazy_gettext
        from wtforms import PasswordField, StringField
        return {'connection_string': PasswordField(lazy_gettext('ADLS Gen2 Connection String (optional)'), widget=BS3PasswordFieldWidget()), 'tenant_id': StringField(lazy_gettext('Tenant ID (Active Directory)'), widget=BS3TextFieldWidget())}

    @classmethod
    def get_ui_field_behaviour(cls) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Returns custom field behaviour.'
        return {'hidden_fields': ['schema', 'port'], 'relabeling': {'login': 'Client ID (Active Directory)', 'password': 'ADLS Gen2 Key / Client Secret (Active Directory)', 'host': 'ADLS Gen2 Account Name'}, 'placeholders': {'extra': 'additional options for use with FileService and AzureFileVolume', 'login': 'client id', 'password': 'key / secret', 'host': 'storage account name', 'connection_string': 'connection string (overrides auth)', 'tenant_id': 'tenant id'}}

    def __init__(self, adls_conn_id: str, public_read: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.conn_id = adls_conn_id
        self.public_read = public_read

    def _get_field(self, extra_dict, field_name):
        if False:
            i = 10
            return i + 15
        prefix = 'extra__adls__'
        if field_name.startswith('extra__'):
            raise ValueError(f"Got prefixed name {field_name}; please remove the '{prefix}' prefix when using this method.")
        if field_name in extra_dict:
            return extra_dict[field_name] or None
        return extra_dict.get(f'{prefix}{field_name}') or None

    @cached_property
    def service_client(self) -> DataLakeServiceClient:
        if False:
            while True:
                i = 10
        'Return the DataLakeServiceClient object (cached).'
        return self.get_conn()

    def get_conn(self) -> DataLakeServiceClient:
        if False:
            print('Hello World!')
        'Return the DataLakeServiceClient object.'
        conn = self.get_connection(self.conn_id)
        extra = conn.extra_dejson or {}
        connection_string = self._get_field(extra, 'connection_string')
        if connection_string:
            return DataLakeServiceClient.from_connection_string(connection_string, **extra)
        credential: Credentials
        tenant = self._get_field(extra, 'tenant_id')
        if tenant:
            app_id = conn.login
            app_secret = conn.password
            credential = ClientSecretCredential(tenant, app_id, app_secret)
        elif conn.password:
            credential = conn.password
        else:
            managed_identity_client_id = self._get_field(extra, 'managed_identity_client_id')
            workload_identity_tenant_id = self._get_field(extra, 'workload_identity_tenant_id')
            credential = AzureIdentityCredentialAdapter(managed_identity_client_id=managed_identity_client_id, workload_identity_tenant_id=workload_identity_tenant_id)
        return DataLakeServiceClient(account_url=f'https://{conn.host}.dfs.core.windows.net', credential=credential, **extra)

    def create_file_system(self, file_system_name: str) -> None:
        if False:
            return 10
        'Create a new file system under the specified account.\n\n        A container acts as a file system for your files.\n\n        If the file system with the same name already exists, a ResourceExistsError will\n        be raised. This method returns a client with which to interact with the newly\n        created file system.\n        '
        try:
            file_system_client = self.service_client.create_file_system(file_system=file_system_name)
            self.log.info('Created file system: %s', file_system_client.file_system_name)
        except ResourceExistsError:
            self.log.info('Attempted to create file system %r but it already exists.', file_system_name)
        except Exception as e:
            self.log.info('Error while attempting to create file system %r: %s', file_system_name, e)
            raise

    def get_file_system(self, file_system: FileSystemProperties | str) -> FileSystemClient:
        if False:
            while True:
                i = 10
        'Get a client to interact with the specified file system.\n\n        :param file_system: This can either be the name of the file system\n            or an instance of FileSystemProperties.\n        '
        try:
            file_system_client = self.service_client.get_file_system_client(file_system=file_system)
            return file_system_client
        except ResourceNotFoundError:
            self.log.info("file system %r doesn't exists.", file_system)
            raise
        except Exception as e:
            self.log.info('Error while attempting to get file system %r: %s', file_system, e)
            raise

    def create_directory(self, file_system_name: FileSystemProperties | str, directory_name: str, **kwargs) -> DataLakeDirectoryClient:
        if False:
            i = 10
            return i + 15
        'Create a directory under the specified file system.\n\n        :param file_system_name: Name of the file system or instance of FileSystemProperties.\n        :param directory_name: Name of the directory which needs to be created in the file system.\n        '
        result = self.get_file_system(file_system_name).create_directory(directory_name, kwargs)
        return result

    def get_directory_client(self, file_system_name: FileSystemProperties | str, directory_name: DirectoryProperties | str) -> DataLakeDirectoryClient:
        if False:
            for i in range(10):
                print('nop')
        'Get the specific directory under the specified file system.\n\n        :param file_system_name: Name of the file system or instance of FileSystemProperties.\n        :param directory_name: Name of the directory or instance of DirectoryProperties which needs to be\n            retrieved from the file system.\n        '
        try:
            directory_client = self.get_file_system(file_system_name).get_directory_client(directory_name)
            return directory_client
        except ResourceNotFoundError:
            self.log.info("Directory %s doesn't exists in the file system %s", directory_name, file_system_name)
            raise
        except Exception as e:
            self.log.info(e)
            raise

    def create_file(self, file_system_name: FileSystemProperties | str, file_name: str) -> DataLakeFileClient:
        if False:
            return 10
        'Create a file under the file system.\n\n        :param file_system_name: Name of the file system or instance of FileSystemProperties.\n        :param file_name: Name of the file which needs to be created in the file system.\n        '
        file_client = self.get_file_system(file_system_name).create_file(file_name)
        return file_client

    def upload_file(self, file_system_name: FileSystemProperties | str, file_name: str, file_path: str, overwrite: bool=False, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Create a file with data in the file system.\n\n        :param file_system_name: Name of the file system or instance of FileSystemProperties.\n        :param file_name: Name of the file to be created with name.\n        :param file_path: Path to the file to load.\n        :param overwrite: Boolean flag to overwrite an existing file or not.\n        '
        file_client = self.create_file(file_system_name, file_name)
        with open(file_path, 'rb') as data:
            file_client.upload_data(data, overwrite=overwrite, kwargs=kwargs)

    def upload_file_to_directory(self, file_system_name: str, directory_name: str, file_name: str, file_path: str, overwrite: bool=False, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Upload data to a file.\n\n        :param file_system_name: Name of the file system or instance of FileSystemProperties.\n        :param directory_name: Name of the directory.\n        :param file_name: Name of the file to be created with name.\n        :param file_path: Path to the file to load.\n        :param overwrite: Boolean flag to overwrite an existing file or not.\n        '
        directory_client = self.get_directory_client(file_system_name, directory_name=directory_name)
        file_client = directory_client.create_file(file_name, kwargs=kwargs)
        with open(file_path, 'rb') as data:
            file_client.upload_data(data, overwrite=overwrite, kwargs=kwargs)

    def list_files_directory(self, file_system_name: FileSystemProperties | str, directory_name: str) -> list[str]:
        if False:
            while True:
                i = 10
        'List files or directories under the specified file system.\n\n        :param file_system_name: Name of the file system or instance of FileSystemProperties.\n        :param directory_name: Name of the directory.\n        '
        paths = self.get_file_system(file_system=file_system_name).get_paths(directory_name)
        directory_lists = []
        for path in paths:
            directory_lists.append(path.name)
        return directory_lists

    def list_file_system(self, prefix: str | None=None, include_metadata: bool=False, **kwargs: Any) -> list[str]:
        if False:
            return 10
        'List file systems under the specified account.\n\n        :param prefix:\n            Filters the results to return only file systems whose names\n            begin with the specified prefix.\n        :param include_metadata: Specifies that file system metadata be returned in the response.\n            The default value is `False`.\n        '
        file_system = self.service_client.list_file_systems(name_starts_with=prefix, include_metadata=include_metadata)
        file_system_list = []
        for fs in file_system:
            file_system_list.append(fs.name)
        return file_system_list

    def delete_file_system(self, file_system_name: FileSystemProperties | str) -> None:
        if False:
            while True:
                i = 10
        'Delete the file system.\n\n        :param file_system_name: Name of the file system or instance of FileSystemProperties.\n        '
        try:
            self.service_client.delete_file_system(file_system_name)
            self.log.info('Deleted file system: %s', file_system_name)
        except ResourceNotFoundError:
            self.log.info("file system %r doesn't exists.", file_system_name)
        except Exception as e:
            self.log.info('Error while attempting to deleting file system %r: %s', file_system_name, e)
            raise

    def delete_directory(self, file_system_name: FileSystemProperties | str, directory_name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Delete the specified directory in a file system.\n\n        :param file_system_name: Name of the file system or instance of FileSystemProperties.\n        :param directory_name: Name of the directory.\n        '
        directory_client = self.get_directory_client(file_system_name, directory_name)
        directory_client.delete_directory()

    def test_connection(self):
        if False:
            i = 10
            return i + 15
        'Test ADLS Gen2 Storage connection.'
        try:
            next(self.get_conn().list_file_systems(), None)
            return (True, 'Successfully connected to ADLS Gen2 Storage.')
        except Exception as e:
            return (False, str(e))