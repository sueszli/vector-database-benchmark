import io
import random
from typing import Any, Dict, Optional
from unittest import mock
from dagster import resource
from dagster._config.pythonic_config import ConfigurableResource
from dagster._core.definitions.resource_definition import dagster_maintained_resource
from dagster._utils.cached_method import cached_method
from dagster_azure.blob import FakeBlobServiceClient
from .utils import ResourceNotFoundError

@dagster_maintained_resource
@resource({'account_name': str})
def fake_adls2_resource(context):
    if False:
        for i in range(10):
            print('nop')
    return FakeADLS2Resource(account_name=context.resource_config['account_name'])

class FakeADLS2Resource(ConfigurableResource):
    """Stateful mock of an ADLS2Resource for testing.

    Wraps a ``mock.MagicMock``. Containers are implemented using an in-memory dict.
    """
    account_name: str
    storage_account: Optional[str] = None

    @classmethod
    def _is_dagster_maintained(cls) -> bool:
        if False:
            return 10
        return True

    @property
    @cached_method
    def adls2_client(self) -> 'FakeADLS2ServiceClient':
        if False:
            for i in range(10):
                print('nop')
        return FakeADLS2ServiceClient(self.account_name)

    @property
    @cached_method
    def blob_client(self) -> FakeBlobServiceClient:
        if False:
            return 10
        return FakeBlobServiceClient(self.account_name)

    @property
    def lease_client_constructor(self) -> Any:
        if False:
            i = 10
            return i + 15
        return FakeLeaseClient

class FakeLeaseClient:

    def __init__(self, client):
        if False:
            print('Hello World!')
        self.client = client
        self.id = None
        self.client._lease = self

    def acquire(self, lease_duration=-1):
        if False:
            print('Hello World!')
        if self.id is None:
            self.id = random.randint(0, 2 ** 9)
        else:
            raise Exception('Lease already held')

    def release(self):
        if False:
            for i in range(10):
                print('nop')
        self.id = None

    def is_valid(self, lease):
        if False:
            while True:
                i = 10
        if self.id is None:
            return True
        return lease == self.id

class FakeADLS2ServiceClient:
    """Stateful mock of an ADLS2 service client for testing.

    Wraps a ``mock.MagicMock``. Containers are implemented using an in-memory dict.
    """

    def __init__(self, account_name, credential='fake-creds'):
        if False:
            for i in range(10):
                print('nop')
        self._account_name = account_name
        self._credential = mock.MagicMock()
        self._credential.account_key = credential
        self._file_systems = {}

    @property
    def account_name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._account_name

    @property
    def credential(self):
        if False:
            for i in range(10):
                print('nop')
        return self._credential

    @property
    def file_systems(self):
        if False:
            return 10
        return self._file_systems

    def get_file_system_client(self, file_system):
        if False:
            while True:
                i = 10
        return self._file_systems.setdefault(file_system, FakeADLS2FilesystemClient(self.account_name, file_system))

    def get_file_client(self, file_system, file_path):
        if False:
            while True:
                i = 10
        return self.get_file_system_client(file_system).get_file_client(file_path)

class FakeADLS2FilesystemClient:
    """Stateful mock of an ADLS2 filesystem client for testing."""

    def __init__(self, account_name, file_system_name):
        if False:
            for i in range(10):
                print('nop')
        self._file_system: Dict[str, FakeADLS2FileClient] = {}
        self._account_name = account_name
        self._file_system_name = file_system_name

    @property
    def account_name(self):
        if False:
            return 10
        return self._account_name

    @property
    def file_system_name(self):
        if False:
            while True:
                i = 10
        return self._file_system_name

    def keys(self):
        if False:
            while True:
                i = 10
        return self._file_system.keys()

    def get_file_system_properties(self):
        if False:
            while True:
                i = 10
        return {'account_name': self.account_name, 'file_system_name': self.file_system_name}

    def has_file(self, path):
        if False:
            return 10
        return bool(self._file_system.get(path))

    def get_file_client(self, file_path):
        if False:
            for i in range(10):
                print('nop')
        self._file_system.setdefault(file_path, FakeADLS2FileClient(self, file_path))
        return self._file_system[file_path]

    def create_file(self, file):
        if False:
            while True:
                i = 10
        self._file_system.setdefault(file, FakeADLS2FileClient(fs_client=self, name=file))
        return self._file_system[file]

    def delete_file(self, file):
        if False:
            i = 10
            return i + 15
        for k in list(self._file_system.keys()):
            if k.startswith(file):
                del self._file_system[k]

class FakeADLS2FileClient:
    """Stateful mock of an ADLS2 file client for testing."""

    def __init__(self, name, fs_client):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.contents = None
        self._lease = None
        self.fs_client = fs_client

    @property
    def lease(self):
        if False:
            while True:
                i = 10
        return self._lease if self._lease is None else self._lease.id

    def get_file_properties(self):
        if False:
            for i in range(10):
                print('nop')
        if self.contents is None:
            raise ResourceNotFoundError('File does not exist!')
        lease_id = None if self._lease is None else self._lease.id
        return {'lease': lease_id}

    def upload_data(self, contents, overwrite=False, lease=None):
        if False:
            for i in range(10):
                print('nop')
        if self._lease is not None:
            if not self._lease.is_valid(lease):
                raise Exception('Invalid lease!')
        if self.contents is not None or overwrite is True:
            if isinstance(contents, str):
                self.contents = contents.encode('utf8')
            elif isinstance(contents, io.BytesIO):
                self.contents = contents.read()
            elif isinstance(contents, io.StringIO):
                self.contents = contents.read().encode('utf8')
            elif isinstance(contents, bytes):
                self.contents = contents
            else:
                self.contents = contents

    def download_file(self):
        if False:
            for i in range(10):
                print('nop')
        if self.contents is None:
            raise ResourceNotFoundError('File does not exist!')
        return FakeADLS2FileDownloader(contents=self.contents)

    def delete_file(self, lease=None):
        if False:
            print('Hello World!')
        if self._lease is not None:
            if not self._lease.is_valid(lease):
                raise Exception('Invalid lease!')
        self.fs_client.delete_file(self.name)

class FakeADLS2FileDownloader:
    """Mock of an ADLS2 file downloader for testing."""

    def __init__(self, contents):
        if False:
            while True:
                i = 10
        self.contents = contents

    def readall(self):
        if False:
            print('Hello World!')
        return self.contents

    def readinto(self, fileobj):
        if False:
            return 10
        fileobj.write(self.contents)