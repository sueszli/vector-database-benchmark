from io import BytesIO
from pathlib import Path
from tempfile import gettempdir
from typing import Any
from typing import Optional
from typing import Type
from typing import Union
from typing_extensions import Self
from . import BlobDeposit
from . import BlobRetrieval
from . import BlobStorageClient
from . import BlobStorageClientConfig
from . import BlobStorageConfig
from . import BlobStorageConnection
from . import SyftObjectRetrieval
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...service.response import SyftSuccess
from ...types.blob_storage import BlobStorageEntry
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.blob_storage import SecureFilePathLocation
from ...types.syft_object import SYFT_OBJECT_VERSION_1

@serializable()
class OnDiskBlobDeposit(BlobDeposit):
    __canonical_name__ = 'OnDiskBlobDeposit'
    __version__ = SYFT_OBJECT_VERSION_1

    def write(self, data: BytesIO) -> Union[SyftSuccess, SyftError]:
        if False:
            print('Hello World!')
        from ...service.service import from_api_or_context
        write_to_disk_method = from_api_or_context(func_or_path='blob_storage.write_to_disk', syft_node_location=self.syft_node_location, syft_client_verify_key=self.syft_client_verify_key)
        return write_to_disk_method(data=data.read(), uid=self.blob_storage_entry_id)

class OnDiskBlobStorageConnection(BlobStorageConnection):
    _base_directory: Path

    def __init__(self, base_directory: Path) -> None:
        if False:
            while True:
                i = 10
        self._base_directory = base_directory

    def __enter__(self) -> Self:
        if False:
            print('Hello World!')
        return self

    def __exit__(self, *exc) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def read(self, fp: SecureFilePathLocation, type_: Optional[Type]) -> BlobRetrieval:
        if False:
            while True:
                i = 10
        file_path = self._base_directory / fp.path
        return SyftObjectRetrieval(syft_object=file_path.read_bytes(), file_name=file_path.name, type_=type_)

    def allocate(self, obj: CreateBlobStorageEntry) -> Union[SecureFilePathLocation, SyftError]:
        if False:
            for i in range(10):
                print('nop')
        try:
            return SecureFilePathLocation(path=str((self._base_directory / obj.file_name).absolute()))
        except Exception as e:
            return SyftError(message=f'Failed to allocate: {e}')

    def write(self, obj: BlobStorageEntry) -> BlobDeposit:
        if False:
            for i in range(10):
                print('nop')
        return OnDiskBlobDeposit(blob_storage_entry_id=obj.id)

    def delete(self, fp: SecureFilePathLocation) -> Union[SyftSuccess, SyftError]:
        if False:
            for i in range(10):
                print('nop')
        try:
            (self._base_directory / fp.path).unlink()
            return SyftSuccess(message='Successfully deleted file.')
        except FileNotFoundError as e:
            return SyftError(message=f'Failed to delete file: {e}')

@serializable()
class OnDiskBlobStorageClientConfig(BlobStorageClientConfig):
    base_directory: Path = Path(gettempdir())

@serializable()
class OnDiskBlobStorageClient(BlobStorageClient):
    config: OnDiskBlobStorageClientConfig

    def __init__(self, **data: Any):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**data)
        self.config.base_directory.mkdir(exist_ok=True)

    def connect(self) -> BlobStorageConnection:
        if False:
            while True:
                i = 10
        return OnDiskBlobStorageConnection(self.config.base_directory)

@serializable()
class OnDiskBlobStorageConfig(BlobStorageConfig):
    client_type: Type[BlobStorageClient] = OnDiskBlobStorageClient
    client_config: OnDiskBlobStorageClientConfig = OnDiskBlobStorageClientConfig()