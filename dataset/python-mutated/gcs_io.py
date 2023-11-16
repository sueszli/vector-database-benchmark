"""Provides an Apache Beam API for operating on GCS."""
from __future__ import annotations
from core.platform import models
import apache_beam as beam
import result
from typing import List, Optional, Tuple, TypedDict, Union
MYPY = False
if MYPY:
    from mypy_imports import app_identity_services
    from mypy_imports import storage_services
storage_services = models.Registry.import_storage_services()
app_identity_services = models.Registry.import_app_identity_services()
BUCKET = app_identity_services.get_gcs_resource_bucket_name()

class ReadFile(beam.PTransform):
    """Read files form the GCS."""

    def __init__(self, bucket: str=BUCKET, label: Optional[str]=None) -> None:
        if False:
            return 10
        'Initializes the ReadFile PTransform.\n\n        Args:\n            bucket: str. The bucket name on the GCS.\n            label: Optional[str]. The label of the PTransform.\n        '
        super().__init__(label=label)
        self.bucket = bucket

    def expand(self, file_paths: beam.PCollection) -> beam.PCollection:
        if False:
            return 10
        'Returns PCollection with file data.\n\n        Args:\n            file_paths: PCollection. The collection of filepaths that will\n                be read.\n\n        Returns:\n            PCollection. The PCollection of the file data.\n        '
        return file_paths | 'Read the file' >> beam.Map(self._read_file)

    def _read_file(self, file_path: str) -> result.Result[Tuple[str, Union[bytes, str]]]:
        if False:
            while True:
                i = 10
        'Helper function to read the contents of a file.\n\n        Args:\n            file_path: str. The name of the file that will be read.\n\n        Returns:\n            data: Tuple[str, bytes]. The file data.\n        '
        try:
            file_data = storage_services.get(self.bucket, file_path)
            return result.Ok((file_path, file_data))
        except Exception:
            err_message: str = 'The file does not exists.'
            return result.Err((file_path, err_message))

class FileObjectDict(TypedDict):
    """Dictionary representing file object that will be written to GCS."""
    filepath: str
    data: bytes

class WriteFile(beam.PTransform):
    """Write files to GCS."""

    def __init__(self, mime_type: str='application/octet-stream', bucket: str=BUCKET, label: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        'Initializes the WriteFile PTransform.\n\n        Args:\n            mime_type: str. The mime_type to assign to the file.\n            bucket: str. The bucket name on the GCS.\n            label: Optional[str]. The label of the PTransform.\n        '
        super().__init__(label=label)
        self.mime_type = mime_type
        self.bucket = bucket

    def expand(self, file_objects: beam.PCollection) -> beam.PCollection:
        if False:
            for i in range(10):
                print('nop')
        'Returns the PCollection of files that have written to the GCS.\n\n        Args:\n            file_objects: PCollection. The collection of file paths and data\n                that will be written.\n\n        Returns:\n            PCollection. The PCollection of the number of bytes that has\n            written to GCS.\n        '
        return file_objects | 'Write files to GCS' >> beam.Map(self._write_file)

    def _write_file(self, file_obj: FileObjectDict) -> int:
        if False:
            print('Hello World!')
        'Helper function to write file to the GCS.\n\n        Args:\n            file_obj: FileObjectDict. The dictionary having file\n                path and file data.\n\n        Returns:\n            int. Returns the number of bytes that has been written to GCS.\n        '
        storage_services.commit(self.bucket, file_obj['filepath'], file_obj['data'], self.mime_type)
        return len(file_obj['data'])

class DeleteFile(beam.PTransform):
    """Delete files from GCS."""

    def __init__(self, bucket: str=BUCKET, label: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initializes the DeleteFile PTransform.\n\n        Args:\n            bucket: str. The bucket name on the GCS.\n            label: Optional[str]. The label of the PTransform.\n        '
        super().__init__(label=label)
        self.bucket = bucket

    def expand(self, file_paths: beam.PCollection) -> beam.pvalue.PDone:
        if False:
            return 10
        'Deletes the files in given PCollection.\n\n        Args:\n            file_paths: PCollection. The collection of filepaths that will\n                be deleted.\n\n        Returns:\n            PCollection. The PCollection of the file data.\n        '
        return file_paths | 'Delete the file' >> beam.Map(self._delete_file)

    def _delete_file(self, file_path: str) -> None:
        if False:
            while True:
                i = 10
        'Helper function to delete the file.\n\n        Args:\n            file_path: str. The name of the file that will be deleted.\n        '
        storage_services.delete(self.bucket, file_path)

class GetFiles(beam.PTransform):
    """Get all files with specefic prefix."""

    def __init__(self, bucket: str=BUCKET, label: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        'Initializes the GetFiles PTransform.\n\n        Args:\n            bucket: str. The bucket name on the GCS.\n            label: Optional[str]. The label of the PTransform.\n        '
        super().__init__(label=label)
        self.bucket = bucket

    def expand(self, prefixes: beam.PCollection) -> beam.PCollection:
        if False:
            print('Hello World!')
        'Returns PCollection with file names.\n\n        Args:\n            prefixes: PCollection. The collection of filepath prefixes.\n\n        Returns:\n            PCollection. The PCollection of the file names.\n        '
        return prefixes | 'Get names of the files' >> beam.Map(self._get_file_with_prefix)

    def _get_file_with_prefix(self, prefix: str) -> List[str]:
        if False:
            return 10
        'Helper function to get file names with the prefix.\n\n        Args:\n            prefix: str. The prefix path of which we want to list\n                all the files.\n\n        Returns:\n            filepaths: List[str]. The file name as key and size of file\n            as value.\n        '
        list_of_blobs = storage_services.listdir(self.bucket, prefix)
        return list((blob.name for blob in list_of_blobs))