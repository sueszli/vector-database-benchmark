"""Azure Blob Storage client.
"""
import errno
import io
import logging
import os
import re
import tempfile
import time
from apache_beam.internal.azure import auth
from apache_beam.io.filesystemio import Downloader
from apache_beam.io.filesystemio import DownloaderStream
from apache_beam.io.filesystemio import Uploader
from apache_beam.io.filesystemio import UploaderStream
from apache_beam.options.pipeline_options import AzureOptions
from apache_beam.utils import retry
from apache_beam.utils.annotations import deprecated
_LOGGER = logging.getLogger(__name__)
try:
    from azure.core.exceptions import ResourceNotFoundError
    from azure.storage.blob import BlobServiceClient, ContentSettings
    AZURE_DEPS_INSTALLED = True
except ImportError:
    AZURE_DEPS_INSTALLED = False
DEFAULT_READ_BUFFER_SIZE = 16 * 1024 * 1024
MAX_BATCH_OPERATION_SIZE = 100

def parse_azfs_path(azfs_path, blob_optional=False, get_account=False):
    if False:
        while True:
            i = 10
    'Return the storage account, the container and\n  blob names of the given azfs:// path.\n  '
    match = re.match('^azfs://([a-z0-9]{3,24})/([a-z0-9](?![a-z0-9-]*--[a-z0-9-]*)[a-z0-9-]{1,61}[a-z0-9])/(.*)$', azfs_path)
    if match is None or (match.group(3) == '' and (not blob_optional)):
        raise ValueError('Azure Blob Storage path must be in the form azfs://<storage-account>/<container>/<path>.')
    result = None
    if get_account:
        result = (match.group(1), match.group(2), match.group(3))
    else:
        result = (match.group(2), match.group(3))
    return result

def get_azfs_url(storage_account, container, blob=''):
    if False:
        while True:
            i = 10
    'Returns the url in the form of\n   https://account.blob.core.windows.net/container/blob-name\n  '
    return 'https://' + storage_account + '.blob.core.windows.net/' + container + '/' + blob

class Blob:
    """A Blob in Azure Blob Storage."""

    def __init__(self, etag, name, last_updated, size, mime_type):
        if False:
            while True:
                i = 10
        self.etag = etag
        self.name = name
        self.last_updated = last_updated
        self.size = size
        self.mime_type = mime_type

class BlobStorageIOError(IOError, retry.PermanentException):
    """Blob Strorage IO error that should not be retried."""
    pass

class BlobStorageError(Exception):
    """Blob Storage client error."""

    def __init__(self, message=None, code=None):
        if False:
            print('Hello World!')
        self.message = message
        self.code = code

class BlobStorageIO(object):
    """Azure Blob Storage I/O client."""

    def __init__(self, client=None, pipeline_options=None):
        if False:
            return 10
        if client is None:
            azure_options = pipeline_options.view_as(AzureOptions)
            connect_str = azure_options.azure_connection_string or os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            if connect_str:
                self.client = BlobServiceClient.from_connection_string(conn_str=connect_str)
            else:
                credential = auth.get_service_credentials(pipeline_options)
                self.client = BlobServiceClient(account_url=azure_options.blob_service_endpoint, credential=credential)
        else:
            self.client = client
        if not AZURE_DEPS_INSTALLED:
            raise RuntimeError('Azure dependencies are not installed. Unable to run.')

    def open(self, filename, mode='r', read_buffer_size=DEFAULT_READ_BUFFER_SIZE, mime_type='application/octet-stream'):
        if False:
            while True:
                i = 10
        "Open an Azure Blob Storage file path for reading or writing.\n\n    Args:\n      filename (str): Azure Blob Storage file path in the form\n                      ``azfs://<storage-account>/<container>/<path>``.\n      mode (str): ``'r'`` for reading or ``'w'`` for writing.\n      read_buffer_size (int): Buffer size to use during read operations.\n      mime_type (str): Mime type to set for write operations.\n\n    Returns:\n      Azure Blob Storage file object.\n    Raises:\n      ValueError: Invalid open file mode.\n    "
        if mode == 'r' or mode == 'rb':
            downloader = BlobStorageDownloader(self.client, filename, buffer_size=read_buffer_size)
            return io.BufferedReader(DownloaderStream(downloader, read_buffer_size=read_buffer_size, mode=mode), buffer_size=read_buffer_size)
        elif mode == 'w' or mode == 'wb':
            uploader = BlobStorageUploader(self.client, filename, mime_type)
            return io.BufferedWriter(UploaderStream(uploader, mode=mode), buffer_size=128 * 1024)
        else:
            raise ValueError('Invalid file open mode: %s.' % mode)

    @retry.with_exponential_backoff(retry_filter=retry.retry_on_beam_io_error_filter)
    def copy(self, src, dest):
        if False:
            i = 10
            return i + 15
        'Copies a single Azure Blob Storage blob from src to dest.\n\n    Args:\n      src: Blob Storage file path pattern in the form\n           azfs://<storage-account>/<container>/[name].\n      dest: Blob Storage file path pattern in the form\n            azfs://<storage-account>/<container>/[name].\n\n    Raises:\n      TimeoutError: on timeout.\n    '
        (src_storage_account, src_container, src_blob) = parse_azfs_path(src, get_account=True)
        (dest_container, dest_blob) = parse_azfs_path(dest)
        source_blob = get_azfs_url(src_storage_account, src_container, src_blob)
        copied_blob = self.client.get_blob_client(dest_container, dest_blob)
        try:
            copied_blob.start_copy_from_url(source_blob)
        except ResourceNotFoundError as e:
            message = e.reason
            code = e.status_code
            raise BlobStorageError(message, code)

    def copy_tree(self, src, dest):
        if False:
            print('Hello World!')
        'Renames the given Azure Blob storage directory and its contents\n    recursively from src to dest.\n\n    Args:\n      src: Blob Storage file path pattern in the form\n           azfs://<storage-account>/<container>/[name].\n      dest: Blob Storage file path pattern in the form\n            azfs://<storage-account>/<container>/[name].\n\n    Returns:\n      List of tuples of (src, dest, exception) where exception is None if the\n      operation succeeded or the relevant exception if the operation failed.\n    '
        assert src.endswith('/')
        assert dest.endswith('/')
        results = []
        for entry in self.list_prefix(src):
            rel_path = entry[len(src):]
            try:
                self.copy(entry, dest + rel_path)
                results.append((entry, dest + rel_path, None))
            except BlobStorageError as e:
                results.append((entry, dest + rel_path, e))
        return results

    def copy_paths(self, src_dest_pairs):
        if False:
            print('Hello World!')
        'Copies the given Azure Blob Storage blobs from src to dest. This can\n    handle directory or file paths.\n\n    Args:\n      src_dest_pairs: List of (src, dest) tuples of\n                      azfs://<storage-account>/<container>/[name] file paths\n                      to copy from src to dest.\n\n    Returns:\n      List of tuples of (src, dest, exception) in the same order as the\n      src_dest_pairs argument, where exception is None if the operation\n      succeeded or the relevant exception if the operation failed.\n    '
        if not src_dest_pairs:
            return []
        results = []
        for (src_path, dest_path) in src_dest_pairs:
            if src_path.endswith('/') and dest_path.endswith('/'):
                try:
                    results += self.copy_tree(src_path, dest_path)
                except BlobStorageError as e:
                    results.append((src_path, dest_path, e))
            elif not src_path.endswith('/') and (not dest_path.endswith('/')):
                try:
                    self.copy(src_path, dest_path)
                    results.append((src_path, dest_path, None))
                except BlobStorageError as e:
                    results.append((src_path, dest_path, e))
            else:
                e = BlobStorageError('Unable to copy mismatched paths' + '(directory, non-directory): %s, %s' % (src_path, dest_path), 400)
                results.append((src_path, dest_path, e))
        return results

    def rename(self, src, dest):
        if False:
            print('Hello World!')
        'Renames the given Azure Blob Storage blob from src to dest.\n\n    Args:\n      src: Blob Storage file path pattern in the form\n           azfs://<storage-account>/<container>/[name].\n      dest: Blob Storage file path pattern in the form\n            azfs://<storage-account>/<container>/[name].\n    '
        self.copy(src, dest)
        self.delete(src)

    def rename_files(self, src_dest_pairs):
        if False:
            print('Hello World!')
        'Renames the given Azure Blob Storage blobs from src to dest.\n\n    Args:\n      src_dest_pairs: List of (src, dest) tuples of\n                      azfs://<storage-account>/<container>/[name]\n                      file paths to rename from src to dest.\n    Returns: List of tuples of (src, dest, exception) in the same order as the\n             src_dest_pairs argument, where exception is None if the operation\n             succeeded or the relevant exception if the operation failed.\n    '
        if not src_dest_pairs:
            return []
        for (src, dest) in src_dest_pairs:
            if src.endswith('/') or dest.endswith('/'):
                raise ValueError('Unable to rename a directory.')
        copy_results = self.copy_paths(src_dest_pairs)
        paths_to_delete = [src for (src, _, error) in copy_results if error is None]
        delete_results = self.delete_files(paths_to_delete)
        results = []
        delete_results_dict = {src: error for (src, error) in delete_results}
        for (src, dest, error) in copy_results:
            if error is not None:
                results.append((src, dest, error))
            elif delete_results_dict[src] is not None:
                results.append((src, dest, delete_results_dict[src]))
            else:
                results.append((src, dest, None))
        return results

    def exists(self, path):
        if False:
            return 10
        'Returns whether the given Azure Blob Storage blob exists.\n\n    Args:\n      path: Azure Blob Storage file path pattern in the form\n            azfs://<storage-account>/<container>/[name].\n    '
        try:
            self._blob_properties(path)
            return True
        except ResourceNotFoundError as e:
            if e.status_code == 404:
                return False
            else:
                raise

    def size(self, path):
        if False:
            for i in range(10):
                print('nop')
        'Returns the size of a single Blob Storage blob.\n\n    This method does not perform glob expansion. Hence the\n    given path must be for a single Blob Storage blob.\n\n    Returns: size of the Blob Storage blob in bytes.\n    '
        return self._blob_properties(path).size

    def last_updated(self, path):
        if False:
            i = 10
            return i + 15
        'Returns the last updated epoch time of a single\n    Azure Blob Storage blob.\n\n    This method does not perform glob expansion. Hence the\n    given path must be for a single Azure Blob Storage blob.\n\n    Returns: last updated time of the Azure Blob Storage blob\n    in seconds.\n    '
        return self._updated_to_seconds(self._blob_properties(path).last_modified)

    def checksum(self, path):
        if False:
            while True:
                i = 10
        'Looks up the checksum of an Azure Blob Storage blob.\n\n    Args:\n      path: Azure Blob Storage file path pattern in the form\n            azfs://<storage-account>/<container>/[name].\n    '
        return self._blob_properties(path).etag

    def _status(self, path):
        if False:
            print('Hello World!')
        'For internal use only; no backwards-compatibility guarantees.\n\n    Returns supported fields (checksum, last_updated, size) of a single object\n    as a dict at once.\n\n    This method does not perform glob expansion. Hence the given path must be\n    for a single blob property.\n\n    Returns: dict of fields of the blob property.\n    '
        properties = self._blob_properties(path)
        file_status = {}
        if hasattr(properties, 'etag'):
            file_status['checksum'] = properties.etag
        if hasattr(properties, 'last_modified'):
            file_status['last_updated'] = self._updated_to_seconds(properties.last_modified)
        if hasattr(properties, 'size'):
            file_status['size'] = properties.size
        return file_status

    @retry.with_exponential_backoff(retry_filter=retry.retry_on_beam_io_error_filter)
    def _blob_properties(self, path):
        if False:
            return 10
        'Returns a blob properties object for the given path\n\n    This method does not perform glob expansion. Hence the given path must be\n    for a single blob properties object.\n\n    Returns: blob properties.\n    '
        (container, blob) = parse_azfs_path(path)
        blob_to_check = self.client.get_blob_client(container, blob)
        try:
            properties = blob_to_check.get_blob_properties()
        except ResourceNotFoundError as e:
            message = e.reason
            code = e.status_code
            raise BlobStorageError(message, code)
        return properties

    @staticmethod
    def _updated_to_seconds(updated):
        if False:
            for i in range(10):
                print('nop')
        'Helper function transform the updated field of response to seconds'
        return time.mktime(updated.timetuple()) - time.timezone + updated.microsecond / 1000000.0

    @retry.with_exponential_backoff(retry_filter=retry.retry_on_beam_io_error_filter)
    def delete(self, path):
        if False:
            while True:
                i = 10
        'Deletes a single blob at the given Azure Blob Storage path.\n\n    Args:\n      path: Azure Blob Storage file path pattern in the form\n            azfs://<storage-account>/<container>/[name].\n    '
        (container, blob) = parse_azfs_path(path)
        blob_to_delete = self.client.get_blob_client(container, blob)
        try:
            blob_to_delete.delete_blob()
        except ResourceNotFoundError as e:
            if e.status_code == 404:
                return
            else:
                logging.error('HTTP error while deleting file %s', path)
                raise e

    def delete_paths(self, paths):
        if False:
            for i in range(10):
                print('nop')
        'Deletes the given Azure Blob Storage blobs from src to dest.\n    This can handle directory or file paths.\n\n    Args:\n      paths: list of Azure Blob Storage paths in the form\n             azfs://<storage-account>/<container>/[name] that give the\n             file blobs to be deleted.\n\n    Returns:\n      List of tuples of (src, dest, exception) in the same order as the\n      src_dest_pairs argument, where exception is None if the operation\n      succeeded or the relevant exception if the operation failed.\n    '
        (directories, blobs) = ([], [])
        for path in paths:
            if path.endswith('/'):
                directories.append(path)
            else:
                blobs.append(path)
        results = {}
        for directory in directories:
            directory_result = dict(self.delete_tree(directory))
            results.update(directory_result)
        blobs_results = dict(self.delete_files(blobs))
        results.update(blobs_results)
        return results

    def delete_tree(self, root):
        if False:
            i = 10
            return i + 15
        'Deletes all blobs under the given Azure BlobStorage virtual\n    directory.\n\n    Args:\n      path: Azure Blob Storage file path pattern in the form\n            azfs://<storage-account>/<container>/[name]\n            (ending with a "/").\n\n    Returns:\n      List of tuples of (path, exception), where each path is a blob\n      under the given root. exception is None if the operation succeeded\n      or the relevant exception if the operation failed.\n    '
        assert root.endswith('/')
        paths_to_delete = self.list_prefix(root)
        return self.delete_files(paths_to_delete)

    def delete_files(self, paths):
        if False:
            for i in range(10):
                print('nop')
        'Deletes the given Azure Blob Storage blobs from src to dest.\n\n    Args:\n      paths: list of Azure Blob Storage paths in the form\n             azfs://<storage-account>/<container>/[name] that give the\n             file blobs to be deleted.\n\n    Returns:\n      List of tuples of (src, dest, exception) in the same order as the\n      src_dest_pairs argument, where exception is None if the operation\n      succeeded or the relevant exception if the operation failed.\n    '
        if not paths:
            return []
        (containers, blobs) = zip(*[parse_azfs_path(path, get_account=False) for path in paths])
        grouped_blobs = {container: [] for container in containers}
        for (container, blob) in zip(containers, blobs):
            grouped_blobs[container].append(blob)
        results = {}
        for (container, blobs) in grouped_blobs.items():
            for i in range(0, len(blobs), MAX_BATCH_OPERATION_SIZE):
                blobs_to_delete = blobs[i:i + MAX_BATCH_OPERATION_SIZE]
                results.update(self._delete_batch(container, blobs_to_delete))
        final_results = [(path, results[parse_azfs_path(path, get_account=False)]) for path in paths]
        return final_results

    @retry.with_exponential_backoff(retry_filter=retry.retry_on_beam_io_error_filter)
    def _delete_batch(self, container, blobs):
        if False:
            for i in range(10):
                print('nop')
        'A helper method. Azure Blob Storage Python Client allows batch\n    deletions for blobs within the same container.\n\n    Args:\n      container: container name.\n      blobs: list of blobs to be deleted.\n\n    Returns:\n      Dictionary of the form {(container, blob): error}, where error is\n      None if the operation succeeded.\n    '
        container_client = self.client.get_container_client(container)
        results = {}
        for blob in blobs:
            try:
                response = container_client.delete_blob(blob)
                results[container, blob] = response
            except ResourceNotFoundError as e:
                results[container, blob] = e.status_code
        return results

    @deprecated(since='2.45.0', current='list_files')
    def list_prefix(self, path, with_metadata=False):
        if False:
            for i in range(10):
                print('nop')
        'Lists files matching the prefix.\n\n    Args:\n      path: Azure Blob Storage file path pattern in the form\n            azfs://<storage-account>/<container>/[name].\n      with_metadata: Experimental. Specify whether returns file metadata.\n\n    Returns:\n      If ``with_metadata`` is False: dict of file name -> size; if\n        ``with_metadata`` is True: dict of file name -> tuple(size, timestamp).\n    '
        file_info = {}
        for file_metadata in self.list_files(path, with_metadata):
            file_info[file_metadata[0]] = file_metadata[1]
        return file_info

    def list_files(self, path, with_metadata=False):
        if False:
            while True:
                i = 10
        'Lists files matching the prefix.\n\n    Args:\n      path: Azure Blob Storage file path pattern in the form\n            azfs://<storage-account>/<container>/[name].\n      with_metadata: Experimental. Specify whether returns file metadata.\n\n    Returns:\n      If ``with_metadata`` is False: generator of tuple(file name, size); if\n      ``with_metadata`` is True: generator of\n      tuple(file name, tuple(size, timestamp)).\n    '
        (storage_account, container, blob) = parse_azfs_path(path, blob_optional=True, get_account=True)
        file_info = set()
        counter = 0
        start_time = time.time()
        if with_metadata:
            logging.debug('Starting the file information of the input')
        else:
            logging.debug('Starting the size estimation of the input')
        container_client = self.client.get_container_client(container)
        response = retry.with_exponential_backoff(retry_filter=retry.retry_on_beam_io_error_filter)(container_client.list_blobs)(name_starts_with=blob)
        for item in response:
            file_name = 'azfs://%s/%s/%s' % (storage_account, container, item.name)
            if file_name not in file_info:
                file_info.add(file_name)
                counter += 1
                if counter % 10000 == 0:
                    if with_metadata:
                        logging.info('Finished computing file information of: %s files', len(file_info))
                    else:
                        logging.info('Finished computing size of: %s files', len(file_info))
                if with_metadata:
                    yield (file_name, (item.size, self._updated_to_seconds(item.last_modified)))
                else:
                    yield (file_name, item.size)
        logging.log(logging.INFO if counter > 0 else logging.DEBUG, 'Finished listing %s files in %s seconds.', counter, time.time() - start_time)

class BlobStorageDownloader(Downloader):

    def __init__(self, client, path, buffer_size):
        if False:
            for i in range(10):
                print('nop')
        self._client = client
        self._path = path
        (self._container, self._blob) = parse_azfs_path(path)
        self._buffer_size = buffer_size
        self._blob_to_download = self._client.get_blob_client(self._container, self._blob)
        try:
            properties = self._get_object_properties()
        except ResourceNotFoundError as http_error:
            if http_error.status_code == 404:
                raise IOError(errno.ENOENT, 'Not found: %s' % self._path)
            else:
                _LOGGER.error('HTTP error while requesting file %s: %s', self._path, http_error)
                raise
        self._size = properties.size

    @retry.with_exponential_backoff(retry_filter=retry.retry_on_beam_io_error_filter)
    def _get_object_properties(self):
        if False:
            for i in range(10):
                print('nop')
        return self._blob_to_download.get_blob_properties()

    @property
    def size(self):
        if False:
            while True:
                i = 10
        return self._size

    def get_range(self, start, end):
        if False:
            while True:
                i = 10
        blob_data = self._blob_to_download.download_blob(start, end - start)
        return blob_data.readall()

class BlobStorageUploader(Uploader):

    def __init__(self, client, path, mime_type='application/octet-stream'):
        if False:
            return 10
        self._client = client
        self._path = path
        (self._container, self._blob) = parse_azfs_path(path)
        self._content_settings = ContentSettings(mime_type)
        self._blob_to_upload = self._client.get_blob_client(self._container, self._blob)
        self._temporary_file = tempfile.NamedTemporaryFile()

    def put(self, data):
        if False:
            i = 10
            return i + 15
        self._temporary_file.write(data.tobytes())

    def finish(self):
        if False:
            while True:
                i = 10
        self._temporary_file.seek(0)
        with open(self._temporary_file.name, 'rb') as f:
            self._blob_to_upload.upload_blob(f.read(), overwrite=True, content_settings=self._content_settings)