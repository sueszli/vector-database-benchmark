"""AWS S3 client
"""
import errno
import io
import logging
import re
import time
import traceback
from apache_beam.io.aws.clients.s3 import messages
from apache_beam.io.filesystemio import Downloader
from apache_beam.io.filesystemio import DownloaderStream
from apache_beam.io.filesystemio import Uploader
from apache_beam.io.filesystemio import UploaderStream
from apache_beam.utils import retry
from apache_beam.utils.annotations import deprecated
try:
    from apache_beam.io.aws.clients.s3 import boto3_client
    BOTO3_INSTALLED = True
except ImportError:
    BOTO3_INSTALLED = False
MAX_BATCH_OPERATION_SIZE = 100

def parse_s3_path(s3_path, object_optional=False):
    if False:
        i = 10
        return i + 15
    'Return the bucket and object names of the given s3:// path.'
    match = re.match('^s3://([^/]+)/(.*)$', s3_path)
    if match is None or (match.group(2) == '' and (not object_optional)):
        raise ValueError('S3 path must be in the form s3://<bucket>/<object>.')
    return (match.group(1), match.group(2))

class S3IO(object):
    """S3 I/O client."""

    def __init__(self, client=None, options=None):
        if False:
            return 10
        if client is None and options is None:
            raise ValueError('Must provide one of client or options')
        if client is not None:
            self.client = client
        elif BOTO3_INSTALLED:
            self.client = boto3_client.Client(options=options)
        else:
            message = 'AWS dependencies are not installed, and no alternative client was provided to S3IO.'
            raise RuntimeError(message)

    def open(self, filename, mode='r', read_buffer_size=16 * 1024 * 1024, mime_type='application/octet-stream'):
        if False:
            return 10
        "Open an S3 file path for reading or writing.\n\n    Args:\n      filename (str): S3 file path in the form ``s3://<bucket>/<object>``.\n      mode (str): ``'r'`` for reading or ``'w'`` for writing.\n      read_buffer_size (int): Buffer size to use during read operations.\n      mime_type (str): Mime type to set for write operations.\n\n    Returns:\n      S3 file object.\n\n    Raises:\n      ValueError: Invalid open file mode.\n    "
        if mode == 'r' or mode == 'rb':
            downloader = S3Downloader(self.client, filename, buffer_size=read_buffer_size)
            return io.BufferedReader(DownloaderStream(downloader, mode=mode), buffer_size=read_buffer_size)
        elif mode == 'w' or mode == 'wb':
            uploader = S3Uploader(self.client, filename, mime_type)
            return io.BufferedWriter(UploaderStream(uploader, mode=mode), buffer_size=128 * 1024)
        else:
            raise ValueError('Invalid file open mode: %s.' % mode)

    @deprecated(since='2.45.0', current='list_files')
    def list_prefix(self, path, with_metadata=False):
        if False:
            for i in range(10):
                print('nop')
        'Lists files matching the prefix.\n\n    ``list_prefix`` has been deprecated. Use `list_files` instead, which returns\n    a generator of file information instead of a dict.\n\n    Args:\n      path: S3 file path pattern in the form s3://<bucket>/[name].\n      with_metadata: Experimental. Specify whether returns file metadata.\n\n    Returns:\n      If ``with_metadata`` is False: dict of file name -> size; if\n        ``with_metadata`` is True: dict of file name -> tuple(size, timestamp).\n    '
        file_info = {}
        for file_metadata in self.list_files(path, with_metadata):
            file_info[file_metadata[0]] = file_metadata[1]
        return file_info

    def list_files(self, path, with_metadata=False):
        if False:
            for i in range(10):
                print('nop')
        'Lists files matching the prefix.\n\n    Args:\n      path: S3 file path pattern in the form s3://<bucket>/[name].\n      with_metadata: Experimental. Specify whether returns file metadata.\n\n    Returns:\n      If ``with_metadata`` is False: generator of tuple(file name, size); if\n      ``with_metadata`` is True: generator of\n      tuple(file name, tuple(size, timestamp)).\n    '
        (bucket, prefix) = parse_s3_path(path, object_optional=True)
        request = messages.ListRequest(bucket=bucket, prefix=prefix)
        file_info = set()
        counter = 0
        start_time = time.time()
        if with_metadata:
            logging.debug('Starting the file information of the input')
        else:
            logging.debug('Starting the size estimation of the input')
        while True:
            try:
                response = retry.with_exponential_backoff(retry_filter=retry.retry_on_server_errors_and_timeout_filter)(self.client.list)(request)
            except messages.S3ClientError as e:
                if e.code == 404:
                    break
                else:
                    raise e
            for item in response.items:
                file_name = 's3://%s/%s' % (bucket, item.key)
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
            if response.next_token:
                request.continuation_token = response.next_token
            else:
                break
        logging.log(logging.INFO if counter > 0 else logging.DEBUG, 'Finished listing %s files in %s seconds.', counter, time.time() - start_time)
        return file_info

    def checksum(self, path):
        if False:
            print('Hello World!')
        'Looks up the checksum of an S3 object.\n\n    Args:\n      path: S3 file path pattern in the form s3://<bucket>/<name>.\n    '
        return self._s3_object(path).etag

    @retry.with_exponential_backoff(retry_filter=retry.retry_on_server_errors_and_timeout_filter)
    def copy(self, src, dest):
        if False:
            i = 10
            return i + 15
        'Copies a single S3 file object from src to dest.\n\n    Args:\n      src: S3 file path pattern in the form s3://<bucket>/<name>.\n      dest: S3 file path pattern in the form s3://<bucket>/<name>.\n\n    Raises:\n      TimeoutError: on timeout.\n    '
        (src_bucket, src_key) = parse_s3_path(src)
        (dest_bucket, dest_key) = parse_s3_path(dest)
        request = messages.CopyRequest(src_bucket, src_key, dest_bucket, dest_key)
        self.client.copy(request)

    def copy_paths(self, src_dest_pairs):
        if False:
            i = 10
            return i + 15
        'Copies the given S3 objects from src to dest. This can handle directory\n    or file paths.\n\n    Args:\n      src_dest_pairs: list of (src, dest) tuples of s3://<bucket>/<name> file\n                      paths to copy from src to dest\n    Returns: List of tuples of (src, dest, exception) in the same order as the\n            src_dest_pairs argument, where exception is None if the operation\n            succeeded or the relevant exception if the operation failed.\n    '
        if not src_dest_pairs:
            return []
        results = []
        for (src_path, dest_path) in src_dest_pairs:
            if src_path.endswith('/') and dest_path.endswith('/'):
                try:
                    results += self.copy_tree(src_path, dest_path)
                except messages.S3ClientError as err:
                    results.append((src_path, dest_path, err))
            elif not src_path.endswith('/') and (not dest_path.endswith('/')):
                (src_bucket, src_key) = parse_s3_path(src_path)
                (dest_bucket, dest_key) = parse_s3_path(dest_path)
                request = messages.CopyRequest(src_bucket, src_key, dest_bucket, dest_key)
                try:
                    self.client.copy(request)
                    results.append((src_path, dest_path, None))
                except messages.S3ClientError as err:
                    results.append((src_path, dest_path, err))
            else:
                e = messages.S3ClientError("Can't copy mismatched paths (one directory, one non-directory):" + ' %s, %s' % (src_path, dest_path), 400)
                results.append((src_path, dest_path, e))
        return results

    def copy_tree(self, src, dest):
        if False:
            while True:
                i = 10
        "Renames the given S3 directory and it's contents recursively\n    from src to dest.\n\n    Args:\n      src: S3 file path pattern in the form s3://<bucket>/<name>/.\n      dest: S3 file path pattern in the form s3://<bucket>/<name>/.\n\n    Returns:\n      List of tuples of (src, dest, exception) where exception is None if the\n      operation succeeded or the relevant exception if the operation failed.\n    "
        assert src.endswith('/')
        assert dest.endswith('/')
        results = []
        for entry in self.list_prefix(src):
            rel_path = entry[len(src):]
            try:
                self.copy(entry, dest + rel_path)
                results.append((entry, dest + rel_path, None))
            except messages.S3ClientError as e:
                results.append((entry, dest + rel_path, e))
        return results

    @retry.with_exponential_backoff(retry_filter=retry.retry_on_server_errors_and_timeout_filter)
    def delete(self, path):
        if False:
            i = 10
            return i + 15
        'Deletes a single S3 file object from src to dest.\n\n    Args:\n      src: S3 file path pattern in the form s3://<bucket>/<name>/.\n      dest: S3 file path pattern in the form s3://<bucket>/<name>/.\n\n    Returns:\n      List of tuples of (src, dest, exception) in the same order as the\n      src_dest_pairs argument, where exception is None if the operation\n      succeeded or the relevant exception if the operation failed.\n    '
        (bucket, object_path) = parse_s3_path(path)
        request = messages.DeleteRequest(bucket, object_path)
        try:
            self.client.delete(request)
        except messages.S3ClientError as e:
            if e.code == 404:
                return
            else:
                logging.error('HTTP error while deleting file %s: %s', path, 3)
                raise e

    def delete_paths(self, paths):
        if False:
            for i in range(10):
                print('nop')
        'Deletes the given S3 objects from src to dest. This can handle directory\n    or file paths.\n\n    Args:\n      src: S3 file path pattern in the form s3://<bucket>/<name>/.\n      dest: S3 file path pattern in the form s3://<bucket>/<name>/.\n\n    Returns:\n      List of tuples of (src, dest, exception) in the same order as the\n      src_dest_pairs argument, where exception is None if the operation\n      succeeded or the relevant exception if the operation failed.\n    '
        (directories, not_directories) = ([], [])
        for path in paths:
            if path.endswith('/'):
                directories.append(path)
            else:
                not_directories.append(path)
        results = {}
        for directory in directories:
            dir_result = dict(self.delete_tree(directory))
            results.update(dir_result)
        not_directory_results = dict(self.delete_files(not_directories))
        results.update(not_directory_results)
        return results

    def delete_files(self, paths, max_batch_size=1000):
        if False:
            while True:
                i = 10
        'Deletes the given S3 file object from src to dest.\n\n    Args:\n      paths: List of S3 file paths in the form s3://<bucket>/<name>\n      max_batch_size: Largest number of keys to send to the client to be deleted\n      simultaneously\n\n    Returns: List of tuples of (path, exception) in the same order as the paths\n             argument, where exception is None if the operation succeeded or\n             the relevant exception if the operation failed.\n    '
        if not paths:
            return []
        (buckets, keys) = zip(*[parse_s3_path(path) for path in paths])
        grouped_keys = {bucket: [] for bucket in buckets}
        for (bucket, key) in zip(buckets, keys):
            grouped_keys[bucket].append(key)
        results = {}
        for (bucket, keys) in grouped_keys.items():
            for i in range(0, len(keys), max_batch_size):
                minibatch_keys = keys[i:i + max_batch_size]
                results.update(self._delete_minibatch(bucket, minibatch_keys))
        final_results = [(path, results[parse_s3_path(path)]) for path in paths]
        return final_results

    @retry.with_exponential_backoff(retry_filter=retry.retry_on_server_errors_and_timeout_filter)
    def _delete_minibatch(self, bucket, keys):
        if False:
            i = 10
            return i + 15
        'A helper method. Boto3 allows batch deletions\n    for files within the same bucket.\n\n    Args:\n      bucket: String bucket name\n      keys: List of keys to be deleted in the bucket\n\n    Returns: dict of the form {(bucket, key): error}, where error is None if the\n    operation succeeded\n    '
        request = messages.DeleteBatchRequest(bucket, keys)
        results = {}
        try:
            response = self.client.delete_batch(request)
            for key in response.deleted:
                results[bucket, key] = None
            for (key, error) in zip(response.failed, response.errors):
                results[bucket, key] = error
        except messages.S3ClientError as e:
            for key in keys:
                results[bucket, key] = e
        return results

    def delete_tree(self, root):
        if False:
            for i in range(10):
                print('nop')
        'Deletes all objects under the given S3 directory.\n\n    Args:\n      path: S3 root path in the form s3://<bucket>/<name>/ (ending with a "/")\n\n    Returns: List of tuples of (path, exception), where each path is an object\n            under the given root. exception is None if the operation succeeded\n            or the relevant exception if the operation failed.\n    '
        assert root.endswith('/')
        paths = self.list_prefix(root)
        return self.delete_files(paths)

    def size(self, path):
        if False:
            return 10
        'Returns the size of a single S3 object.\n\n    This method does not perform glob expansion. Hence the given path must be\n    for a single S3 object.\n\n    Returns: size of the S3 object in bytes.\n    '
        return self._s3_object(path).size

    def rename(self, src, dest):
        if False:
            return 10
        'Renames the given S3 object from src to dest.\n\n    Args:\n      src: S3 file path pattern in the form s3://<bucket>/<name>.\n      dest: S3 file path pattern in the form s3://<bucket>/<name>.\n    '
        self.copy(src, dest)
        self.delete(src)

    def last_updated(self, path):
        if False:
            for i in range(10):
                print('nop')
        'Returns the last updated epoch time of a single S3 object.\n\n    This method does not perform glob expansion. Hence the given path must be\n    for a single S3 object.\n\n    Returns: last updated time of the S3 object in second.\n    '
        return self._updated_to_seconds(self._s3_object(path).last_modified)

    def exists(self, path):
        if False:
            print('Hello World!')
        'Returns whether the given S3 object exists.\n\n    Args:\n      path: S3 file path pattern in the form s3://<bucket>/<name>.\n    '
        try:
            self._s3_object(path)
            return True
        except messages.S3ClientError as e:
            if e.code == 404:
                return False
            else:
                raise

    def _status(self, path):
        if False:
            while True:
                i = 10
        'For internal use only; no backwards-compatibility guarantees.\n\n    Returns supported fields (checksum, last_updated, size) of a single object\n    as a dict at once.\n\n    This method does not perform glob expansion. Hence the given path must be\n    for a single S3 object.\n\n    Returns: dict of fields of the S3 object.\n    '
        s3_object = self._s3_object(path)
        file_status = {}
        if hasattr(s3_object, 'etag'):
            file_status['checksum'] = s3_object.etag
        if hasattr(s3_object, 'last_modified'):
            file_status['last_updated'] = self._updated_to_seconds(s3_object.last_modified)
        if hasattr(s3_object, 'size'):
            file_status['size'] = s3_object.size
        return file_status

    @retry.with_exponential_backoff(retry_filter=retry.retry_on_server_errors_and_timeout_filter)
    def _s3_object(self, path):
        if False:
            print('Hello World!')
        'Returns a S3 object metadata for the given path\n\n    This method does not perform glob expansion. Hence the given path must be\n    for a single S3 object.\n\n    Returns: S3 object metadata.\n    '
        (bucket, object) = parse_s3_path(path)
        request = messages.GetRequest(bucket, object)
        return self.client.get_object_metadata(request)

    @staticmethod
    def _updated_to_seconds(updated):
        if False:
            return 10
        'Helper function transform the updated field of response to seconds'
        return time.mktime(updated.timetuple()) - time.timezone + updated.microsecond / 1000000.0

    def rename_files(self, src_dest_pairs):
        if False:
            return 10
        'Renames the given S3 objects from src to dest.\n\n    Args:\n      src_dest_pairs: list of (src, dest) tuples of s3://<bucket>/<name> file\n                      paths to rename from src to dest\n    Returns: List of tuples of (src, dest, exception) in the same order as the\n            src_dest_pairs argument, where exception is None if the operation\n            succeeded or the relevant exception if the operation failed.\n    '
        if not src_dest_pairs:
            return []
        for (src, dest) in src_dest_pairs:
            if src.endswith('/') or dest.endswith('/'):
                raise ValueError('Cannot rename a directory')
        copy_results = self.copy_paths(src_dest_pairs)
        paths_to_delete = [src for (src, _, err) in copy_results if err is None]
        delete_results = self.delete_files(paths_to_delete)
        delete_results_dict = {src: err for (src, err) in delete_results}
        rename_results = []
        for (src, dest, err) in copy_results:
            if err is not None:
                rename_results.append((src, dest, err))
            elif delete_results_dict[src] is not None:
                rename_results.append((src, dest, delete_results_dict[src]))
            else:
                rename_results.append((src, dest, None))
        return rename_results

class S3Downloader(Downloader):

    def __init__(self, client, path, buffer_size):
        if False:
            print('Hello World!')
        self._client = client
        self._path = path
        (self._bucket, self._name) = parse_s3_path(path)
        self._buffer_size = buffer_size
        self._get_request = messages.GetRequest(bucket=self._bucket, object=self._name)
        try:
            metadata = self._get_object_metadata(self._get_request)
        except messages.S3ClientError as e:
            if e.code == 404:
                raise IOError(errno.ENOENT, 'Not found: %s' % self._path)
            else:
                logging.error('HTTP error while requesting file %s: %s', self._path, 3)
                raise
        self._size = metadata.size

    @retry.with_exponential_backoff(retry_filter=retry.retry_on_server_errors_and_timeout_filter)
    def _get_object_metadata(self, get_request):
        if False:
            i = 10
            return i + 15
        return self._client.get_object_metadata(get_request)

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        return self._size

    def get_range(self, start, end):
        if False:
            return 10
        return self._client.get_range(self._get_request, start, end)

class S3Uploader(Uploader):

    def __init__(self, client, path, mime_type='application/octet-stream'):
        if False:
            print('Hello World!')
        self._client = client
        self._path = path
        (self._bucket, self._name) = parse_s3_path(path)
        self._mime_type = mime_type
        self.part_number = 1
        self.buffer = b''
        self.last_error = None
        self.upload_id = None
        self.parts = []
        self._start_upload()

    @retry.no_retries
    def _start_upload(self):
        if False:
            return 10
        try:
            request = messages.UploadRequest(self._bucket, self._name, self._mime_type)
            response = self._client.create_multipart_upload(request)
            self.upload_id = response.upload_id
        except Exception as e:
            logging.error('Error in _start_upload while inserting file %s: %s', self._path, traceback.format_exc())
            self.last_error = e
            raise e

    def put(self, data):
        if False:
            i = 10
            return i + 15
        MIN_WRITE_SIZE = 5 * 1024 * 1024
        MAX_WRITE_SIZE = 5 * 1024 * 1024 * 1024
        self.buffer += data.tobytes()
        while len(self.buffer) >= MIN_WRITE_SIZE:
            chunk = self.buffer[:MAX_WRITE_SIZE]
            self._write_to_s3(chunk)
            self.buffer = self.buffer[MAX_WRITE_SIZE:]

    def _write_to_s3(self, data):
        if False:
            while True:
                i = 10
        try:
            request = messages.UploadPartRequest(self._bucket, self._name, self.upload_id, self.part_number, data)
            response = self._client.upload_part(request)
            self.parts.append({'ETag': response.etag, 'PartNumber': response.part_number})
            self.part_number = self.part_number + 1
        except messages.S3ClientError as e:
            self.last_error = e
            if e.code == 404:
                raise IOError(errno.ENOENT, 'Not found: %s' % self._path)
            else:
                logging.error('HTTP error while requesting file %s: %s', self._path, 3)
                raise

    def finish(self):
        if False:
            print('Hello World!')
        if len(self.buffer) > 0:
            self._write_to_s3(self.buffer)
        if self.last_error is not None:
            raise self.last_error
        request = messages.CompleteMultipartUploadRequest(self._bucket, self._name, self.upload_id, self.parts)
        self._client.complete_multipart_upload(request)