"""Tests for Google Cloud Storage client."""
import datetime
import errno
import io
import logging
import os
import random
import time
import unittest
from email.message import Message
import httplib2
import mock
import apache_beam
from apache_beam.metrics import monitoring_infos
from apache_beam.metrics.execution import MetricsEnvironment
from apache_beam.metrics.metricbase import MetricName
try:
    from apache_beam.io.gcp import gcsio, resource_identifiers
    from apache_beam.io.gcp.internal.clients import storage
    from apitools.base.py.exceptions import HttpError
except ImportError:
    HttpError = None
DEFAULT_GCP_PROJECT = 'apache-beam-testing'
DEFAULT_PROJECT_NUMBER = 1

class FakeGcsClient(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.objects = FakeGcsObjects()
        self.buckets = FakeGcsBuckets()
        self._http = object()

class FakeFile(object):

    def __init__(self, bucket, obj, contents, generation, crc32c=None, last_updated=None):
        if False:
            while True:
                i = 10
        self.bucket = bucket
        self.object = obj
        self.contents = contents
        self.generation = generation
        self.crc32c = crc32c
        self.last_updated = last_updated

    def get_metadata(self):
        if False:
            while True:
                i = 10
        last_updated_datetime = None
        if self.last_updated:
            last_updated_datetime = datetime.datetime.utcfromtimestamp(self.last_updated)
        return storage.Object(bucket=self.bucket, name=self.object, generation=self.generation, size=len(self.contents), crc32c=self.crc32c, updated=last_updated_datetime)

class FakeGcsBuckets(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def get_bucket(self, bucket):
        if False:
            i = 10
            return i + 15
        return storage.Bucket(name=bucket, projectNumber=DEFAULT_PROJECT_NUMBER)

    def Get(self, get_request):
        if False:
            return 10
        return self.get_bucket(get_request.bucket)

class FakeGcsObjects(object):

    def __init__(self):
        if False:
            return 10
        self.files = {}
        self.last_generation = {}
        self.list_page_tokens = {}
        self._fail_when_getting_metadata = []
        self._fail_when_reading = []

    def add_file(self, f, fail_when_getting_metadata=False, fail_when_reading=False):
        if False:
            i = 10
            return i + 15
        self.files[f.bucket, f.object] = f
        self.last_generation[f.bucket, f.object] = f.generation
        if fail_when_getting_metadata:
            self._fail_when_getting_metadata.append(f)
        if fail_when_reading:
            self._fail_when_reading.append(f)

    def get_file(self, bucket, obj):
        if False:
            while True:
                i = 10
        return self.files.get((bucket, obj), None)

    def delete_file(self, bucket, obj):
        if False:
            for i in range(10):
                print('nop')
        del self.files[bucket, obj]

    def get_last_generation(self, bucket, obj):
        if False:
            return 10
        return self.last_generation.get((bucket, obj), 0)

    def Get(self, get_request, download=None):
        if False:
            i = 10
            return i + 15
        f = self.get_file(get_request.bucket, get_request.object)
        if f is None:
            raise HttpError({'status': 404}, None, None)
        if download is None:
            if f in self._fail_when_getting_metadata:
                raise HttpError({'status': 429}, None, None)
            return f.get_metadata()
        else:
            if f in self._fail_when_reading:
                raise HttpError({'status': 429}, None, None)
            stream = download.stream

            def get_range_callback(start, end):
                if False:
                    while True:
                        i = 10
                if not 0 <= start <= end < len(f.contents):
                    raise ValueError('start=%d end=%d len=%s' % (start, end, len(f.contents)))
                stream.write(f.contents[start:end + 1])
            download.GetRange = get_range_callback

    def Insert(self, insert_request, upload=None):
        if False:
            print('Hello World!')
        assert upload is not None
        generation = self.get_last_generation(insert_request.bucket, insert_request.name) + 1
        f = FakeFile(insert_request.bucket, insert_request.name, b'', generation)
        stream = upload.stream
        data_list = []
        while True:
            data = stream.read(1024 * 1024)
            if not data:
                break
            data_list.append(data)
        f.contents = b''.join(data_list)
        self.add_file(f)
    REWRITE_TOKEN = 'test_token'

    def Rewrite(self, rewrite_request):
        if False:
            for i in range(10):
                print('nop')
        if rewrite_request.rewriteToken == self.REWRITE_TOKEN:
            dest_object = storage.Object()
            return storage.RewriteResponse(done=True, objectSize=100, resource=dest_object, totalBytesRewritten=100)
        src_file = self.get_file(rewrite_request.sourceBucket, rewrite_request.sourceObject)
        if not src_file:
            raise HttpError(httplib2.Response({'status': '404'}), '404 Not Found', 'https://fake/url')
        generation = self.get_last_generation(rewrite_request.destinationBucket, rewrite_request.destinationObject) + 1
        dest_file = FakeFile(rewrite_request.destinationBucket, rewrite_request.destinationObject, src_file.contents, generation)
        self.add_file(dest_file)
        time.sleep(10)
        return storage.RewriteResponse(done=False, objectSize=100, rewriteToken=self.REWRITE_TOKEN, totalBytesRewritten=5)

    def Delete(self, delete_request):
        if False:
            return 10
        if self.get_file(delete_request.bucket, delete_request.object):
            self.delete_file(delete_request.bucket, delete_request.object)
        else:
            raise HttpError(httplib2.Response({'status': '404'}), '404 Not Found', 'https://fake/url')

    def List(self, list_request):
        if False:
            print('Hello World!')
        bucket = list_request.bucket
        prefix = list_request.prefix or ''
        matching_files = []
        for (file_bucket, file_name) in sorted(iter(self.files)):
            if bucket == file_bucket and file_name.startswith(prefix):
                file_object = self.files[file_bucket, file_name].get_metadata()
                matching_files.append(file_object)
        items_per_page = 5
        if not list_request.pageToken:
            range_start = 0
        else:
            if list_request.pageToken not in self.list_page_tokens:
                raise ValueError('Invalid page token.')
            range_start = self.list_page_tokens[list_request.pageToken]
            del self.list_page_tokens[list_request.pageToken]
        result = storage.Objects(items=matching_files[range_start:range_start + items_per_page])
        if range_start + items_per_page < len(matching_files):
            next_range_start = range_start + items_per_page
            next_page_token = '_page_token_%s_%s_%d' % (bucket, prefix, next_range_start)
            self.list_page_tokens[next_page_token] = next_range_start
            result.nextPageToken = next_page_token
        return result

class FakeApiCall(object):

    def __init__(self, exception, response):
        if False:
            i = 10
            return i + 15
        self.exception = exception
        self.is_error = exception is not None
        self.response = response

class FakeBatchApiRequest(object):

    def __init__(self, **unused_kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.operations = []

    def Add(self, service, method, request):
        if False:
            return 10
        self.operations.append((service, method, request))

    def Execute(self, unused_http, **unused_kwargs):
        if False:
            for i in range(10):
                print('nop')
        api_calls = []
        for (service, method, request) in self.operations:
            exception = None
            response = None
            try:
                response = getattr(service, method)(request)
            except Exception as e:
                exception = e
            api_calls.append(FakeApiCall(exception, response))
        return api_calls

@unittest.skipIf(HttpError is None, 'GCP dependencies are not installed')
class TestGCSPathParser(unittest.TestCase):
    BAD_GCS_PATHS = ['gs://', 'gs://bucket', 'gs:///name', 'gs:///', 'gs:/blah/bucket/name']

    def test_gcs_path(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(gcsio.parse_gcs_path('gs://bucket/name'), ('bucket', 'name'))
        self.assertEqual(gcsio.parse_gcs_path('gs://bucket/name/sub'), ('bucket', 'name/sub'))

    def test_bad_gcs_path(self):
        if False:
            while True:
                i = 10
        for path in self.BAD_GCS_PATHS:
            self.assertRaises(ValueError, gcsio.parse_gcs_path, path)
        self.assertRaises(ValueError, gcsio.parse_gcs_path, 'gs://bucket/')

    def test_gcs_path_object_optional(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(gcsio.parse_gcs_path('gs://bucket/name', object_optional=True), ('bucket', 'name'))
        self.assertEqual(gcsio.parse_gcs_path('gs://bucket/', object_optional=True), ('bucket', ''))

    def test_bad_gcs_path_object_optional(self):
        if False:
            return 10
        for path in self.BAD_GCS_PATHS:
            self.assertRaises(ValueError, gcsio.parse_gcs_path, path, True)

class SampleOptions(object):

    def __init__(self, project, region, kms_key=None):
        if False:
            print('Hello World!')
        self.project = DEFAULT_GCP_PROJECT
        self.region = region
        self.dataflow_kms_key = kms_key

@unittest.skipIf(HttpError is None, 'GCP dependencies are not installed')
@mock.patch.multiple('time', time=mock.MagicMock(side_effect=range(100)), sleep=mock.MagicMock())
class TestGCSIO(unittest.TestCase):

    def _insert_random_file(self, client, path, size, generation=1, crc32c=None, last_updated=None, fail_when_getting_metadata=False, fail_when_reading=False):
        if False:
            return 10
        (bucket, name) = gcsio.parse_gcs_path(path)
        f = FakeFile(bucket, name, os.urandom(size), generation, crc32c=crc32c, last_updated=last_updated)
        client.objects.add_file(f, fail_when_getting_metadata, fail_when_reading)
        return f

    def setUp(self):
        if False:
            return 10
        self.client = FakeGcsClient()
        self.gcs = gcsio.GcsIO(self.client)

    def test_default_bucket_name(self):
        if False:
            return 10
        self.assertEqual(gcsio.default_gcs_bucket_name(DEFAULT_GCP_PROJECT, 'us-central1'), 'dataflow-staging-us-central1-77b801c0838aee13391c0d1885860494')

    def test_default_bucket_name_failure(self):
        if False:
            return 10
        self.assertEqual(gcsio.get_or_create_default_gcs_bucket(SampleOptions(DEFAULT_GCP_PROJECT, 'us-central1', kms_key='kmskey!')), None)

    def test_num_retries(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(gcsio.GcsIO().client.num_retries, 20)

    def test_retry_func(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsNotNone(gcsio.GcsIO().client.retry_func)

    def test_exists(self):
        if False:
            i = 10
            return i + 15
        file_name = 'gs://gcsio-test/dummy_file'
        file_size = 1234
        self._insert_random_file(self.client, file_name, file_size)
        self.assertFalse(self.gcs.exists(file_name + 'xyz'))
        self.assertTrue(self.gcs.exists(file_name))

    @mock.patch.object(FakeGcsObjects, 'Get')
    def test_exists_failure(self, mock_get):
        if False:
            for i in range(10):
                print('nop')
        mock_get.side_effect = HttpError({'status': 400}, None, None)
        file_name = 'gs://gcsio-test/dummy_file'
        file_size = 1234
        self._insert_random_file(self.client, file_name, file_size)
        with self.assertRaises(HttpError) as cm:
            self.gcs.exists(file_name)
        self.assertEqual(400, cm.exception.status_code)

    def test_checksum(self):
        if False:
            for i in range(10):
                print('nop')
        file_name = 'gs://gcsio-test/dummy_file'
        file_size = 1234
        checksum = 'deadbeef'
        self._insert_random_file(self.client, file_name, file_size, crc32c=checksum)
        self.assertTrue(self.gcs.exists(file_name))
        self.assertEqual(checksum, self.gcs.checksum(file_name))

    def test_size(self):
        if False:
            while True:
                i = 10
        file_name = 'gs://gcsio-test/dummy_file'
        file_size = 1234
        self._insert_random_file(self.client, file_name, file_size)
        self.assertTrue(self.gcs.exists(file_name))
        self.assertEqual(1234, self.gcs.size(file_name))

    def test_last_updated(self):
        if False:
            while True:
                i = 10
        file_name = 'gs://gcsio-test/dummy_file'
        file_size = 1234
        last_updated = 123456.78
        self._insert_random_file(self.client, file_name, file_size, last_updated=last_updated)
        self.assertTrue(self.gcs.exists(file_name))
        self.assertEqual(last_updated, self.gcs.last_updated(file_name))

    def test_file_status(self):
        if False:
            for i in range(10):
                print('nop')
        file_name = 'gs://gcsio-test/dummy_file'
        file_size = 1234
        last_updated = 123456.78
        checksum = 'deadbeef'
        self._insert_random_file(self.client, file_name, file_size, last_updated=last_updated, crc32c=checksum)
        file_checksum = self.gcs.checksum(file_name)
        file_status = self.gcs._status(file_name)
        self.assertEqual(file_status['size'], file_size)
        self.assertEqual(file_status['checksum'], file_checksum)
        self.assertEqual(file_status['last_updated'], last_updated)

    def test_file_mode(self):
        if False:
            while True:
                i = 10
        file_name = 'gs://gcsio-test/dummy_mode_file'
        with self.gcs.open(file_name, 'wb') as f:
            assert f.mode == 'wb'
        with self.gcs.open(file_name, 'rb') as f:
            assert f.mode == 'rb'

    def test_bad_file_modes(self):
        if False:
            for i in range(10):
                print('nop')
        file_name = 'gs://gcsio-test/dummy_mode_file'
        with self.assertRaises(ValueError):
            self.gcs.open(file_name, 'w+')
        with self.assertRaises(ValueError):
            self.gcs.open(file_name, 'r+b')

    def test_empty_batches(self):
        if False:
            return 10
        self.assertEqual([], self.gcs.copy_batch([]))
        self.assertEqual([], self.gcs.delete_batch([]))

    def test_delete(self):
        if False:
            i = 10
            return i + 15
        file_name = 'gs://gcsio-test/delete_me'
        file_size = 1024
        self.gcs.delete(file_name)
        self._insert_random_file(self.client, file_name, file_size)
        self.assertTrue(gcsio.parse_gcs_path(file_name) in self.client.objects.files)
        self.gcs.delete(file_name)
        self.assertFalse(gcsio.parse_gcs_path(file_name) in self.client.objects.files)

    @mock.patch('apache_beam.io.gcp.gcsio.auth.get_service_credentials', wraps=lambda pipeline_options: None)
    @mock.patch('apache_beam.io.gcp.gcsio.get_new_http')
    def test_user_agent_passed(self, get_new_http_mock, get_service_creds_mock):
        if False:
            print('Hello World!')
        client = gcsio.GcsIO()
        try:
            client.get_bucket('mabucket')
        except:
            pass
        call = get_new_http_mock.return_value.request.mock_calls[-2]
        self.assertIn('apache-beam/%s (GPN:Beam)' % apache_beam.__version__, call[2]['headers']['User-Agent'])

    @mock.patch('apache_beam.io.gcp.gcsio.BatchApiRequest')
    def test_delete_batch(self, *unused_args):
        if False:
            for i in range(10):
                print('nop')
        gcsio.BatchApiRequest = FakeBatchApiRequest
        file_name_pattern = 'gs://gcsio-test/delete_me_%d'
        file_size = 1024
        num_files = 10
        result = self.gcs.delete_batch([file_name_pattern % i for i in range(num_files)])
        self.assertTrue(result)
        for (i, (file_name, exception)) in enumerate(result):
            self.assertEqual(file_name, file_name_pattern % i)
            self.assertEqual(exception, None)
            self.assertFalse(self.gcs.exists(file_name_pattern % i))
        for i in range(num_files):
            self._insert_random_file(self.client, file_name_pattern % i, file_size)
        for i in range(num_files):
            self.assertTrue(self.gcs.exists(file_name_pattern % i))
        self.gcs.delete_batch([file_name_pattern % i for i in range(num_files)])
        for i in range(num_files):
            self.assertFalse(self.gcs.exists(file_name_pattern % i))

    def test_copy(self):
        if False:
            while True:
                i = 10
        src_file_name = 'gs://gcsio-test/source'
        dest_file_name = 'gs://gcsio-test/dest'
        file_size = 1024
        self._insert_random_file(self.client, src_file_name, file_size)
        self.assertTrue(gcsio.parse_gcs_path(src_file_name) in self.client.objects.files)
        self.assertFalse(gcsio.parse_gcs_path(dest_file_name) in self.client.objects.files)
        self.gcs.copy(src_file_name, dest_file_name, dest_kms_key_name='kms_key')
        self.assertTrue(gcsio.parse_gcs_path(src_file_name) in self.client.objects.files)
        self.assertTrue(gcsio.parse_gcs_path(dest_file_name) in self.client.objects.files)
        with self.assertRaisesRegex(HttpError, 'Not Found'):
            self.gcs.copy('gs://gcsio-test/non-existent', 'gs://gcsio-test/non-existent-destination')

    @mock.patch('apache_beam.io.gcp.gcsio.BatchApiRequest')
    def test_copy_batch(self, *unused_args):
        if False:
            for i in range(10):
                print('nop')
        gcsio.BatchApiRequest = FakeBatchApiRequest
        from_name_pattern = 'gs://gcsio-test/copy_me_%d'
        to_name_pattern = 'gs://gcsio-test/destination_%d'
        file_size = 1024
        num_files = 10
        result = self.gcs.copy_batch([(from_name_pattern % i, to_name_pattern % i) for i in range(num_files)], dest_kms_key_name='kms_key')
        self.assertTrue(result)
        for (i, (src, dest, exception)) in enumerate(result):
            self.assertEqual(src, from_name_pattern % i)
            self.assertEqual(dest, to_name_pattern % i)
            self.assertTrue(isinstance(exception, IOError))
            self.assertEqual(exception.errno, errno.ENOENT)
            self.assertFalse(self.gcs.exists(from_name_pattern % i))
            self.assertFalse(self.gcs.exists(to_name_pattern % i))
        for i in range(num_files):
            self._insert_random_file(self.client, from_name_pattern % i, file_size)
        for i in range(num_files):
            self.assertTrue(self.gcs.exists(from_name_pattern % i))
        self.gcs.copy_batch([(from_name_pattern % i, to_name_pattern % i) for i in range(num_files)])
        for i in range(num_files):
            self.assertTrue(self.gcs.exists(from_name_pattern % i))
            self.assertTrue(self.gcs.exists(to_name_pattern % i))

    def test_copytree(self):
        if False:
            i = 10
            return i + 15
        src_dir_name = 'gs://gcsio-test/source/'
        dest_dir_name = 'gs://gcsio-test/dest/'
        file_size = 1024
        paths = ['a', 'b/c', 'b/d']
        for path in paths:
            src_file_name = src_dir_name + path
            dest_file_name = dest_dir_name + path
            self._insert_random_file(self.client, src_file_name, file_size)
            self.assertTrue(gcsio.parse_gcs_path(src_file_name) in self.client.objects.files)
            self.assertFalse(gcsio.parse_gcs_path(dest_file_name) in self.client.objects.files)
        self.gcs.copytree(src_dir_name, dest_dir_name)
        for path in paths:
            src_file_name = src_dir_name + path
            dest_file_name = dest_dir_name + path
            self.assertTrue(gcsio.parse_gcs_path(src_file_name) in self.client.objects.files)
            self.assertTrue(gcsio.parse_gcs_path(dest_file_name) in self.client.objects.files)

    def test_rename(self):
        if False:
            return 10
        src_file_name = 'gs://gcsio-test/source'
        dest_file_name = 'gs://gcsio-test/dest'
        file_size = 1024
        self._insert_random_file(self.client, src_file_name, file_size)
        self.assertTrue(gcsio.parse_gcs_path(src_file_name) in self.client.objects.files)
        self.assertFalse(gcsio.parse_gcs_path(dest_file_name) in self.client.objects.files)
        self.gcs.rename(src_file_name, dest_file_name)
        self.assertFalse(gcsio.parse_gcs_path(src_file_name) in self.client.objects.files)
        self.assertTrue(gcsio.parse_gcs_path(dest_file_name) in self.client.objects.files)

    def test_full_file_read(self):
        if False:
            print('Hello World!')
        file_name = 'gs://gcsio-test/full_file'
        file_size = 5 * 1024 * 1024 + 100
        random_file = self._insert_random_file(self.client, file_name, file_size)
        f = self.gcs.open(file_name)
        self.assertEqual(f.mode, 'r')
        f.seek(0, os.SEEK_END)
        self.assertEqual(f.tell(), file_size)
        self.assertEqual(f.read(), b'')
        f.seek(0)
        self.assertEqual(f.read(), random_file.contents)

    def test_file_random_seek(self):
        if False:
            return 10
        file_name = 'gs://gcsio-test/seek_file'
        file_size = 5 * 1024 * 1024 - 100
        random_file = self._insert_random_file(self.client, file_name, file_size)
        f = self.gcs.open(file_name)
        random.seed(0)
        for _ in range(0, 10):
            a = random.randint(0, file_size - 1)
            b = random.randint(0, file_size - 1)
            (start, end) = (min(a, b), max(a, b))
            f.seek(start)
            self.assertEqual(f.tell(), start)
            self.assertEqual(f.read(end - start + 1), random_file.contents[start:end + 1])
            self.assertEqual(f.tell(), end + 1)

    def test_file_iterator(self):
        if False:
            i = 10
            return i + 15
        file_name = 'gs://gcsio-test/iterating_file'
        lines = []
        line_count = 10
        for _ in range(line_count):
            line_length = random.randint(100, 500)
            line = os.urandom(line_length).replace(b'\n', b' ') + b'\n'
            lines.append(line)
        contents = b''.join(lines)
        (bucket, name) = gcsio.parse_gcs_path(file_name)
        self.client.objects.add_file(FakeFile(bucket, name, contents, 1))
        f = self.gcs.open(file_name)
        read_lines = 0
        for line in f:
            read_lines += 1
        self.assertEqual(read_lines, line_count)

    def test_file_read_line(self):
        if False:
            return 10
        file_name = 'gs://gcsio-test/read_line_file'
        lines = []
        read_buffer_size = 1024
        lines.append(b'x' * 1023 + b'\n')
        for _ in range(1, 1000):
            line_length = random.randint(100, 500)
            line = os.urandom(line_length).replace(b'\n', b' ') + b'\n'
            lines.append(line)
        contents = b''.join(lines)
        file_size = len(contents)
        (bucket, name) = gcsio.parse_gcs_path(file_name)
        self.client.objects.add_file(FakeFile(bucket, name, contents, 1))
        f = self.gcs.open(file_name, read_buffer_size=read_buffer_size)
        f.seek(0)
        self.assertEqual(f.readline(), lines[0])
        self.assertEqual(f.tell(), len(lines[0]))
        self.assertEqual(f.readline(), lines[1])
        f.seek(file_size - len(lines[-1]) - 1)
        self.assertEqual(f.readline(), b'\n')
        f.seek(file_size)
        self.assertEqual(f.readline(), b'')
        random.seed(0)
        for _ in range(0, 10):
            start = random.randint(0, file_size - 1)
            line_index = 0
            chars_left = start
            while True:
                next_line_length = len(lines[line_index])
                if chars_left - next_line_length < 0:
                    break
                chars_left -= next_line_length
                line_index += 1
            f.seek(start)
            self.assertEqual(f.readline(), lines[line_index][chars_left:])

    def test_file_write(self):
        if False:
            while True:
                i = 10
        file_name = 'gs://gcsio-test/write_file'
        file_size = 5 * 1024 * 1024 + 2000
        contents = os.urandom(file_size)
        f = self.gcs.open(file_name, 'w')
        self.assertEqual(f.mode, 'w')
        f.write(contents[0:1000])
        f.write(contents[1000:1024 * 1024])
        f.write(contents[1024 * 1024:])
        f.close()
        (bucket, name) = gcsio.parse_gcs_path(file_name)
        self.assertEqual(self.client.objects.get_file(bucket, name).contents, contents)

    def test_file_close(self):
        if False:
            while True:
                i = 10
        file_name = 'gs://gcsio-test/close_file'
        file_size = 5 * 1024 * 1024 + 2000
        contents = os.urandom(file_size)
        f = self.gcs.open(file_name, 'w')
        self.assertEqual(f.mode, 'w')
        f.write(contents)
        f.close()
        f.close()
        (bucket, name) = gcsio.parse_gcs_path(file_name)
        self.assertEqual(self.client.objects.get_file(bucket, name).contents, contents)

    def test_file_flush(self):
        if False:
            print('Hello World!')
        file_name = 'gs://gcsio-test/flush_file'
        file_size = 5 * 1024 * 1024 + 2000
        contents = os.urandom(file_size)
        (bucket, name) = gcsio.parse_gcs_path(file_name)
        f = self.gcs.open(file_name, 'w')
        self.assertEqual(f.mode, 'w')
        f.write(contents[0:1000])
        f.flush()
        f.write(contents[1000:1024 * 1024])
        f.flush()
        f.flush()
        f.write(contents[1024 * 1024:])
        f.close()
        self.assertEqual(self.client.objects.get_file(bucket, name).contents, contents)

    def test_context_manager(self):
        if False:
            print('Hello World!')
        file_name = 'gs://gcsio-test/context_manager_file'
        file_size = 1024
        contents = os.urandom(file_size)
        with self.gcs.open(file_name, 'w') as f:
            f.write(contents)
        (bucket, name) = gcsio.parse_gcs_path(file_name)
        self.assertEqual(self.client.objects.get_file(bucket, name).contents, contents)
        with self.gcs.open(file_name) as f:
            self.assertEqual(f.read(), contents)
        with self.assertRaises(ZeroDivisionError):
            with self.gcs.open(file_name) as f:
                f.read(0 // 0)

    def test_list_prefix(self):
        if False:
            return 10
        bucket_name = 'gcsio-test'
        objects = [('cow/cat/fish', 2), ('cow/cat/blubber', 3), ('cow/dog/blubber', 4)]
        for (object_name, size) in objects:
            file_name = 'gs://%s/%s' % (bucket_name, object_name)
            self._insert_random_file(self.client, file_name, size)
        test_cases = [('gs://gcsio-test/c', [('cow/cat/fish', 2), ('cow/cat/blubber', 3), ('cow/dog/blubber', 4)]), ('gs://gcsio-test/cow/', [('cow/cat/fish', 2), ('cow/cat/blubber', 3), ('cow/dog/blubber', 4)]), ('gs://gcsio-test/cow/cat/fish', [('cow/cat/fish', 2)])]
        for (file_pattern, expected_object_names) in test_cases:
            expected_file_names = [('gs://%s/%s' % (bucket_name, object_name), size) for (object_name, size) in expected_object_names]
            self.assertEqual(set(self.gcs.list_prefix(file_pattern).items()), set(expected_file_names))

    def test_mime_binary_encoding(self):
        if False:
            i = 10
            return i + 15
        from apitools.base.py.transfer import email_generator
        generator_cls = email_generator.BytesGenerator
        output_buffer = io.BytesIO()
        generator = generator_cls(output_buffer)
        test_msg = 'a\nb\r\nc\n\r\n\n\nd'
        message = Message()
        message.set_payload(test_msg)
        generator._handle_text(message)
        self.assertEqual(test_msg.encode('ascii'), output_buffer.getvalue())

    def test_downloader_monitoring_info(self):
        if False:
            while True:
                i = 10
        MetricsEnvironment.process_wide_container().reset()
        file_name = 'gs://gcsio-metrics-test/dummy_mode_file'
        file_size = 5 * 1024 * 1024 + 100
        random_file = self._insert_random_file(self.client, file_name, file_size)
        self.gcs.open(file_name, 'r')
        resource = resource_identifiers.GoogleCloudStorageBucket(random_file.bucket)
        labels = {monitoring_infos.SERVICE_LABEL: 'Storage', monitoring_infos.METHOD_LABEL: 'Objects.get', monitoring_infos.RESOURCE_LABEL: resource, monitoring_infos.GCS_BUCKET_LABEL: random_file.bucket, monitoring_infos.GCS_PROJECT_ID_LABEL: str(DEFAULT_PROJECT_NUMBER), monitoring_infos.STATUS_LABEL: 'ok'}
        metric_name = MetricName(None, None, urn=monitoring_infos.API_REQUEST_COUNT_URN, labels=labels)
        metric_value = MetricsEnvironment.process_wide_container().get_counter(metric_name).get_cumulative()
        self.assertEqual(metric_value, 2)

    @mock.patch.object(FakeGcsBuckets, 'Get')
    def test_downloader_fail_to_get_project_number(self, mock_get):
        if False:
            return 10
        mock_get.side_effect = HttpError({'status': 403}, None, None)
        MetricsEnvironment.process_wide_container().reset()
        file_name = 'gs://gcsio-metrics-test/dummy_mode_file'
        file_size = 5 * 1024 * 1024 + 100
        random_file = self._insert_random_file(self.client, file_name, file_size)
        self.gcs.open(file_name, 'r')
        resource = resource_identifiers.GoogleCloudStorageBucket(random_file.bucket)
        labels = {monitoring_infos.SERVICE_LABEL: 'Storage', monitoring_infos.METHOD_LABEL: 'Objects.get', monitoring_infos.RESOURCE_LABEL: resource, monitoring_infos.GCS_BUCKET_LABEL: random_file.bucket, monitoring_infos.GCS_PROJECT_ID_LABEL: str(DEFAULT_PROJECT_NUMBER), monitoring_infos.STATUS_LABEL: 'ok'}
        metric_name = MetricName(None, None, urn=monitoring_infos.API_REQUEST_COUNT_URN, labels=labels)
        metric_value = MetricsEnvironment.process_wide_container().get_counter(metric_name).get_cumulative()
        self.assertEqual(metric_value, 0)
        labels_without_project_id = {monitoring_infos.SERVICE_LABEL: 'Storage', monitoring_infos.METHOD_LABEL: 'Objects.get', monitoring_infos.RESOURCE_LABEL: resource, monitoring_infos.GCS_BUCKET_LABEL: random_file.bucket, monitoring_infos.STATUS_LABEL: 'ok'}
        metric_name = MetricName(None, None, urn=monitoring_infos.API_REQUEST_COUNT_URN, labels=labels_without_project_id)
        metric_value = MetricsEnvironment.process_wide_container().get_counter(metric_name).get_cumulative()
        self.assertEqual(metric_value, 2)

    def test_downloader_fail_non_existent_object(self):
        if False:
            print('Hello World!')
        file_name = 'gs://gcsio-metrics-test/dummy_mode_file'
        with self.assertRaises(IOError):
            self.gcs.open(file_name, 'r')

    def test_downloader_fail_when_getting_metadata(self):
        if False:
            return 10
        file_name = 'gs://gcsio-metrics-test/dummy_mode_file'
        file_size = 5 * 1024 * 1024 + 100
        self._insert_random_file(self.client, file_name, file_size, fail_when_getting_metadata=True)
        with self.assertRaises(HttpError):
            self.gcs.open(file_name, 'r')

    def test_downloader_fail_when_reading(self):
        if False:
            for i in range(10):
                print('nop')
        file_name = 'gs://gcsio-metrics-test/dummy_mode_file'
        file_size = 5 * 1024 * 1024 + 100
        self._insert_random_file(self.client, file_name, file_size, fail_when_reading=True)
        with self.assertRaises(HttpError):
            self.gcs.open(file_name, 'r')

    def test_uploader_monitoring_info(self):
        if False:
            i = 10
            return i + 15
        MetricsEnvironment.process_wide_container().reset()
        file_name = 'gs://gcsio-metrics-test/dummy_mode_file'
        file_size = 5 * 1024 * 1024 + 100
        random_file = self._insert_random_file(self.client, file_name, file_size)
        f = self.gcs.open(file_name, 'w')
        resource = resource_identifiers.GoogleCloudStorageBucket(random_file.bucket)
        labels = {monitoring_infos.SERVICE_LABEL: 'Storage', monitoring_infos.METHOD_LABEL: 'Objects.insert', monitoring_infos.RESOURCE_LABEL: resource, monitoring_infos.GCS_BUCKET_LABEL: random_file.bucket, monitoring_infos.GCS_PROJECT_ID_LABEL: str(DEFAULT_PROJECT_NUMBER), monitoring_infos.STATUS_LABEL: 'ok'}
        f.close()
        metric_name = MetricName(None, None, urn=monitoring_infos.API_REQUEST_COUNT_URN, labels=labels)
        metric_value = MetricsEnvironment.process_wide_container().get_counter(metric_name).get_cumulative()
        self.assertEqual(metric_value, 1)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()