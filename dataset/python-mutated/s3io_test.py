"""Tests for S3 client."""
import logging
import os
import random
import time
import unittest
from apache_beam.io.aws import s3io
from apache_beam.io.aws.clients.s3 import fake_client
from apache_beam.io.aws.clients.s3 import messages
from apache_beam.options import pipeline_options

class TestS3PathParser(unittest.TestCase):
    BAD_S3_PATHS = ['s3://', 's3://bucket', 's3:///name', 's3:///', 's3:/blah/bucket/name']

    def test_s3_path(self):
        if False:
            while True:
                i = 10
        self.assertEqual(s3io.parse_s3_path('s3://bucket/name'), ('bucket', 'name'))
        self.assertEqual(s3io.parse_s3_path('s3://bucket/name/sub'), ('bucket', 'name/sub'))

    def test_bad_s3_path(self):
        if False:
            return 10
        for path in self.BAD_S3_PATHS:
            self.assertRaises(ValueError, s3io.parse_s3_path, path)
        self.assertRaises(ValueError, s3io.parse_s3_path, 's3://bucket/')

    def test_s3_path_object_optional(self):
        if False:
            return 10
        self.assertEqual(s3io.parse_s3_path('s3://bucket/name', object_optional=True), ('bucket', 'name'))
        self.assertEqual(s3io.parse_s3_path('s3://bucket/', object_optional=True), ('bucket', ''))

    def test_bad_s3_path_object_optional(self):
        if False:
            return 10
        for path in self.BAD_S3_PATHS:
            self.assertRaises(ValueError, s3io.parse_s3_path, path, True)

class TestS3IO(unittest.TestCase):

    def _insert_random_file(self, client, path, size):
        if False:
            i = 10
            return i + 15
        (bucket, name) = s3io.parse_s3_path(path)
        contents = os.urandom(size)
        fakeFile = fake_client.FakeFile(bucket, name, contents)
        if self.USE_MOCK:
            self.client.add_file(fakeFile)
        else:
            f = self.aws.open(path, 'w')
            f.write(contents)
            f.close()
        return fakeFile

    def setUp(self):
        if False:
            while True:
                i = 10
        self.USE_MOCK = True
        self.TEST_DATA_PATH = 's3://random-data-sets/beam_tests/'
        if self.USE_MOCK:
            self.client = fake_client.FakeS3Client()
            (test_data_bucket, _) = s3io.parse_s3_path(self.TEST_DATA_PATH)
            self.client.known_buckets.add(test_data_bucket)
            self.aws = s3io.S3IO(self.client)
        else:
            self.aws = s3io.S3IO(options=pipeline_options.S3Options())
            self.client = self.aws.client

    def test_size(self):
        if False:
            return 10
        file_name = self.TEST_DATA_PATH + 'dummy_file'
        file_size = 1234
        self._insert_random_file(self.client, file_name, file_size)
        self.assertTrue(self.aws.exists(file_name))
        self.assertEqual(1234, self.aws.size(file_name))
        self.aws.delete(file_name)

    def test_last_updated(self):
        if False:
            return 10
        file_name = self.TEST_DATA_PATH + 'dummy_file'
        file_size = 1234
        self._insert_random_file(self.client, file_name, file_size)
        self.assertTrue(self.aws.exists(file_name))
        tolerance = 5 * 60
        result = self.aws.last_updated(file_name)
        self.assertAlmostEqual(result, time.time(), delta=tolerance)
        self.aws.delete(file_name)

    def test_checksum(self):
        if False:
            return 10
        file_name = self.TEST_DATA_PATH + 'checksum'
        file_size = 1024
        file_ = self._insert_random_file(self.client, file_name, file_size)
        original_etag = self.aws.checksum(file_name)
        self.aws.delete(file_name)
        with self.aws.open(file_name, 'w') as f:
            f.write(file_.contents)
        rewritten_etag = self.aws.checksum(file_name)
        self.assertEqual(original_etag, rewritten_etag)
        self.assertEqual(len(original_etag), 36)
        self.assertTrue(original_etag.endswith('-1"'))
        self.aws.delete(file_name)

    def test_file_status(self):
        if False:
            print('Hello World!')
        file_name = self.TEST_DATA_PATH + 'metadata'
        file_size = 1024
        self._insert_random_file(self.client, file_name, file_size)
        file_checksum = self.aws.checksum(file_name)
        file_timestamp = self.aws.last_updated(file_name)
        file_status = self.aws._status(file_name)
        self.assertEqual(file_status['size'], file_size)
        self.assertEqual(file_status['checksum'], file_checksum)
        self.assertEqual(file_status['last_updated'], file_timestamp)
        self.aws.delete(file_name)

    def test_copy(self):
        if False:
            print('Hello World!')
        src_file_name = self.TEST_DATA_PATH + 'source'
        dest_file_name = self.TEST_DATA_PATH + 'dest'
        file_size = 1024
        self._insert_random_file(self.client, src_file_name, file_size)
        self.assertTrue(src_file_name in self.aws.list_prefix(self.TEST_DATA_PATH))
        self.assertFalse(dest_file_name in self.aws.list_prefix(self.TEST_DATA_PATH))
        self.aws.copy(src_file_name, dest_file_name)
        self.assertTrue(src_file_name in self.aws.list_prefix(self.TEST_DATA_PATH))
        self.assertTrue(dest_file_name in self.aws.list_prefix(self.TEST_DATA_PATH))
        self.aws.delete_files([src_file_name, dest_file_name])
        with self.assertRaises(messages.S3ClientError) as err:
            self.aws.copy(self.TEST_DATA_PATH + 'non-existent', self.TEST_DATA_PATH + 'non-existent-destination')
        self.assertTrue('Not Found' in err.exception.message)

    def test_copy_paths(self):
        if False:
            return 10
        from_name_pattern = self.TEST_DATA_PATH + 'copy_me_%d'
        to_name_pattern = self.TEST_DATA_PATH + 'destination_%d'
        file_size = 1024
        num_files = 10
        src_dest_pairs = [(from_name_pattern % i, to_name_pattern % i) for i in range(num_files)]
        result = self.aws.copy_paths(src_dest_pairs)
        self.assertTrue(result)
        for (i, (src, dest, exception)) in enumerate(result):
            self.assertEqual(src, from_name_pattern % i)
            self.assertEqual(dest, to_name_pattern % i)
            self.assertTrue(isinstance(exception, messages.S3ClientError))
            self.assertEqual(exception.code, 404)
            self.assertFalse(self.aws.exists(from_name_pattern % i))
            self.assertFalse(self.aws.exists(to_name_pattern % i))
        for i in range(num_files):
            self._insert_random_file(self.client, from_name_pattern % i, file_size)
        for i in range(num_files):
            self.assertTrue(self.aws.exists(from_name_pattern % i))
        result = self.aws.copy_paths(src_dest_pairs)
        for i in range(num_files):
            self.assertTrue(self.aws.exists(from_name_pattern % i))
            self.assertTrue(self.aws.exists(to_name_pattern % i))
        for (i, (src, dest, exception)) in enumerate(result):
            self.assertEqual(src_dest_pairs[i], (src, dest))
            self.assertEqual(exception, None)
        all_files = set().union(*[set(pair) for pair in src_dest_pairs])
        self.aws.delete_files(all_files)

    def test_copy_paths_error(self):
        if False:
            for i in range(10):
                print('nop')
        n_real_files = 3
        from_path = self.TEST_DATA_PATH + 'copy_paths/'
        files = [from_path + '%d' % i for i in range(n_real_files)]
        to_path = self.TEST_DATA_PATH + 'destination/'
        destinations = [to_path + '%d' % i for i in range(n_real_files)]
        for file_ in files:
            self._insert_random_file(self.client, file_, 1024)
        sources = files + [from_path + 'X', from_path + 'fake_directory_1/', from_path + 'fake_directory_2/']
        destinations += [to_path + 'X', to_path + 'fake_directory_1/', to_path + 'fake_directory_2']
        result = self.aws.copy_paths(list(zip(sources, destinations)))
        self.assertEqual(len(result), len(sources) - 1)
        for (_, _, err) in result[:n_real_files]:
            self.assertTrue(err is None)
        for (_, _, err) in result[n_real_files:]:
            self.assertIsInstance(err, messages.S3ClientError)
        self.assertEqual(result[-2][2].code, 404)
        self.assertEqual(result[-1][2].code, 400)
        self.aws.delete_files(files)
        self.aws.delete_files(destinations)

    def test_copy_tree(self):
        if False:
            print('Hello World!')
        src_dir_name = self.TEST_DATA_PATH + 'source/'
        dest_dir_name = self.TEST_DATA_PATH + 'dest/'
        file_size = 1024
        paths = ['a', 'b/c', 'b/d']
        for path in paths:
            src_file_name = src_dir_name + path
            dest_file_name = dest_dir_name + path
            self._insert_random_file(self.client, src_file_name, file_size)
            self.assertTrue(src_file_name in self.aws.list_prefix(self.TEST_DATA_PATH))
            self.assertFalse(dest_file_name in self.aws.list_prefix(self.TEST_DATA_PATH))
        results = self.aws.copy_tree(src_dir_name, dest_dir_name)
        for (src_file_name, dest_file_name, err) in results:
            self.assertTrue(src_dir_name in src_file_name)
            self.assertTrue(dest_dir_name in dest_file_name)
            self.assertIsNone(err)
            self.assertTrue(src_file_name in self.aws.list_prefix(self.TEST_DATA_PATH))
            self.assertTrue(dest_file_name in self.aws.list_prefix(self.TEST_DATA_PATH))
        for path in paths:
            src_file_name = src_dir_name + path
            dest_file_name = dest_dir_name + path
            self.aws.delete_files([src_file_name, dest_file_name])

    def test_rename(self):
        if False:
            for i in range(10):
                print('nop')
        src_file_name = self.TEST_DATA_PATH + 'source'
        dest_file_name = self.TEST_DATA_PATH + 'dest'
        file_size = 1024
        self._insert_random_file(self.client, src_file_name, file_size)
        self.assertTrue(src_file_name in self.aws.list_prefix(self.TEST_DATA_PATH))
        self.assertFalse(dest_file_name in self.aws.list_prefix(self.TEST_DATA_PATH))
        self.aws.rename(src_file_name, dest_file_name)
        self.assertFalse(src_file_name in self.aws.list_prefix(self.TEST_DATA_PATH))
        self.assertTrue(dest_file_name in self.aws.list_prefix(self.TEST_DATA_PATH))
        self.aws.delete_files([src_file_name, dest_file_name])

    def test_rename_files(self):
        if False:
            i = 10
            return i + 15
        from_name_pattern = self.TEST_DATA_PATH + 'to_rename_%d'
        to_name_pattern = self.TEST_DATA_PATH + 'been_renamed_%d'
        file_size = 1024
        num_files = 10
        src_dest_pairs = [(from_name_pattern % i, to_name_pattern % i) for i in range(num_files)]
        result = self.aws.rename_files(src_dest_pairs)
        self.assertTrue(result)
        for (i, (src, dest, exception)) in enumerate(result):
            self.assertEqual(src, from_name_pattern % i)
            self.assertEqual(dest, to_name_pattern % i)
            self.assertTrue(isinstance(exception, messages.S3ClientError))
            self.assertEqual(exception.code, 404)
            self.assertFalse(self.aws.exists(from_name_pattern % i))
            self.assertFalse(self.aws.exists(to_name_pattern % i))
        for i in range(num_files):
            self._insert_random_file(self.client, from_name_pattern % i, file_size)
        for i in range(num_files):
            self.assertTrue(self.aws.exists(from_name_pattern % i))
            self.assertFalse(self.aws.exists(to_name_pattern % i))
        self.aws.rename_files(src_dest_pairs)
        for i in range(num_files):
            self.assertFalse(self.aws.exists(from_name_pattern % i))
            self.assertTrue(self.aws.exists(to_name_pattern % i))
        all_files = set().union(*[set(pair) for pair in src_dest_pairs])
        self.aws.delete_files(all_files)

    def test_rename_files_with_errors(self):
        if False:
            i = 10
            return i + 15
        real_prefix = self.TEST_DATA_PATH + 'rename_batch_%s'
        fake_prefix = 's3://fake-bucket-68ae4b0ef7b9/rename_batch_%s'
        src_dest_pairs = [(prefix % 'src', prefix % 'dest') for prefix in (real_prefix, fake_prefix)]
        self._insert_random_file(self.client, real_prefix % 'src', 1024)
        result = self.aws.rename_files(src_dest_pairs)
        self.assertEqual(result[0][0], src_dest_pairs[0][0])
        self.assertEqual(result[0][1], src_dest_pairs[0][1])
        self.assertIsNone(result[0][2])
        self.assertEqual(result[1][0], src_dest_pairs[1][0])
        self.assertEqual(result[1][1], src_dest_pairs[1][1])
        self.assertEqual(result[1][2].code, 404)
        self.aws.delete(real_prefix % 'dest')

    def test_rename_files_with_errors_directory(self):
        if False:
            return 10
        dir_name = self.TEST_DATA_PATH + 'rename_dir/'
        file_name = dir_name + 'file'
        self._insert_random_file(self.client, file_name, 1024)
        self.assertTrue(self.aws.exists(file_name))
        with self.assertRaises(ValueError):
            self.aws.rename_files([(file_name, self.TEST_DATA_PATH + 'dir_dest/')])
        self.aws.delete(file_name)

    def test_delete_paths(self):
        if False:
            print('Hello World!')
        prefix = self.TEST_DATA_PATH + 'delete_paths/'
        file_names = [prefix + 'a', prefix + 'b/c']
        for file_name in file_names:
            self._insert_random_file(self.client, file_name, 1024)
        self.assertTrue(self.aws.exists(file_names[0]))
        self.assertTrue(self.aws.exists(file_names[1]))
        paths = [prefix + 'a', prefix + 'b/']
        self.aws.delete_paths(paths)
        self.assertFalse(self.aws.exists(file_names[0]))
        self.assertFalse(self.aws.exists(file_names[1]))

    def test_delete(self):
        if False:
            i = 10
            return i + 15
        file_name = self.TEST_DATA_PATH + 'delete_file'
        file_size = 1024
        self.aws.delete(file_name)
        self._insert_random_file(self.aws.client, file_name, file_size)
        files = self.aws.list_prefix(self.TEST_DATA_PATH)
        self.assertTrue(file_name in files)
        self.aws.delete(file_name)
        self.assertFalse(self.aws.exists(file_name))

    def test_delete_files(self, *unused_args):
        if False:
            return 10
        file_name_pattern = self.TEST_DATA_PATH + 'delete_batch/%d'
        file_size = 1024
        num_files = 5
        result = self.aws.delete_files([file_name_pattern % i for i in range(num_files)])
        self.assertTrue(result)
        for (i, (file_name, exception)) in enumerate(result):
            self.assertEqual(file_name, file_name_pattern % i)
            self.assertEqual(exception, None)
            self.assertFalse(self.aws.exists(file_name_pattern % i))
        for i in range(num_files):
            self._insert_random_file(self.client, file_name_pattern % i, file_size)
        for i in range(num_files):
            self.assertTrue(self.aws.exists(file_name_pattern % i))
        self.aws.delete_files([file_name_pattern % i for i in range(num_files)])
        for i in range(num_files):
            self.assertFalse(self.aws.exists(file_name_pattern % i))

    def test_delete_files_with_errors(self, *unused_args):
        if False:
            print('Hello World!')
        real_file = self.TEST_DATA_PATH + 'delete_batch/file'
        fake_file = 's3://fake-bucket-68ae4b0ef7b9/delete_batch/file'
        filenames = [real_file, fake_file]
        result = self.aws.delete_files(filenames)
        self.assertEqual(result[0][0], filenames[0])
        self.assertIsNone(result[0][1])
        self.assertEqual(result[1][0], filenames[1])
        self.assertEqual(result[1][1].code, 404)

    def test_delete_tree(self):
        if False:
            while True:
                i = 10
        root_path = self.TEST_DATA_PATH + 'delete_tree/'
        leaf_paths = ['a', 'b/c', 'b/d', 'b/d/e']
        paths = [root_path + leaf for leaf in leaf_paths]
        file_size = 1024
        for path in paths:
            self._insert_random_file(self.client, path, file_size)
        for path in paths:
            self.assertTrue(self.aws.exists(path))
        self.aws.delete_tree(root_path)
        for path in paths:
            self.assertFalse(self.aws.exists(path))

    def test_exists(self):
        if False:
            return 10
        file_name = self.TEST_DATA_PATH + 'exists'
        file_size = 1024
        self.assertFalse(self.aws.exists(file_name))
        self._insert_random_file(self.aws.client, file_name, file_size)
        self.assertTrue(self.aws.exists(file_name))
        self.aws.delete(file_name)
        self.assertFalse(self.aws.exists(file_name))

    def test_file_mode(self):
        if False:
            print('Hello World!')
        file_name = self.TEST_DATA_PATH + 'jerry/pigpen/bobby'
        with self.aws.open(file_name, 'w') as f:
            assert f.mode == 'w'
        with self.aws.open(file_name, 'r') as f:
            assert f.mode == 'r'
        self.aws.delete(file_name)

    def test_full_file_read(self):
        if False:
            while True:
                i = 10
        file_name = self.TEST_DATA_PATH + 'jerry/pigpen/phil'
        file_size = 1024
        f = self._insert_random_file(self.aws.client, file_name, file_size)
        contents = f.contents
        f = self.aws.open(file_name)
        self.assertEqual(f.mode, 'r')
        f.seek(0, os.SEEK_END)
        self.assertEqual(f.tell(), file_size)
        self.assertEqual(f.read(), b'')
        f.seek(0)
        self.assertEqual(f.read(), contents)
        self.aws.delete(file_name)

    def test_file_write(self):
        if False:
            i = 10
            return i + 15
        file_name = self.TEST_DATA_PATH + 'write_file'
        file_size = 8 * 1024 * 1024 + 2000
        contents = os.urandom(file_size)
        f = self.aws.open(file_name, 'w')
        self.assertEqual(f.mode, 'w')
        f.write(contents[0:1000])
        f.write(contents[1000:1024 * 1024])
        f.write(contents[1024 * 1024:])
        f.close()
        new_f = self.aws.open(file_name, 'r')
        new_f_contents = new_f.read()
        self.assertEqual(new_f_contents, contents)
        self.aws.delete(file_name)

    def test_file_mime_type(self):
        if False:
            i = 10
            return i + 15
        if self.USE_MOCK:
            self.skipTest("The boto3_client mock doesn't support mime_types")
        mime_type = 'example/example'
        file_name = self.TEST_DATA_PATH + 'write_file'
        f = self.aws.open(file_name, 'w', mime_type=mime_type)
        f.write(b'a string of binary text')
        f.close()
        (bucket, key) = s3io.parse_s3_path(file_name)
        metadata = self.client.get_object_metadata(messages.GetRequest(bucket, key))
        self.assertEqual(mime_type, metadata.mime_type)
        self.aws.delete(file_name)

    def test_file_random_seek(self):
        if False:
            while True:
                i = 10
        file_name = self.TEST_DATA_PATH + 'write_seek_file'
        file_size = 5 * 1024 * 1024 - 100
        contents = os.urandom(file_size)
        with self.aws.open(file_name, 'w') as wf:
            wf.write(contents)
        f = self.aws.open(file_name)
        random.seed(0)
        for _ in range(0, 10):
            a = random.randint(0, file_size - 1)
            b = random.randint(0, file_size - 1)
            (start, end) = (min(a, b), max(a, b))
            f.seek(start)
            self.assertEqual(f.tell(), start)
            self.assertEqual(f.read(end - start + 1), contents[start:end + 1])
            self.assertEqual(f.tell(), end + 1)
        self.aws.delete(file_name)

    def test_file_flush(self):
        if False:
            i = 10
            return i + 15
        file_name = self.TEST_DATA_PATH + 'flush_file'
        file_size = 5 * 1024 * 1024 + 2000
        contents = os.urandom(file_size)
        f = self.aws.open(file_name, 'w')
        self.assertEqual(f.mode, 'w')
        f.write(contents[0:1000])
        f.flush()
        f.write(contents[1000:1024 * 1024])
        f.flush()
        f.flush()
        f.write(contents[1024 * 1024:])
        f.close()
        new_f = self.aws.open(file_name, 'r')
        new_f_contents = new_f.read()
        self.assertEqual(new_f_contents, contents)
        self.aws.delete(file_name)

    def test_file_iterator(self):
        if False:
            i = 10
            return i + 15
        file_name = self.TEST_DATA_PATH + 'iterate_file'
        lines = []
        line_count = 10
        for _ in range(line_count):
            line_length = random.randint(100, 500)
            line = os.urandom(line_length).replace(b'\n', b' ') + b'\n'
            lines.append(line)
        contents = b''.join(lines)
        with self.aws.open(file_name, 'w') as wf:
            wf.write(contents)
        f = self.aws.open(file_name)
        read_lines = 0
        for line in f:
            read_lines += 1
        self.assertEqual(read_lines, line_count)
        self.aws.delete(file_name)

    def test_file_read_line(self):
        if False:
            while True:
                i = 10
        file_name = self.TEST_DATA_PATH + 'read_line_file'
        lines = []
        read_buffer_size = 1099
        lines.append(b'x' * 1023 + b'\n')
        for _ in range(1, 1000):
            line_length = random.randint(100, 500)
            line = os.urandom(line_length).replace(b'\n', b' ') + b'\n'
            lines.append(line)
        contents = b''.join(lines)
        file_size = len(contents)
        with self.aws.open(file_name, 'wb') as wf:
            wf.write(contents)
        f = self.aws.open(file_name, 'rb', read_buffer_size=read_buffer_size)
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
        self.aws.delete(file_name)

    def test_file_close(self):
        if False:
            while True:
                i = 10
        file_name = self.TEST_DATA_PATH + 'close_file'
        file_size = 5 * 1024 * 1024 + 2000
        contents = os.urandom(file_size)
        f = self.aws.open(file_name, 'w')
        self.assertEqual(f.mode, 'w')
        f.write(contents)
        f.close()
        f.close()
        with self.aws.open(file_name, 'r') as f:
            read_contents = f.read()
        self.assertEqual(read_contents, contents)
        self.aws.delete(file_name)

    def test_context_manager(self):
        if False:
            return 10
        file_name = self.TEST_DATA_PATH + 'context_manager_file'
        file_size = 1024
        contents = os.urandom(file_size)
        with self.aws.open(file_name, 'w') as f:
            f.write(contents)
        with self.aws.open(file_name, 'r') as f:
            self.assertEqual(f.read(), contents)
        self.aws.delete(file_name)

    def test_list_prefix(self):
        if False:
            i = 10
            return i + 15
        objects = [('jerry/pigpen/phil', 5), ('jerry/pigpen/bobby', 3), ('jerry/billy/bobby', 4)]
        for (object_name, size) in objects:
            file_name = self.TEST_DATA_PATH + object_name
            self._insert_random_file(self.aws.client, file_name, size)
        test_cases = [(self.TEST_DATA_PATH + 'j', [('jerry/pigpen/phil', 5), ('jerry/pigpen/bobby', 3), ('jerry/billy/bobby', 4)]), (self.TEST_DATA_PATH + 'jerry/', [('jerry/pigpen/phil', 5), ('jerry/pigpen/bobby', 3), ('jerry/billy/bobby', 4)]), (self.TEST_DATA_PATH + 'jerry/pigpen/phil', [('jerry/pigpen/phil', 5)])]
        for (file_pattern, expected_object_names) in test_cases:
            expected_file_names = [(self.TEST_DATA_PATH + object_name, size) for (object_name, size) in expected_object_names]
            self.assertEqual(set(self.aws.list_prefix(file_pattern).items()), set(expected_file_names))
        for (object_name, size) in objects:
            self.aws.delete(self.TEST_DATA_PATH + object_name)

    def test_midsize_file(self):
        if False:
            i = 10
            return i + 15
        file_name = self.TEST_DATA_PATH + 'midsized'
        file_size = 6 * 1024 * 1024
        self._insert_random_file(self.aws.client, file_name, file_size)
        with self.aws.open(file_name, 'r') as f:
            self.assertEqual(len(f.read()), file_size)
        self.aws.delete(file_name)

    def test_zerosize_file(self):
        if False:
            return 10
        file_name = self.TEST_DATA_PATH + 'zerosized'
        file_size = 0
        self._insert_random_file(self.aws.client, file_name, file_size)
        with self.aws.open(file_name, 'r') as f:
            self.assertEqual(len(f.read()), file_size)
        self.aws.delete(file_name)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()