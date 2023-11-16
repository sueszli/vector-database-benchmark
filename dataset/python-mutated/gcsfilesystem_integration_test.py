"""Integration tests for gcsfilesystem module.

Runs tests against Google Cloud Storage service.
Instantiates a TestPipeline to get options such as GCP project name, but
doesn't actually start a Beam pipeline or test any specific runner.

To run these tests manually:
  ./gradlew :sdks:python:test-suites:dataflow:integrationTest     -Dtests=apache_beam.io.gcp.gcsfilesystem_integration_test:GcsFileSystemIntegrationTest # pylint: disable=line-too-long
"""
import logging
import unittest
import uuid
import pytest
from apache_beam.io.filesystem import BeamIOError
from apache_beam.io.filesystems import FileSystems
from apache_beam.testing.test_pipeline import TestPipeline
try:
    from apache_beam.io.gcp.gcsfilesystem import GCSFileSystem
    fs_not_available = False
except ImportError:
    fs_not_available = True

@unittest.skipIf(fs_not_available, 'GCP dependencies are not installed')
class GcsFileSystemIntegrationTest(unittest.TestCase):
    INPUT_FILE = 'gs://dataflow-samples/shakespeare/kinglear.txt'
    INPUT_FILE_LARGE = 'gs://apache-beam-samples-us-east1/wikipedia_edits/wiki_data-000000000000.json'

    def setUp(self):
        if False:
            while True:
                i = 10
        self.test_pipeline = TestPipeline(is_integration_test=True)
        self.runner_name = type(self.test_pipeline.runner).__name__
        if self.runner_name != 'TestDataflowRunner':
            raise unittest.SkipTest('This test only runs with TestDataflowRunner.')
        self.project = self.test_pipeline.get_option('project')
        self.gcs_tempdir = self.test_pipeline.get_option('temp_location') + '/gcs_it-' + str(uuid.uuid4())
        self.fs = GCSFileSystem(self.test_pipeline.get_pipeline_options())

    def tearDown(self):
        if False:
            return 10
        FileSystems.delete([self.gcs_tempdir + '/'])

    def _verify_copy(self, src, dest):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(FileSystems.exists(src), 'src does not exist: %s' % src)
        self.assertTrue(FileSystems.exists(dest), 'dest does not exist: %s' % dest)
        src_checksum = self.fs.checksum(src)
        dest_checksum = self.fs.checksum(dest)
        self.assertEqual(src_checksum, dest_checksum)

    def _verify_rename(self, src, dest):
        if False:
            i = 10
            return i + 15
        self.assertFalse(FileSystems.exists(src), 'file %s not renamed' % src)
        self.assertTrue(FileSystems.exists(dest), 'file not renamed to %s' % dest)

    def _test_copy(self, name, max_bytes_rewritten_per_call=None, src=None):
        if False:
            return 10
        src = src or self.INPUT_FILE
        dest = self.gcs_tempdir + '/%s' % name
        extra_kwargs = {}
        if max_bytes_rewritten_per_call is not None:
            extra_kwargs['max_bytes_rewritten_per_call'] = max_bytes_rewritten_per_call
        self.fs.copy([src], [dest])
        self._verify_copy(src, dest)

    @pytest.mark.it_postcommit
    def test_copy(self):
        if False:
            i = 10
            return i + 15
        self._test_copy('test_copy')

    @pytest.mark.it_postcommit
    def test_rename(self):
        if False:
            for i in range(10):
                print('nop')
        num_copies = 10
        srcs = [self.INPUT_FILE] * num_copies
        dests = [self.gcs_tempdir + '/%s_%d' % (self.INPUT_FILE, i) for i in range(num_copies)]
        self.fs.copy(srcs, dests)
        new_names = [self.gcs_tempdir + '/%s_%d' % ('renamed', i) for i in range(num_copies)]
        self.fs.rename(dests, new_names)
        for (_old, _new) in list(zip(dests, new_names)):
            self._verify_rename(_old, _new)

    @pytest.mark.it_postcommit
    def test_rename_error(self):
        if False:
            print('Hello World!')
        num_copies = 10
        srcs = [self.INPUT_FILE] * num_copies
        dests = [self.gcs_tempdir + '/%s_%d' % (self.INPUT_FILE, i) for i in range(num_copies)]
        self.fs.copy(srcs, dests)
        new_names = [self.gcs_tempdir + '/%s_%d' % ('renamed', i) for i in range(num_copies)]
        bad_name = self.gcs_tempdir + '/errorbadwrong'
        dests[int(num_copies / 2)] = bad_name
        with self.assertRaises(BeamIOError):
            self.fs.rename(dests, new_names)
        for (_old, _new) in list(zip(dests, new_names)):
            if _old != bad_name:
                self._verify_rename(_old, _new)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()