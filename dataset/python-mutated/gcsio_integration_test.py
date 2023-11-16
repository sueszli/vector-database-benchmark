"""Integration tests for gcsio module.

Runs tests against Google Cloud Storage service.
Instantiates a TestPipeline to get options such as GCP project name, but
doesn't actually start a Beam pipeline or test any specific runner.

Options:
  --kms_key_name=projects/<project-name>/locations/<region>/keyRings/      <key-ring-name>/cryptoKeys/<key-name>/cryptoKeyVersions/<version>
    Pass a Cloud KMS key name to test GCS operations using customer managed
    encryption keys (CMEK).

Cloud KMS permissions:
The project's Cloud Storage service account requires Encrypter/Decrypter
permissions for the key specified in --kms_key_name.

To run these tests manually:
  ./gradlew :sdks:python:test-suites:dataflow:integrationTest     -Dtests=apache_beam.io.gcp.gcsio_integration_test:GcsIOIntegrationTest     -DkmsKeyName=KMS_KEY_NAME
"""
import logging
import unittest
import uuid
import pytest
from apache_beam.io.filesystems import FileSystems
from apache_beam.testing.test_pipeline import TestPipeline
try:
    from apache_beam.io.gcp import gcsio
except ImportError:
    gcsio = None

@unittest.skipIf(gcsio is None, 'GCP dependencies are not installed')
class GcsIOIntegrationTest(unittest.TestCase):
    INPUT_FILE = 'gs://dataflow-samples/shakespeare/kinglear.txt'
    INPUT_FILE_LARGE = 'gs://apache-beam-samples-us-east1/wikipedia_edits/wiki_data-000000000000.json'

    def setUp(self):
        if False:
            print('Hello World!')
        self.test_pipeline = TestPipeline(is_integration_test=True)
        self.runner_name = type(self.test_pipeline.runner).__name__
        if self.runner_name != 'TestDataflowRunner':
            raise unittest.SkipTest('This test only runs with TestDataflowRunner.')
        self.project = self.test_pipeline.get_option('project')
        self.gcs_tempdir = self.test_pipeline.get_option('temp_location') + '/gcs_it-' + str(uuid.uuid4())
        self.kms_key_name = self.test_pipeline.get_option('kms_key_name')
        self.gcsio = gcsio.GcsIO()

    def tearDown(self):
        if False:
            return 10
        FileSystems.delete([self.gcs_tempdir + '/'])

    def _verify_copy(self, src, dst, dst_kms_key_name=None):
        if False:
            while True:
                i = 10
        self.assertTrue(FileSystems.exists(src), 'src does not exist: %s' % src)
        self.assertTrue(FileSystems.exists(dst), 'dst does not exist: %s' % dst)
        src_checksum = self.gcsio.checksum(src)
        dst_checksum = self.gcsio.checksum(dst)
        self.assertEqual(src_checksum, dst_checksum)
        actual_dst_kms_key = self.gcsio.kms_key(dst)
        if actual_dst_kms_key is None:
            self.assertEqual(actual_dst_kms_key, dst_kms_key_name)
        else:
            self.assertTrue(actual_dst_kms_key.startswith(dst_kms_key_name), 'got: %s, wanted startswith: %s' % (actual_dst_kms_key, dst_kms_key_name))

    def _test_copy(self, name, kms_key_name=None, max_bytes_rewritten_per_call=None, src=None):
        if False:
            while True:
                i = 10
        src = src or self.INPUT_FILE
        dst = self.gcs_tempdir + '/%s' % name
        extra_kwargs = {}
        if max_bytes_rewritten_per_call is not None:
            extra_kwargs['max_bytes_rewritten_per_call'] = max_bytes_rewritten_per_call
        self.gcsio.copy(src, dst, kms_key_name, **extra_kwargs)
        self._verify_copy(src, dst, kms_key_name)

    @pytest.mark.it_postcommit
    def test_copy(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_copy('test_copy')

    @pytest.mark.it_postcommit
    def test_copy_kms(self):
        if False:
            for i in range(10):
                print('nop')
        if self.kms_key_name is None:
            raise unittest.SkipTest('--kms_key_name not specified')
        self._test_copy('test_copy_kms', self.kms_key_name)

    @pytest.mark.it_postcommit
    def test_copy_rewrite_token(self):
        if False:
            for i in range(10):
                print('nop')
        if self.kms_key_name is None:
            raise unittest.SkipTest('--kms_key_name not specified')
        rewrite_responses = []
        self.gcsio._set_rewrite_response_callback(lambda response: rewrite_responses.append(response))
        self._test_copy('test_copy_rewrite_token', kms_key_name=self.kms_key_name, max_bytes_rewritten_per_call=50 * 1024 * 1024, src=self.INPUT_FILE_LARGE)
        self.assertTrue(any((not r.done for r in rewrite_responses)))

    def _test_copy_batch(self, name, kms_key_name=None, max_bytes_rewritten_per_call=None, src=None):
        if False:
            return 10
        num_copies = 10
        srcs = [src or self.INPUT_FILE] * num_copies
        dsts = [self.gcs_tempdir + '/%s_%d' % (name, i) for i in range(num_copies)]
        src_dst_pairs = list(zip(srcs, dsts))
        extra_kwargs = {}
        if max_bytes_rewritten_per_call is not None:
            extra_kwargs['max_bytes_rewritten_per_call'] = max_bytes_rewritten_per_call
        result_statuses = self.gcsio.copy_batch(src_dst_pairs, kms_key_name, **extra_kwargs)
        for status in result_statuses:
            self.assertIsNone(status[2], status)
        for (_src, _dst) in src_dst_pairs:
            self._verify_copy(_src, _dst, kms_key_name)

    @pytest.mark.it_postcommit
    def test_copy_batch(self):
        if False:
            i = 10
            return i + 15
        self._test_copy_batch('test_copy_batch')

    @pytest.mark.it_postcommit
    def test_copy_batch_kms(self):
        if False:
            for i in range(10):
                print('nop')
        if self.kms_key_name is None:
            raise unittest.SkipTest('--kms_key_name not specified')
        self._test_copy_batch('test_copy_batch_kms', self.kms_key_name)

    @pytest.mark.it_postcommit
    def test_copy_batch_rewrite_token(self):
        if False:
            return 10
        if self.kms_key_name is None:
            raise unittest.SkipTest('--kms_key_name not specified')
        rewrite_responses = []
        self.gcsio._set_rewrite_response_callback(lambda response: rewrite_responses.append(response))
        self._test_copy_batch('test_copy_batch_rewrite_token', kms_key_name=self.kms_key_name, max_bytes_rewritten_per_call=50 * 1024 * 1024, src=self.INPUT_FILE_LARGE)
        self.assertTrue(any((not r.done for r in rewrite_responses)))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()