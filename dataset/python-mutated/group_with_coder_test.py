"""Test for the custom coders example."""
import logging
import unittest
import uuid
import pytest
from apache_beam.examples.cookbook import group_with_coder
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.test_utils import read_files_from_pattern
try:
    from apache_beam.io.gcp import gcsio
except ImportError:
    gcsio = None
group_with_coder.PlayerCoder.decode = lambda self, s: group_with_coder.Player(s.decode('utf-8'))

def create_content_input_file(path, records):
    if False:
        while True:
            i = 10
    logging.info('Creating file: %s', path)
    gcs = gcsio.GcsIO()
    with gcs.open(path, 'w') as f:
        for record in records:
            f.write(b'%s\n' % record.encode('utf-8'))
    return path

@unittest.skipIf(gcsio is None, 'GCP dependencies are not installed')
@pytest.mark.examples_postcommit
class GroupWithCoderTest(unittest.TestCase):
    SAMPLE_RECORDS = ['joe,10', 'fred,3', 'mary,7', 'joe,20', 'fred,6', 'ann,5', 'joe,30', 'ann,10', 'mary,1']

    def setUp(self):
        if False:
            return 10
        self.test_pipeline = TestPipeline(is_integration_test=True)
        self.temp_location = self.test_pipeline.get_option('temp_location')
        self.input_file = create_content_input_file('/'.join([self.temp_location, str(uuid.uuid4()), 'input.txt']), self.SAMPLE_RECORDS)

    @pytest.mark.sickbay_dataflow
    def test_basics_with_type_check(self):
        if False:
            while True:
                i = 10
        output = '/'.join([self.temp_location, str(uuid.uuid4()), 'result'])
        extra_opts = {'input': self.input_file, 'output': output}
        group_with_coder.run(self.test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
        results = []
        lines = read_files_from_pattern('%s*' % output).splitlines()
        for line in lines:
            (name, points) = line.split(',')
            results.append((name, int(points)))
            logging.info('result: %s', results)
        self.assertEqual(sorted(results), sorted([('x:ann', 15), ('x:fred', 9), ('x:joe', 60), ('x:mary', 8)]))

    def test_basics_without_type_check(self):
        if False:
            while True:
                i = 10
        output = '/'.join([self.temp_location, str(uuid.uuid4()), 'result'])
        extra_opts = {'input': self.input_file, 'output': output}
        with self.assertRaises(Exception) as context:
            group_with_coder.run(self.test_pipeline.get_full_options_as_args(**extra_opts) + ['--no_pipeline_type_check'], save_main_session=False)
        self.assertIn('Unable to deterministically encode', str(context.exception))
        self.assertIn('CombinePerKey(sum)/GroupByKey', str(context.exception))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()