"""Test for the juliaset example."""
import logging
import os
import re
import tempfile
import unittest
import pytest
from apache_beam.examples.complete.juliaset.juliaset import juliaset
from apache_beam.testing.util import open_shards

@pytest.mark.examples_postcommit
class JuliaSetTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.test_files = {}
        self.test_files['output_coord_file_name'] = self.generate_temp_file()
        self.test_files['output_image_file_name'] = self.generate_temp_file()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        for test_file in self.test_files.values():
            if os.path.exists(test_file):
                os.remove(test_file)

    def generate_temp_file(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            return temp_file.name

    def run_example(self, grid_size, image_file_name=None):
        if False:
            i = 10
            return i + 15
        args = ['--coordinate_output=%s' % self.test_files['output_coord_file_name'], '--grid_size=%s' % grid_size]
        if image_file_name is not None:
            args.append('--image_output=%s' % image_file_name)
        juliaset.run(args)

    def test_output_file_format(self):
        if False:
            while True:
                i = 10
        grid_size = 5
        self.run_example(grid_size)
        with open_shards(self.test_files['output_coord_file_name'] + '-*-of-*') as result_file:
            output_lines = result_file.readlines()
            self.assertEqual(grid_size, len(output_lines))
            for line in output_lines:
                coordinates = re.findall('(\\(\\d+, \\d+, \\d+\\))', line)
                self.assertTrue(coordinates)
                self.assertEqual(grid_size, len(coordinates))

    @unittest.skip('TODO(silviuc): Reactivate the test when --image_output is supported.')
    def test_generate_fractal_image(self):
        if False:
            for i in range(10):
                print('nop')
        temp_image_file = self.test_files['output_image_file_name']
        self.run_example(10, image_file_name=temp_image_file)
        self.assertTrue(os.stat(temp_image_file).st_size > 0)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()