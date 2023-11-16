"""Ensures that pywrap_gradient_exclusions.cc is up-to-date."""
import os
from tensorflow.python.eager import gradient_input_output_exclusions
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test

class GradientInputOutputExclusionsTest(test.TestCase):

    def testGeneratedFileMatchesHead(self):
        if False:
            return 10
        expected_contents = gradient_input_output_exclusions.get_contents()
        filename = os.path.join(resource_loader.get_root_dir_with_all_resources(), resource_loader.get_path_to_datafile('pywrap_gradient_exclusions.cc'))
        actual_contents = file_io.read_file_to_string(filename)
        sanitized_actual_contents = actual_contents.replace('\r', '')
        sanitized_expected_contents = expected_contents.replace('\r', '')
        self.assertEqual(sanitized_actual_contents, sanitized_expected_contents, '\npywrap_gradient_exclusions.cc needs to be updated.\nPlease regenerate using:\nbazel run tensorflow/python/eager:gen_gradient_input_output_exclusions -- $PWD/tensorflow/python/eager/pywrap_gradient_exclusions.cc')
if __name__ == '__main__':
    test.main()