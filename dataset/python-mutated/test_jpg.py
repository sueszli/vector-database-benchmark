import unittest
import shutil
import os
from . import base

class JpgTestCase(base.ShellParserTestCase, unittest.TestCase):
    extension = 'jpg'

    def get_jpeg_filename(self, contents_filename):
        if False:
            i = 10
            return i + 15
        temp_filename = self.get_temp_filename()
        jpeg_filename = temp_filename + '.jpeg'
        os.remove(temp_filename)
        shutil.copyfile(contents_filename, jpeg_filename)
        return jpeg_filename

    def test_jpeg_synonym_cli(self):
        if False:
            return 10
        'Make sure .jpeg synonym works in cli'
        jpeg_filename = self.get_jpeg_filename(self.raw_text_filename)
        self.compare_cli_output(jpeg_filename, self.get_expected_filename(self.raw_text_filename))
        os.remove(jpeg_filename)

    def test_jpeg_synonym_python(self):
        if False:
            while True:
                i = 10
        'Make sure .jpeg synonym works in python'
        jpeg_filename = self.get_jpeg_filename(self.raw_text_filename)
        self.compare_python_output(jpeg_filename, self.get_expected_filename(self.raw_text_filename))
        os.remove(jpeg_filename)