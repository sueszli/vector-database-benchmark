from easydict import EasyDict
import pytest
import unittest
from unittest import mock
from unittest.mock import patch
import pathlib as pl
import os
import shutil
from ding.utils.profiler_helper import Profiler, register_profiler

@pytest.mark.unittest
class TestProfilerModule:

    def assertIsFile(self, path):
        if False:
            print('Hello World!')
        if not pl.Path(path).resolve().is_file():
            raise AssertionError('File does not exist: %s' % str(path))

    def test(self):
        if False:
            while True:
                i = 10
        profiler = Profiler()

        def register_mock(write_profile, pr, folder_path):
            if False:
                print('Hello World!')
            profiler.write_profile(pr, folder_path)

        def clean_up(dir):
            if False:
                i = 10
                return i + 15
            if os.path.exists(dir):
                shutil.rmtree(dir)
        dir = './tmp_test/'
        clean_up(dir)
        with patch('ding.utils.profiler_helper.register_profiler', register_mock):
            profiler.profile(dir)
            file_path = os.path.join(dir, 'profile_tottime.txt')
            self.assertIsFile(file_path)
            file_path = os.path.join(dir, 'profile_cumtime.txt')
            self.assertIsFile(file_path)
            file_path = os.path.join(dir, 'profile.prof')
            self.assertIsFile(file_path)
        clean_up(dir)