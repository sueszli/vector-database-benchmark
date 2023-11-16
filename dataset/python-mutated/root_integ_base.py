import os
from unittest import TestCase
from tests.testing_utils import get_sam_command

class RootIntegBase(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()

    def tearDown(self):
        if False:
            return 10
        super().tearDown()

    def root_command_list(self, info=False, debug=False, version=False, _help=False):
        if False:
            i = 10
            return i + 15
        command_list = [get_sam_command()]
        if info:
            command_list += ['--info']
        if debug:
            command_list += ['--debug']
        if _help:
            command_list += ['--help']
        if version:
            command_list += ['--version']
        return command_list