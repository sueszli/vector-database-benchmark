import json
import os
import random
import shutil
import string
import sys
import time
import unittest
from argparse import Namespace
from datetime import datetime
from nni.tools.trial_tool.base_channel import CommandType
from nni.tools.trial_tool.file_channel import FileChannel, command_path, manager_commands_file_name
sys.path.append('..')
runner_file_name = 'commands/runner_commands.txt'
manager_file_name = 'commands/manager_commands.txt'

class FileChannelTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.args = Namespace()
        self.args.node_count = 1
        self.args.node_id = None
        if os.path.exists(command_path):
            shutil.rmtree(command_path)

    def check_timeout(self, timeout, callback):
        if False:
            while True:
                i = 10
        interval = 0.01
        start = datetime.now().timestamp()
        count = int(timeout / interval)
        for x in range(count):
            if callback():
                break
            time.sleep(interval)
        print('checked {} times, {:3F} seconds'.format(x, datetime.now().timestamp() - start))
if __name__ == '__main__':
    unittest.main()