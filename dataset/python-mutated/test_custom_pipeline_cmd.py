import os
import shutil
import subprocess
import tempfile
import unittest
import uuid

class ModelUploadCMDTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.task_name = 'task-%s' % uuid.uuid4().hex
        print(self.task_name)

    def test_upload_modelcard(self):
        if False:
            return 10
        cmd = f'python -m modelscope.cli.cli pipeline --action create --task_name {self.task_name} '
        (stat, output) = subprocess.getstatusoutput(cmd)
        if stat != 0:
            print(output)
if __name__ == '__main__':
    unittest.main()