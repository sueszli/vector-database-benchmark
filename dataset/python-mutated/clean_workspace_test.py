import unittest
from caffe2.python import workspace

class TestWorkspace(unittest.TestCase):

    def testRootFolder(self):
        if False:
            print('Hello World!')
        self.assertEqual(workspace.ResetWorkspace(), True)
        self.assertEqual(workspace.RootFolder(), '.')
        self.assertEqual(workspace.ResetWorkspace('/tmp/caffe-workspace-test'), True)
        self.assertEqual(workspace.RootFolder(), '/tmp/caffe-workspace-test')