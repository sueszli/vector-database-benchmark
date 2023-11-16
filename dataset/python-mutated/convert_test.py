from caffe2.python import workspace
import unittest

class TestOperator(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        workspace.ResetWorkspace()
if __name__ == '__main__':
    unittest.main()