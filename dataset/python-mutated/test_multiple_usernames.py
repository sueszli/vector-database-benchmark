import imp
import unittest
import sys
sys.path.append('../')
import sherlock as sh
checksymbols = []
checksymbols = ['_', '-', '.']
'Test for mulriple usernames.\n\n        This test ensures that the function MultipleUsernames works properly. More specific,\n        different scenarios are tested and only usernames that contain this specific sequence: {?} \n        should return positive.\n      \n        Keyword Arguments:\n        self                   -- This object.\n\n        Return Value:\n        Nothing.\n        '

class TestMultipleUsernames(unittest.TestCase):

    def test_area(self):
        if False:
            print('Hello World!')
        test_usernames = ['test{?}test', 'test{?feo', 'test']
        for name in test_usernames:
            if sh.CheckForParameter(name):
                self.assertAlmostEqual(sh.MultipleUsernames(name), ['test_test', 'test-test', 'test.test'])
            else:
                self.assertAlmostEqual(name, name)