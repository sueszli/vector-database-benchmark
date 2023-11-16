"""
Validate the boto_iam module
"""
import pytest
from tests.support.case import ModuleCase
try:
    import boto
    NO_BOTO_MODULE = False
except ImportError:
    NO_BOTO_MODULE = True

@pytest.mark.skipif(NO_BOTO_MODULE, reason='Please install the boto library before running boto integration tests.')
class BotoIAMTest(ModuleCase):

    def setUp(self):
        if False:
            print('Hello World!')
        try:
            boto.connect_iam()
        except boto.exception.NoAuthHandlerFound:
            self.skipTest('Please setup boto AWS credentials before running boto integration tests.')

    def test_get_account_id(self):
        if False:
            while True:
                i = 10
        ret = self.run_function('boto_iam.get_account_id')
        self.assertRegex(ret, '^\\d{12}$')