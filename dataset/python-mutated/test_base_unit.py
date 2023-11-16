import unittest
import pytest
import azure.cosmos._base as base
pytestmark = pytest.mark.cosmosEmulator

@pytest.mark.usefixtures('teardown')
class BaseUnitTests(unittest.TestCase):

    def test_is_name_based(self):
        if False:
            print('Hello World!')
        self.assertFalse(base.IsNameBased('dbs/xjwmAA==/'))
        self.assertTrue(base.IsNameBased('dbs/paas_cmr'))