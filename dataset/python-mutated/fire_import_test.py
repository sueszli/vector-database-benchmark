"""Tests importing the fire module."""
import sys
import fire
from fire import testutils
import mock

class FireImportTest(testutils.BaseTestCase):
    """Tests importing Fire."""

    def testFire(self):
        if False:
            while True:
                i = 10
        with mock.patch.object(sys, 'argv', ['commandname']):
            fire.Fire()

    def testFireMethods(self):
        if False:
            while True:
                i = 10
        self.assertIsNotNone(fire.Fire)

    def testNoPrivateMethods(self):
        if False:
            print('Hello World!')
        self.assertTrue(hasattr(fire, 'Fire'))
        self.assertFalse(hasattr(fire, '_Fire'))
if __name__ == '__main__':
    testutils.main()