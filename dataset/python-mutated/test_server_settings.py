import os
from unittest import mock
from zerver.lib.test_classes import ZulipTestCase
from zproject import config

class ConfigTest(ZulipTestCase):

    def test_get_mandatory_secret_succeed(self) -> None:
        if False:
            print('Hello World!')
        secret = config.get_mandatory_secret('shared_secret')
        self.assertGreater(len(secret), 0)

    def test_get_mandatory_secret_failed(self) -> None:
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(config.ZulipSettingsError, 'nonexistent'):
            config.get_mandatory_secret('nonexistent')

    def test_disable_mandatory_secret_check(self) -> None:
        if False:
            while True:
                i = 10
        with mock.patch.dict(os.environ, {'DISABLE_MANDATORY_SECRET_CHECK': 'True'}):
            secret = config.get_mandatory_secret('nonexistent')
        self.assertEqual(secret, '')