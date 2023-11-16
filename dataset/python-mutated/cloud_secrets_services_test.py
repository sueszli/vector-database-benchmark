"""Tests for the Python Cloud Secret services."""
from __future__ import annotations
import types
from core.platform.secrets import cloud_secrets_services
from core.tests import test_utils

class CloudSecretsServicesTests(test_utils.GenericTestBase):
    """Tests for the Python Cloud Secret services."""

    def test_get_secret_returns_existing_secret(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.swap_to_always_return(cloud_secrets_services.CLIENT, 'access_secret_version', types.SimpleNamespace(payload=types.SimpleNamespace(data=b'secre'))):
            self.assertEqual(cloud_secrets_services.get_secret('name'), 'secre')

    def test_get_secret_returns_none_when_secret_does_not_exist(self) -> None:
        if False:
            print('Hello World!')
        with self.swap_to_always_raise(cloud_secrets_services.CLIENT, 'access_secret_version', Exception('Secret not found')):
            self.assertIsNone(cloud_secrets_services.get_secret('name2'))