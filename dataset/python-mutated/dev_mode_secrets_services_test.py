"""Tests for the Python Cloud Secret services."""
from __future__ import annotations
import os
from core.platform.secrets import dev_mode_secrets_services
from core.tests import test_utils

class DevModeSecretsServicesTests(test_utils.GenericTestBase):
    """Tests for the Python Cloud Secret services."""

    def test_get_secret_returns_existing_secret(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.swap(os, 'environ', {'SECRETS': '{"name": "secret"}'}):
            self.assertEqual(dev_mode_secrets_services.get_secret('name'), 'secret')

    def test_get_secret_returns_none_when_secret_does_not_exist(self) -> None:
        if False:
            return 10
        with self.swap(os, 'environ', {'SECRETS': '{"name": "secret"}'}):
            self.assertIsNone(dev_mode_secrets_services.get_secret('name2'))