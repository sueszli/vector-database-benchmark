"""Tests for core.platform.app_identity.gae_app_identity_services."""
from __future__ import annotations
from core import feconf
from core.platform.app_identity import gae_app_identity_services
from core.tests import test_utils

class GaeAppIdentityServicesTests(test_utils.GenericTestBase):

    def test_get_application_id(self) -> None:
        if False:
            return 10
        with self.swap(feconf, 'OPPIA_PROJECT_ID', 'some_id'):
            self.assertEqual(gae_app_identity_services.get_application_id(), 'some_id')

    def test_get_application_id_throws_error(self) -> None:
        if False:
            return 10
        with self.swap(feconf, 'OPPIA_PROJECT_ID', None):
            with self.assertRaisesRegex(ValueError, 'Value None for application id is invalid.'):
                gae_app_identity_services.get_application_id()

    def test_get_default_gcs_bucket_name(self) -> None:
        if False:
            return 10
        with self.swap(feconf, 'OPPIA_PROJECT_ID', 'some_id'):
            self.assertEqual(gae_app_identity_services.get_gcs_resource_bucket_name(), 'some_id-resources')