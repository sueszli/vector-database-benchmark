"""Unit tests for scripts/linters/warranted_angular_security_bypasses.py"""
from __future__ import annotations
from core.tests import test_utils
from . import warranted_angular_security_bypasses

class WarrantedAngularSecurityBypassesTests(test_utils.GenericTestBase):

    def test_svg_sanitizer_service_is_present_in_excluded_files(self) -> None:
        if False:
            return 10
        excluded_files = warranted_angular_security_bypasses.EXCLUDED_BYPASS_SECURITY_TRUST_FILES
        self.assertIn('core/templates/services/svg-sanitizer.service.spec.ts', excluded_files)
        self.assertIn('core/templates/services/svg-sanitizer.service.ts', excluded_files)