"""Tests for extensions domain."""
from __future__ import annotations
from core.tests import test_utils
from extensions import domain

class CustomizationArgSpecDomainUnitTests(test_utils.GenericTestBase):
    """Tests for CustomizationArgSpec domain object methods."""

    def test_to_dict(self) -> None:
        if False:
            return 10
        ca_spec = domain.CustomizationArgSpec('name', 'description', {}, None)
        self.assertEqual(ca_spec.to_dict(), {'name': 'name', 'description': 'description', 'schema': {}, 'default_value': None})