"""Unit tests for caching_domain.py"""
from __future__ import annotations
from core.domain import caching_domain
from core.tests import test_utils

class CachingDomainTests(test_utils.GenericTestBase):

    def test_that_domain_object_is_created_correctly(self) -> None:
        if False:
            i = 10
            return i + 15
        memory_cache = caching_domain.MemoryCacheStats(64, 128, 16)
        self.assertEqual(memory_cache.total_allocated_in_bytes, 64)
        self.assertEqual(memory_cache.peak_memory_usage_in_bytes, 128)
        self.assertEqual(memory_cache.total_number_of_keys_stored, 16)