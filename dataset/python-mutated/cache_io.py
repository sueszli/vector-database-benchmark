"""Provides PTransforms for manipulating with cache."""
from __future__ import annotations
from core.domain import caching_services
import apache_beam as beam
from typing import Any

class FlushCache(beam.PTransform):
    """Flushes the memory caches."""

    def expand(self, items: beam.PCollection[Any]) -> beam.pvalue.PDone:
        if False:
            while True:
                i = 10
        'Flushes the memory caches.\n\n        Args:\n            items: PCollection. Items, can also contain just one item.\n\n        Returns:\n            PCollection. An empty PCollection.\n        '
        return items | beam.CombineGlobally(lambda _: []) | beam.Map(lambda _: caching_services.flush_memory_caches())