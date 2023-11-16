"""Domain objects for caching.

Domain objects capture domain-specific logic and are agnostic of how the
objects they represent are stored. All methods and properties in this file
should therefore be independent of the specific storage models used.
"""
from __future__ import annotations

class MemoryCacheStats:
    """Domain object for an Oppia memory profile object that contains
    information about the memory cache.
    """

    def __init__(self, total_allocated_in_bytes: int, peak_memory_usage_in_bytes: int, total_number_of_keys_stored: int) -> None:
        if False:
            return 10
        'Initializes a Memory Cache Stats domain object.\n\n        Args:\n            total_allocated_in_bytes: int. The total number of bytes allocated\n                by the memory cache.\n            peak_memory_usage_in_bytes: int. The highest number of bytes\n                allocated by the memory cache.\n            total_number_of_keys_stored: int. The number of keys stored in the\n                memory cache.\n        '
        self.total_allocated_in_bytes = total_allocated_in_bytes
        self.peak_memory_usage_in_bytes = peak_memory_usage_in_bytes
        self.total_number_of_keys_stored = total_number_of_keys_stored