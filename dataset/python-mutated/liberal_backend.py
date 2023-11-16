from django.core.cache.backends.locmem import LocMemCache

class LiberalKeyValidationMixin:

    def validate_key(self, key):
        if False:
            i = 10
            return i + 15
        pass

class CacheClass(LiberalKeyValidationMixin, LocMemCache):
    pass