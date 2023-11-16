from django.core.cache.backends.locmem import LocMemCache

class CloseHookMixin:
    closed = False

    def close(self, **kwargs):
        if False:
            i = 10
            return i + 15
        self.closed = True

class CacheClass(CloseHookMixin, LocMemCache):
    pass