import copy
import collections
import cloudinary
from cloudinary.cache.adapter.cache_adapter import CacheAdapter
from cloudinary.utils import check_property_enabled

class ResponsiveBreakpointsCache:
    """
    Caches breakpoint values for image resources
    """

    def __init__(self, **cache_options):
        if False:
            return 10
        '\n        Initialize the cache\n\n        :param cache_options: Cache configuration options\n        '
        self._cache_adapter = None
        cache_adapter = cache_options.get('cache_adapter')
        self.set_cache_adapter(cache_adapter)

    def set_cache_adapter(self, cache_adapter):
        if False:
            i = 10
            return i + 15
        '\n        Assigns cache adapter\n\n        :param cache_adapter: The cache adapter used to store and retrieve values\n\n        :return: Returns True if the cache_adapter is valid\n        '
        if cache_adapter is None or not isinstance(cache_adapter, CacheAdapter):
            return False
        self._cache_adapter = cache_adapter
        return True

    @property
    def enabled(self):
        if False:
            print('Hello World!')
        '\n        Indicates whether cache is enabled or not\n\n        :return: Rrue if a _cache_adapter has been set\n        '
        return self._cache_adapter is not None

    @staticmethod
    def _options_to_parameters(**options):
        if False:
            while True:
                i = 10
        '\n        Extract the parameters required in order to calculate the key of the cache.\n\n        :param options: Input options\n\n        :return: A list of values used to calculate the cache key\n        '
        options_copy = copy.deepcopy(options)
        (transformation, _) = cloudinary.utils.generate_transformation_string(**options_copy)
        file_format = options.get('format', '')
        storage_type = options.get('type', 'upload')
        resource_type = options.get('resource_type', 'image')
        return (storage_type, resource_type, transformation, file_format)

    @check_property_enabled
    def get(self, public_id, **options):
        if False:
            i = 10
            return i + 15
        '\n        Retrieve the breakpoints of a particular derived resource identified by the public_id and options\n\n        :param public_id: The public ID of the resource\n        :param options: The public ID of the resource\n\n        :return: Array of responsive breakpoints, None if not found\n        '
        params = self._options_to_parameters(**options)
        return self._cache_adapter.get(public_id, *params)

    @check_property_enabled
    def set(self, public_id, value, **options):
        if False:
            while True:
                i = 10
        '\n        Set responsive breakpoints identified by public ID and options\n\n        :param public_id: The public ID of the resource\n        :param value:  Array of responsive breakpoints to set\n        :param options: Additional options\n\n        :return: True on success or False on failure\n        '
        if not isinstance(value, (list, tuple)):
            raise ValueError('A list of breakpoints is expected')
        (storage_type, resource_type, transformation, file_format) = self._options_to_parameters(**options)
        return self._cache_adapter.set(public_id, storage_type, resource_type, transformation, file_format, value)

    @check_property_enabled
    def delete(self, public_id, **options):
        if False:
            while True:
                i = 10
        '\n        Delete responsive breakpoints identified by public ID and options\n\n        :param public_id: The public ID of the resource\n        :param options: Additional options\n\n        :return: True on success or False on failure\n        '
        params = self._options_to_parameters(**options)
        return self._cache_adapter.delete(public_id, *params)

    @check_property_enabled
    def flush_all(self):
        if False:
            while True:
                i = 10
        '\n        Flush all entries from cache\n\n        :return: True on success or False on failure\n        '
        return self._cache_adapter.flush_all()
instance = ResponsiveBreakpointsCache()