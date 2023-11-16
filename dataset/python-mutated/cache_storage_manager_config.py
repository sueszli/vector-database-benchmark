from streamlit.runtime.caching.storage import CacheStorageManager
from streamlit.runtime.caching.storage.local_disk_cache_storage import LocalDiskCacheStorageManager

def create_default_cache_storage_manager() -> CacheStorageManager:
    if False:
        return 10
    '\n    Get the cache storage manager.\n    It would be used both in server.py and in cli.py to have unified cache storage\n\n    Returns\n    -------\n    CacheStorageManager\n        The cache storage manager.\n\n    '
    return LocalDiskCacheStorageManager()