import enum

class CacheType(enum.Enum):
    """The function cache types we implement."""
    DATA = 'DATA'
    RESOURCE = 'RESOURCE'

def get_decorator_api_name(cache_type: CacheType) -> str:
    if False:
        i = 10
        return i + 15
    'Return the name of the public decorator API for the given CacheType.'
    if cache_type is CacheType.DATA:
        return 'cache_data'
    if cache_type is CacheType.RESOURCE:
        return 'cache_resource'
    raise RuntimeError(f"Unrecognized CacheType '{cache_type}'")