"""Bisection lookup multiple keys."""
from __future__ import absolute_import
__all__ = ['bisect_multi_bytes']

def bisect_multi_bytes(content_lookup, size, keys):
    if False:
        while True:
            i = 10
    'Perform bisection lookups for keys using byte based addressing.\n\n    The keys are looked up via the content_lookup routine. The content_lookup\n    routine gives bisect_multi_bytes information about where to keep looking up\n    to find the data for the key, and bisect_multi_bytes feeds this back into\n    the lookup function until the search is complete. The search is complete\n    when the list of keys which have returned something other than -1 or +1 is\n    empty. Keys which are not found are not returned to the caller.\n\n    :param content_lookup: A callable that takes a list of (offset, key) pairs\n        and returns a list of result tuples ((offset, key), result). Each\n        result can be one of:\n          -1: The key comes earlier in the content.\n          False: The key is not present in the content.\n          +1: The key comes later in the content.\n          Any other value: A final result to return to the caller.\n    :param size: The length of the content.\n    :param keys: The keys to bisect for.\n    :return: An iterator of the results.\n    '
    result = []
    delta = size // 2
    search_keys = [(delta, key) for key in keys]
    while search_keys:
        search_results = content_lookup(search_keys)
        if delta > 1:
            delta = delta // 2
        search_keys = []
        for ((location, key), status) in search_results:
            if status == -1:
                search_keys.append((location - delta, key))
            elif status == 1:
                search_keys.append((location + delta, key))
            elif status == False:
                continue
            else:
                result.append((key, status))
    return result