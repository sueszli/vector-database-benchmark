from typing import Any, Callable, Dict, TypeVar
ReturnT = TypeVar('ReturnT')
FUNCTION_NAME_TO_PER_REQUEST_RESULT: Dict[str, Dict[int, Any]] = {}

def return_same_value_during_entire_request(f: Callable[..., ReturnT]) -> Callable[..., ReturnT]:
    if False:
        return 10
    cache_key = f.__name__
    assert cache_key not in FUNCTION_NAME_TO_PER_REQUEST_RESULT
    FUNCTION_NAME_TO_PER_REQUEST_RESULT[cache_key] = {}

    def wrapper(key: int, *args: Any) -> ReturnT:
        if False:
            return 10
        if key in FUNCTION_NAME_TO_PER_REQUEST_RESULT[cache_key]:
            return FUNCTION_NAME_TO_PER_REQUEST_RESULT[cache_key][key]
        result = f(key, *args)
        FUNCTION_NAME_TO_PER_REQUEST_RESULT[cache_key][key] = result
        return result
    return wrapper

def flush_per_request_cache(cache_key: str) -> None:
    if False:
        while True:
            i = 10
    if cache_key in FUNCTION_NAME_TO_PER_REQUEST_RESULT:
        FUNCTION_NAME_TO_PER_REQUEST_RESULT[cache_key] = {}

def flush_per_request_caches() -> None:
    if False:
        return 10
    for cache_key in FUNCTION_NAME_TO_PER_REQUEST_RESULT:
        FUNCTION_NAME_TO_PER_REQUEST_RESULT[cache_key] = {}