from functools import lru_cache, wraps
from django.http import HttpRequest

def invoker(f, *args, **kwargs):
    if False:
        while True:
            i = 10
    f(*args, **kwargs)

@lru_cache
def cached_eval(s: str) -> None:
    if False:
        return 10
    eval(s)

def test_cached_eval(request: HttpRequest):
    if False:
        i = 10
        return i + 15
    cached_eval(request.GET['bad'])

def test_cached_eval_inner(request: HttpRequest):
    if False:
        i = 10
        return i + 15

    @lru_cache
    def inner(s: str) -> None:
        if False:
            i = 10
            return i + 15
        eval(s)
    inner(request.GET['bad'])

def test_cached_eval_higher_order_function(request: HttpRequest):
    if False:
        print('Hello World!')

    @lru_cache
    def inner(s: str) -> None:
        if False:
            i = 10
            return i + 15
        eval(s)
    invoker(inner, request.GET['bad'])

@lru_cache(maxsize=10)
def cached_eval_with_maxsize(s: str) -> None:
    if False:
        i = 10
        return i + 15
    eval(s)

def test_cached_eval_with_maxsize(request: HttpRequest):
    if False:
        return 10
    cached_eval_with_maxsize(request.GET['bad'])

@lru_cache
def cached_sanitizer(x):
    if False:
        return 10
    return x

def test_cached_sanitizer(request: HttpRequest) -> None:
    if False:
        for i in range(10):
            print('nop')
    sanitized = cached_sanitizer(request.GET['bad'])
    eval(sanitized)

def test_wraps(f, request: HttpRequest):
    if False:
        i = 10
        return i + 15

    @wraps(f)
    def wrapper():
        if False:
            return 10
        return f(request.GET['bad'])
    eval(wrapper())