import os
import threading
from contextlib import contextmanager
from functools import wraps
from ray._private.auto_init_hook import auto_init_ray
RAY_CLIENT_MODE_ATTR = '__ray_client_mode_key__'
is_client_mode_enabled = os.environ.get('RAY_CLIENT_MODE', '0') == '1'
is_client_mode_enabled_by_default = is_client_mode_enabled
os.environ.update({'RAY_CLIENT_MODE': '0'})
is_init_called = False
_client_hook_status_on_thread = threading.local()
_client_hook_status_on_thread.status = True

def _get_client_hook_status_on_thread():
    if False:
        i = 10
        return i + 15
    "Get's the value of `_client_hook_status_on_thread`.\n    Since `_client_hook_status_on_thread` is a thread-local variable, we may\n    need to add and set the 'status' attribute.\n    "
    global _client_hook_status_on_thread
    if not hasattr(_client_hook_status_on_thread, 'status'):
        _client_hook_status_on_thread.status = True
    return _client_hook_status_on_thread.status

def _set_client_hook_status(val: bool):
    if False:
        for i in range(10):
            print('nop')
    global _client_hook_status_on_thread
    _client_hook_status_on_thread.status = val

def _disable_client_hook():
    if False:
        while True:
            i = 10
    global _client_hook_status_on_thread
    out = _get_client_hook_status_on_thread()
    _client_hook_status_on_thread.status = False
    return out

def _explicitly_enable_client_mode():
    if False:
        while True:
            i = 10
    'Force client mode to be enabled.\n    NOTE: This should not be used in tests, use `enable_client_mode`.\n    '
    global is_client_mode_enabled
    is_client_mode_enabled = True

def _explicitly_disable_client_mode():
    if False:
        for i in range(10):
            print('nop')
    global is_client_mode_enabled
    is_client_mode_enabled = False

@contextmanager
def disable_client_hook():
    if False:
        while True:
            i = 10
    val = _disable_client_hook()
    try:
        yield None
    finally:
        _set_client_hook_status(val)

@contextmanager
def enable_client_mode():
    if False:
        for i in range(10):
            print('nop')
    _explicitly_enable_client_mode()
    try:
        yield None
    finally:
        _explicitly_disable_client_mode()

def client_mode_hook(func: callable):
    if False:
        return 10
    "Decorator for whether to use the 'regular' ray version of a function,\n    or the Ray Client version of that function.\n\n    Args:\n        func: This function. This is set when this function is used\n            as a decorator.\n    "
    from ray.util.client import ray

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            return 10
        if client_mode_should_convert():
            if func.__name__ != 'init' or is_client_mode_enabled_by_default:
                return getattr(ray, func.__name__)(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

def client_mode_should_convert():
    if False:
        for i in range(10):
            print('nop')
    'Determines if functions should be converted to client mode.'
    return (is_client_mode_enabled or is_client_mode_enabled_by_default) and _get_client_hook_status_on_thread()

def client_mode_wrap(func):
    if False:
        i = 10
        return i + 15
    'Wraps a function called during client mode for execution as a remote\n    task.\n\n    Can be used to implement public features of ray client which do not\n    belong in the main ray API (`ray.*`), yet require server-side execution.\n    An example is the creation of placement groups:\n    `ray.util.placement_group.placement_group()`. When called on the client\n    side, this function is wrapped in a task to facilitate interaction with\n    the GCS.\n    '

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        from ray.util.client import ray
        auto_init_ray()
        if client_mode_should_convert():
            f = ray.remote(num_cpus=0)(func)
            ref = f.remote(*args, **kwargs)
            return ray.get(ref)
        return func(*args, **kwargs)
    return wrapper

def client_mode_convert_function(func_cls, in_args, in_kwargs, **kwargs):
    if False:
        while True:
            i = 10
    'Runs a preregistered ray RemoteFunction through the ray client.\n\n    The common case for this is to transparently convert that RemoteFunction\n    to a ClientRemoteFunction. This happens in circumstances where the\n    RemoteFunction is declared early, in a library and only then is Ray used in\n    client mode -- necessitating a conversion.\n    '
    from ray.util.client import ray
    key = getattr(func_cls, RAY_CLIENT_MODE_ATTR, None)
    if key is None or not ray._converted_key_exists(key):
        key = ray._convert_function(func_cls)
        setattr(func_cls, RAY_CLIENT_MODE_ATTR, key)
    client_func = ray._get_converted(key)
    return client_func._remote(in_args, in_kwargs, **kwargs)

def client_mode_convert_actor(actor_cls, in_args, in_kwargs, **kwargs):
    if False:
        print('Hello World!')
    'Runs a preregistered actor class on the ray client\n\n    The common case for this decorator is for instantiating an ActorClass\n    transparently as a ClientActorClass. This happens in circumstances where\n    the ActorClass is declared early, in a library and only then is Ray used in\n    client mode -- necessitating a conversion.\n    '
    from ray.util.client import ray
    key = getattr(actor_cls, RAY_CLIENT_MODE_ATTR, None)
    if key is None or not ray._converted_key_exists(key):
        key = ray._convert_actor(actor_cls)
        setattr(actor_cls, RAY_CLIENT_MODE_ATTR, key)
    client_actor = ray._get_converted(key)
    return client_actor._remote(in_args, in_kwargs, **kwargs)