import os
from .encoding import system_decode as decode, system_encode as encode

def get_env_var(name, default=None):
    if False:
        return 10
    try:
        value = os.environ[encode(name)]
    except KeyError:
        return default
    else:
        return decode(value)

def set_env_var(name, value):
    if False:
        i = 10
        return i + 15
    os.environ[encode(name)] = encode(value)

def del_env_var(name):
    if False:
        i = 10
        return i + 15
    value = get_env_var(name)
    if value is not None:
        del os.environ[encode(name)]
    return value

def get_env_vars(upper=os.sep != '/'):
    if False:
        while True:
            i = 10
    return dict(((name if not upper else name.upper(), get_env_var(name)) for name in (decode(name) for name in os.environ)))