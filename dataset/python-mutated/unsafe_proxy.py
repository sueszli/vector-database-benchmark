from __future__ import annotations
from collections.abc import Mapping, Set
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import binary_type, text_type
from ansible.utils.native_jinja import NativeJinjaText
__all__ = ['AnsibleUnsafe', 'wrap_var']

class AnsibleUnsafe(object):
    __UNSAFE__ = True

class AnsibleUnsafeBytes(binary_type, AnsibleUnsafe):

    def decode(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Wrapper method to ensure type conversions maintain unsafe context'
        return AnsibleUnsafeText(super(AnsibleUnsafeBytes, self).decode(*args, **kwargs))

class AnsibleUnsafeText(text_type, AnsibleUnsafe):

    def encode(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Wrapper method to ensure type conversions maintain unsafe context'
        return AnsibleUnsafeBytes(super(AnsibleUnsafeText, self).encode(*args, **kwargs))

class NativeJinjaUnsafeText(NativeJinjaText, AnsibleUnsafeText):
    pass

def _wrap_dict(v):
    if False:
        return 10
    return dict(((wrap_var(k), wrap_var(item)) for (k, item) in v.items()))

def _wrap_sequence(v):
    if False:
        return 10
    'Wraps a sequence with unsafe, not meant for strings, primarily\n    ``tuple`` and ``list``\n    '
    v_type = type(v)
    return v_type((wrap_var(item) for item in v))

def _wrap_set(v):
    if False:
        while True:
            i = 10
    return set((wrap_var(item) for item in v))

def wrap_var(v):
    if False:
        while True:
            i = 10
    if v is None or isinstance(v, AnsibleUnsafe):
        return v
    if isinstance(v, Mapping):
        v = _wrap_dict(v)
    elif isinstance(v, Set):
        v = _wrap_set(v)
    elif is_sequence(v):
        v = _wrap_sequence(v)
    elif isinstance(v, NativeJinjaText):
        v = NativeJinjaUnsafeText(v)
    elif isinstance(v, binary_type):
        v = AnsibleUnsafeBytes(v)
    elif isinstance(v, text_type):
        v = AnsibleUnsafeText(v)
    return v

def to_unsafe_bytes(*args, **kwargs):
    if False:
        while True:
            i = 10
    return wrap_var(to_bytes(*args, **kwargs))

def to_unsafe_text(*args, **kwargs):
    if False:
        print('Hello World!')
    return wrap_var(to_text(*args, **kwargs))