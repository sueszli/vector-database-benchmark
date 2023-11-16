"""
Functions for StringIO objects
"""
import io
readable_types = (io.StringIO,)
writable_types = (io.StringIO,)

def is_stringio(obj):
    if False:
        while True:
            i = 10
    return isinstance(obj, readable_types)

def is_readable(obj):
    if False:
        i = 10
        return i + 15
    return isinstance(obj, readable_types) and obj.readable()

def is_writable(obj):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(obj, writable_types) and obj.writable()