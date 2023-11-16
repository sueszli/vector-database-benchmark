"""
categories: Core,Functions
description: Function objects do not have the ``__module__`` attribute
cause: MicroPython is optimized for reduced code size and RAM usage.
workaround: Use ``sys.modules[function.__globals__['__name__']]`` for non-builtin modules.
"""

def f():
    if False:
        print('Hello World!')
    pass
print(f.__module__)