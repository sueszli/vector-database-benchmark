"""
When frozen, a module that dynamically recreates itself at runtime (by replacing itself in sys.modules) should be
returned by __import__ statement.

This example should return True:
    >>> sys.modules[<dynamic_module>] is __import__(<dynamic_module>)
    True
"""
import sys
import types
foo = None

class DynamicModule(types.ModuleType):
    __file__ = __file__

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name)
        self.foo = 'A new value!'
sys.modules[__name__] = DynamicModule(__name__)