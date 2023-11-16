"""A dummy implementation for use in RegistryTest."""
from syntaxnet.util import registry_test_base

class Impl(registry_test_base.Base):
    """Dummy implementation."""

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        'Creates an implementation with a custom string.'
        self.value = value

    def Get(self):
        if False:
            while True:
                i = 10
        'Returns the current value.'
        return self.value
Alias = Impl

class NonSubclass(object):
    """A class that is not a subclass of registry_test_base.Base."""
    pass
variable = 1

def Function():
    if False:
        i = 10
        return i + 15
    'A dummy function, to exercise type checking.'
    pass