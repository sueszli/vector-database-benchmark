"""A dummy base class for use in RegistryTest."""
from syntaxnet.util import registry

@registry.RegisteredClass
class Base(object):
    """Dummy base class."""

    def Get(self):
        if False:
            i = 10
            return i + 15
        'Overridden in subclasses.'
        return None