"""Registry mechanism implementing the registry pattern for general use."""

class TypeRegistry(object):
    """Provides a type registry for the python registry pattern.

  Contains mappings between types and type specific objects, to implement the
  registry pattern.

  Some example uses of this would be to register different functions depending
  on the type of object.
  """

    def __init__(self):
        if False:
            return 10
        self._registry = {}

    def register(self, obj, value):
        if False:
            for i in range(10):
                print('nop')
        "Registers a Python object within the registry.\n\n    Args:\n      obj: The object to add to the registry.\n      value: The stored value for the 'obj' type.\n\n    Raises:\n      KeyError: If the same obj is used twice.\n    "
        if obj in self._registry:
            raise KeyError(f'{type(obj)} has already been registered.')
        self._registry[obj] = value

    def lookup(self, obj):
        if False:
            return 10
        "Looks up 'obj'.\n\n    Args:\n      obj: The object to lookup within the registry.\n\n    Returns:\n      Value for 'obj' in the registry if found.\n    Raises:\n      LookupError: if 'obj' has not been registered.\n    "
        for registered in self._registry:
            if isinstance(obj, registered):
                return self._registry[registered]
        raise LookupError(f'{type(obj)} has not been registered.')