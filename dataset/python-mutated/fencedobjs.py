"""Fenced objects are wraps for raising TranspilerError when they are modified."""
from qiskit.utils.deprecation import deprecate_func
from .exceptions import TranspilerError

class FencedObject:
    """Given an instance and a list of attributes to fence, raises a TranspilerError when one
    of these attributes is accessed."""

    @deprecate_func(since='0.45.0', additional_msg='Internal use of FencedObject is already removed from pass manager. Implementation of a task subclass with protection for input object modification is now responsibility of the developer.', pending=True)
    def __init__(self, instance, attributes_to_fence):
        if False:
            print('Hello World!')
        self._wrapped = instance
        self._attributes_to_fence = attributes_to_fence

    def __getattribute__(self, name):
        if False:
            return 10
        object.__getattribute__(self, '_check_if_fenced')(name)
        return getattr(object.__getattribute__(self, '_wrapped'), name)

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        object.__getattribute__(self, '_check_if_fenced')('__getitem__')
        return object.__getattribute__(self, '_wrapped')[key]

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        object.__getattribute__(self, '_check_if_fenced')('__setitem__')
        object.__getattribute__(self, '_wrapped')[key] = value

    def _check_if_fenced(self, name):
        if False:
            while True:
                i = 10
        '\n        Checks if the attribute name is in the list of attributes to protect. If so, raises\n        TranspilerError.\n\n        Args:\n            name (string): the attribute name to check\n\n        Raises:\n            TranspilerError: when name is the list of attributes to protect.\n        '
        if name in object.__getattribute__(self, '_attributes_to_fence'):
            raise TranspilerError('The fenced %s has the property %s protected' % (type(object.__getattribute__(self, '_wrapped')), name))

class FencedPropertySet(FencedObject):
    """A property set that cannot be written (via __setitem__)"""

    def __init__(self, property_set_instance):
        if False:
            while True:
                i = 10
        super().__init__(property_set_instance, ['__setitem__'])

class FencedDAGCircuit(FencedObject):
    """A dag circuit that cannot be modified (via remove_op_node)"""

    def __init__(self, dag_circuit_instance):
        if False:
            return 10
        super().__init__(dag_circuit_instance, ['remove_op_node'])