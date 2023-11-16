"""ModuleImpl interface.

In order to be able to expand the types of Modules that are supported without
users having to call the right constructor we use a "pointer-to-implementation"
pattern:

`Module` is the public API class that every user should instantiate. It's
constructor uses `spec` to create a `ModuleImpl` that encapsulates each specific
implementation.
"""
import abc

class ModuleImpl(object):
    """Internal module implementation interface."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def create_apply_graph(self, signature, input_tensors, name):
        if False:
            while True:
                i = 10
        'Applies the module signature to inputs.\n\n    Args:\n      signature: A string with the signature to create.\n      input_tensors: A dictionary of tensors with the inputs.\n      name: A name scope under which to instantiate the signature.\n\n    Returns:\n      A dictionary of output tensors from applying the signature.\n    '
        raise NotImplementedError()

    def get_signature_name(self, signature):
        if False:
            i = 10
            return i + 15
        'Resolves a signature name.'
        if not signature:
            return 'default'
        return signature

    @abc.abstractmethod
    def export(self, path, session):
        if False:
            for i in range(10):
                print('nop')
        'See `Module.export()`.'
        raise NotImplementedError()

    @abc.abstractproperty
    def variable_map(self):
        if False:
            for i in range(10):
                print('nop')
        'See `Module.variable_map`.'
        raise NotImplementedError()