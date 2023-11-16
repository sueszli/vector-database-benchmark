"""NoneTensor and NoneTensorSpec classes."""
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry

class NoneTensor(composite_tensor.CompositeTensor):
    """Composite tensor representation for `None` value."""

    @property
    def _type_spec(self):
        if False:
            return 10
        return NoneTensorSpec()

@type_spec_registry.register('tf.NoneTensorSpec')
class NoneTensorSpec(type_spec.BatchableTypeSpec):
    """Type specification for `None` value."""

    @property
    def value_type(self):
        if False:
            return 10
        return NoneTensor

    def _serialize(self):
        if False:
            while True:
                i = 10
        return ()

    @property
    def _component_specs(self):
        if False:
            while True:
                i = 10
        return []

    def _to_components(self, value):
        if False:
            for i in range(10):
                print('nop')
        return []

    def _from_components(self, components):
        if False:
            for i in range(10):
                print('nop')
        return

    def _to_tensor_list(self, value):
        if False:
            i = 10
            return i + 15
        return []

    @staticmethod
    def from_value(value):
        if False:
            i = 10
            return i + 15
        return NoneTensorSpec()

    def _batch(self, batch_size):
        if False:
            for i in range(10):
                print('nop')
        return NoneTensorSpec()

    def _unbatch(self):
        if False:
            for i in range(10):
                print('nop')
        return NoneTensorSpec()

    def _to_batched_tensor_list(self, value):
        if False:
            i = 10
            return i + 15
        return []

    def _to_legacy_output_types(self):
        if False:
            return 10
        return self

    def _to_legacy_output_shapes(self):
        if False:
            return 10
        return self

    def _to_legacy_output_classes(self):
        if False:
            return 10
        return self

    def most_specific_compatible_shape(self, other):
        if False:
            return 10
        if type(self) is not type(other):
            raise ValueError('No `TypeSpec` is compatible with both {} and {}'.format(self, other))
        return self
type_spec.register_type_spec_from_value_converter(type(None), NoneTensorSpec.from_value)