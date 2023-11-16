"""The implementation of `tf.data.Dataset.shuffle`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util.compat import collections_abc

def _scan(input_dataset, initial_state, scan_func, use_default_device=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    return _ScanDataset(input_dataset, initial_state, scan_func, use_default_device, name=name)

class _ScanDataset(dataset_ops.UnaryDataset):
    """A dataset that scans a function across its input."""

    def __init__(self, input_dataset, initial_state, scan_func, use_default_device=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        'See `scan()` for details.'
        self._input_dataset = input_dataset
        self._initial_state = structure.normalize_element(initial_state)
        self._state_structure = structure.type_spec_from_value(self._initial_state)
        need_to_rerun = True
        while need_to_rerun:
            wrapped_func = structured_function.StructuredFunctionWrapper(scan_func, self._transformation_name(), input_structure=(self._state_structure, input_dataset.element_spec), add_to_graph=False)
            if not (isinstance(wrapped_func.output_types, collections_abc.Sequence) and len(wrapped_func.output_types) == 2):
                raise TypeError(f'Invalid `scan_func`. `scan_func` should return a pair consisting of new state and the output value but its return type is {wrapped_func.output_structure}.')
            (new_state_classes, self._output_classes) = wrapped_func.output_classes
            (new_state_classes, output_classes) = wrapped_func.output_classes
            old_state_classes = nest.map_structure(lambda component_spec: component_spec._to_legacy_output_classes(), self._state_structure)
            for (new_state_class, old_state_class) in zip(nest.flatten(new_state_classes), nest.flatten(old_state_classes)):
                if not issubclass(new_state_class, old_state_class):
                    raise TypeError(f'Invalid `scan_func`. The element classes for the new state must match the initial state. Expected {old_state_classes}, got {new_state_classes}.')
            (new_state_types, output_types) = wrapped_func.output_types
            old_state_types = nest.map_structure(lambda component_spec: component_spec._to_legacy_output_types(), self._state_structure)
            for (new_state_type, old_state_type) in zip(nest.flatten(new_state_types), nest.flatten(old_state_types)):
                if new_state_type != old_state_type:
                    raise TypeError(f'Invalid `scan_func`. The element types for the new state must match the initial state. Expected {old_state_types}, got {new_state_types}.')
            (new_state_shapes, output_shapes) = wrapped_func.output_shapes
            old_state_shapes = nest.map_structure(lambda component_spec: component_spec._to_legacy_output_shapes(), self._state_structure)
            self._element_spec = structure.convert_legacy_structure(output_types, output_shapes, output_classes)
            flat_state_shapes = nest.flatten(old_state_shapes)
            flat_new_state_shapes = nest.flatten(new_state_shapes)
            weakened_state_shapes = [original.most_specific_compatible_shape(new) for (original, new) in zip(flat_state_shapes, flat_new_state_shapes)]
            need_to_rerun = False
            for (original_shape, weakened_shape) in zip(flat_state_shapes, weakened_state_shapes):
                if original_shape.ndims is not None and (weakened_shape.ndims is None or original_shape.as_list() != weakened_shape.as_list()):
                    need_to_rerun = True
                    break
            if need_to_rerun:
                self._state_structure = structure.convert_legacy_structure(old_state_types, nest.pack_sequence_as(old_state_shapes, weakened_state_shapes), old_state_classes)
        self._scan_func = wrapped_func
        self._scan_func.function.add_to_graph(ops.get_default_graph())
        self._name = name
        if use_default_device is not None:
            variant_tensor = ged_ops.scan_dataset(self._input_dataset._variant_tensor, structure.to_tensor_list(self._state_structure, self._initial_state), self._scan_func.function.captured_inputs, f=self._scan_func.function, preserve_cardinality=True, use_default_device=use_default_device, **self._common_args)
        else:
            variant_tensor = ged_ops.scan_dataset(self._input_dataset._variant_tensor, structure.to_tensor_list(self._state_structure, self._initial_state), self._scan_func.function.captured_inputs, f=self._scan_func.function, preserve_cardinality=True, **self._common_args)
        super().__init__(input_dataset, variant_tensor)

    def _functions(self):
        if False:
            return 10
        return [self._scan_func]

    @property
    def element_spec(self):
        if False:
            for i in range(10):
                print('nop')
        return self._element_spec

    def _transformation_name(self):
        if False:
            while True:
                i = 10
        return 'Dataset.scan()'