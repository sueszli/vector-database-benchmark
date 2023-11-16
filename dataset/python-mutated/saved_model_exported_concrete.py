"""ExportedConcreteFunction class and its associated functions.

Part of saved model utils, a shim layer for working with
functions exported/restored from saved models.
This functionality should ultimately be moved into a first-class core API.
"""
import gc
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.trackable import base as trackable

class ExportedConcreteFunction(trackable.Trackable):
    """A callable class that uses captures from the exported SavedModel graph."""
    __slots__ = ('function', 'tensor_map')

    def __init__(self, function, tensor_map):
        if False:
            return 10
        self.function = function
        self.tensor_map = tensor_map

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        bound_arguments = function_type_utils.canonicalize_function_inputs(args, kwargs, self.function._function_type)
        filtered_flat_args = self.function._function_type.unpack_inputs(bound_arguments)
        export_captures = _map_captures_to_created_tensors(self.function.graph.captures, self.tensor_map, self.function)
        return self.function._call_flat(filtered_flat_args, export_captures)

def _map_captures_to_created_tensors(original_captures, tensor_map, function):
    if False:
        while True:
            i = 10
    "Maps eager tensors captured by a function to Graph resources for export.\n\n  Args:\n    original_captures: A dictionary mapping from tensors captured by the\n      function to interior placeholders for those tensors (inside the function\n      body).\n    tensor_map: A dictionary mapping from resource tensors owned by the eager\n      context to resource tensors in the exported graph.\n    function: Function with the original captures. Only used when raising the\n      AssertionError.\n\n  Returns:\n    A list of stand-in tensors which belong to the exported graph, corresponding\n    to the function's captures.\n\n  Raises:\n    AssertionError: If the function references a resource which is not part of\n      `tensor_map`.\n  "
    export_captures = []
    for (exterior, interior) in original_captures:
        mapped_resource = tensor_map.get(exterior, None)
        if mapped_resource is None:
            _raise_untracked_capture_error(function.name, exterior, interior)
        export_captures.append(mapped_resource)
    return export_captures

def _raise_untracked_capture_error(function_name, capture, internal_capture=None, node_path=None):
    if False:
        i = 10
        return i + 15
    'Raises AssertionError due to being unable to export a function.'
    msg = f"Tried to export a function which references an 'untracked' resource. TensorFlow objects (e.g. tf.Variable) captured by functions must be 'tracked' by assigning them to an attribute of a tracked object or assigned to an attribute of the main object directly. See the information below:\n\tFunction name = {function_name}"
    if node_path is not None:
        msg += f'\n\tPath to Function = {node_path}'
    msg += f'\n\tCaptured Tensor = {capture}'
    msg += f'\n\t{_get_trackable_parent_error_string(capture)}'
    if internal_capture is not None:
        msg += f'\n\tInternal Tensor = {internal_capture}'
    raise AssertionError(msg)

def _get_trackable_parent_error_string(capture):
    if False:
        print('Hello World!')
    "Gets error string with the capture's parent object."
    parent = getattr(capture, '_parent_trackable', None)
    if parent is not None:
        return f'Trackable referencing this tensor = {parent()}'
    trackable_referrers = []
    for primary_referrer in gc.get_referrers(capture):
        if isinstance(primary_referrer, trackable.Trackable):
            trackable_referrers.append(primary_referrer)
        for secondary_referrer in gc.get_referrers(primary_referrer):
            if isinstance(secondary_referrer, trackable.Trackable):
                trackable_referrers.append(secondary_referrer)
    return 'Trackable Python objects referring to this tensor (from gc.get_referrers, limited to two hops) = [\n\t\t{}]'.format('\n\t\t'.join([repr(obj) for obj in trackable_referrers]))