"""Manages a graph of Trackable objects."""
import copy
import weakref
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.trackable import base
from tensorflow.python.util.tf_export import tf_export

@tf_export('__internal__.tracking.ObjectGraphView', v1=[])
class ObjectGraphView(trackable_view.TrackableView):
    """Gathers and serializes an object graph."""

    def __init__(self, root, attached_dependencies=None):
        if False:
            return 10
        'Configure the graph view.\n\n    Args:\n      root: A `Trackable` object whose variables (including the variables of\n        dependencies, recursively) should be saved. May be a weak reference.\n      attached_dependencies: List of dependencies to attach to the root object.\n        Used when saving a Checkpoint with a defined root object. To avoid\n        reference cycles, this should use the WeakTrackableReference class.\n    '
        trackable_view.TrackableView.__init__(self, root)
        self._root_ref = root if isinstance(root, weakref.ref) else weakref.ref(root)
        self._attached_dependencies = attached_dependencies

    def __deepcopy__(self, memo):
        if False:
            print('Hello World!')
        strong_root = self._root_ref()
        if strong_root is not None:
            strong_copy = copy.deepcopy(strong_root, memo)
            memo[id(self._root_ref)] = weakref.ref(strong_copy)
        copied = super().__new__(type(self))
        memo[id(self)] = copied
        for (key, value) in vars(self).items():
            setattr(copied, key, copy.deepcopy(value, memo))
        return copied

    def list_children(self, obj, save_type=base.SaveType.CHECKPOINT, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Returns list of all child trackables attached to obj.\n\n    Args:\n      obj: A `Trackable` object.\n      save_type: A string, can be 'savedmodel' or 'checkpoint'.\n      **kwargs: kwargs to use when retrieving the object's children.\n\n    Returns:\n      List of all children attached to the object.\n    "
        children = []
        for (name, ref) in super(ObjectGraphView, self).children(obj, save_type, **kwargs).items():
            children.append(base.TrackableReference(name, ref))
        if obj is self.root and self._attached_dependencies:
            children.extend(self._attached_dependencies)
        return children

    def children(self, obj, save_type=base.SaveType.CHECKPOINT, **kwargs):
        if False:
            i = 10
            return i + 15
        "Returns all child trackables attached to obj.\n\n    Args:\n      obj: A `Trackable` object.\n      save_type: A string, can be 'savedmodel' or 'checkpoint'.\n      **kwargs: kwargs to use when retrieving the object's children.\n\n    Returns:\n      Dictionary of all children attached to the object with name to trackable.\n    "
        children = {}
        for (name, ref) in self.list_children(obj, **kwargs):
            children[name] = ref
        return children

    @property
    def attached_dependencies(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns list of dependencies that should be saved in the checkpoint.\n\n    These dependencies are not tracked by root, but are in the checkpoint.\n    This is defined when the user creates a Checkpoint with both root and kwargs\n    set.\n\n    Returns:\n      A list of TrackableReferences.\n    '
        return self._attached_dependencies

    @property
    def root(self):
        if False:
            while True:
                i = 10
        if isinstance(self._root_ref, weakref.ref):
            derefed = self._root_ref()
            assert derefed is not None
            return derefed
        else:
            return self._root_ref

    def breadth_first_traversal(self):
        if False:
            while True:
                i = 10
        return self._breadth_first_traversal()

    def _breadth_first_traversal(self):
        if False:
            return 10
        'Find shortest paths to all dependencies of self.root.'
        return super(ObjectGraphView, self)._descendants_with_paths()

    def serialize_object_graph(self, saveables_cache=None):
        if False:
            for i in range(10):
                print('nop')
        "Determine checkpoint keys for variables and build a serialized graph.\n\n    Non-slot variables are keyed based on a shortest path from the root saveable\n    to the object which owns the variable (i.e. the one which called\n    `Trackable._add_variable` to create it).\n\n    Slot variables are keyed based on a shortest path to the variable being\n    slotted for, a shortest path to their optimizer, and the slot name.\n\n    Args:\n      saveables_cache: An optional cache storing previously created\n        SaveableObjects created for each Trackable. Maps Trackables to a\n        dictionary of attribute names to Trackable.\n\n    Returns:\n      A tuple of (named_variables, object_graph_proto, feed_additions):\n        named_variables: A dictionary mapping names to variable objects.\n        object_graph_proto: A TrackableObjectGraph protocol buffer\n          containing the serialized object graph and variable references.\n        feed_additions: A dictionary mapping from Tensors to values which should\n          be fed when saving.\n\n    Raises:\n      ValueError: If there are invalid characters in an optimizer's slot names.\n    "
        (named_saveable_objects, object_graph_proto, feed_additions, _) = save_util_v1.serialize_object_graph_with_registered_savers(self, saveables_cache)
        return (named_saveable_objects, object_graph_proto, feed_additions)

    def frozen_saveable_objects(self, object_map=None, to_graph=None, call_with_mapped_captures=None):
        if False:
            for i in range(10):
                print('nop')
        'Creates SaveableObjects with the current object graph frozen.'
        return save_util_v1.frozen_saveables_and_savers(self, object_map, to_graph, call_with_mapped_captures)[0]