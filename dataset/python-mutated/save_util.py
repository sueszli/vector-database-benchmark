"""Extracts tensors for checkpointing while updating a TrackableObjectGraph.

The tensors are extracted from `Trackable._serialize_to_tensors`.
"""
import collections
from typing import Any, Callable, List, Optional, Tuple, Mapping, Union, Dict
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.checkpoint import util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.types import core
from tensorflow.python.util import object_identity
_TrackableData = collections.namedtuple('_TrackableData', ['trackable', 'node_id', 'object_name', 'children_proto', 'slot_variable_proto', 'object_to_save'])

def _split_trackables(trackable_data: List[_TrackableData]) -> Tuple[List[_TrackableData], List[_TrackableData], Dict[str, List[_TrackableData]]]:
    if False:
        i = 10
        return i + 15
    'Splits Trackables into 3 categories (tensor/pystate/registered).'
    tensor_trackables = []
    pystate_trackables = []
    registered_trackables = collections.defaultdict(list)
    for td in trackable_data:
        saver_name = registration.get_registered_saver_name(td.object_to_save)
        if isinstance(td.object_to_save, python_state.PythonState):
            pystate_trackables.append(td)
        elif saver_name:
            registered_trackables[saver_name].append(td)
        else:
            tensor_trackables.append(td)
    return (tensor_trackables, pystate_trackables, registered_trackables)

def _gather_trackable_data(graph_view: graph_view_lib.ObjectGraphView, object_map: Mapping[base.Trackable, base.Trackable]) -> Tuple[List[_TrackableData], Dict[base.Trackable, int]]:
    if False:
        while True:
            i = 10
    'Returns a list of generated TrackableData based on the ObjectGraphView.'
    (trackable_objects, node_paths) = graph_view.breadth_first_traversal()
    object_names = object_identity.ObjectIdentityDictionary()
    for (obj, path) in node_paths.items():
        object_names[obj] = trackable_utils.object_path_to_string(path)
    node_ids = object_identity.ObjectIdentityDictionary()
    for (node_id, node) in enumerate(trackable_objects):
        node_ids[node] = node_id
    slot_variables = util.serialize_slot_variables(trackable_objects=trackable_objects, node_ids=node_ids, object_names=object_names)
    trackable_data = []
    for trackable in trackable_objects:
        children_proto = []
        for child in graph_view.list_children(trackable):
            children_proto.append(trackable_object_graph_pb2.TrackableObjectGraph.TrackableObject.ObjectReference(node_id=node_ids[child.ref], local_name=child.name))
        trackable_data.append(_TrackableData(trackable, node_id=node_ids[trackable], object_name=object_names[trackable], children_proto=children_proto, slot_variable_proto=slot_variables.get(trackable, []), object_to_save=util.get_mapped_trackable(trackable, object_map)))
    return (trackable_data, node_ids)

def _fill_object_graph_proto(trackable_data: List[_TrackableData]) -> trackable_object_graph_pb2.TrackableObjectGraph:
    if False:
        return 10
    'Name non-slot `Trackable`s and add them to `object_graph_proto`.'
    object_graph_proto = trackable_object_graph_pb2.TrackableObjectGraph()
    for (checkpoint_id, td) in enumerate(trackable_data):
        assert td.node_id == checkpoint_id
        object_graph_proto.nodes.add(slot_variables=td.slot_variable_proto, children=td.children_proto)
    return object_graph_proto

def _get_and_write_tensors_to_serialize(tensor_trackables: List[_TrackableData], node_ids: Dict[base.Trackable, int], call_with_mapped_captures: Union[Callable[..., Any], None], cache: Union[Dict[base.Trackable, any], None], object_graph_proto: trackable_object_graph_pb2.TrackableObjectGraph) -> Dict[base.Trackable, Any]:
    if False:
        i = 10
        return i + 15
    'Creates dictionary of tensors to checkpoint, and updates the proto.'
    serialized_tensors = object_identity.ObjectIdentityDictionary()
    for td in tensor_trackables:
        if cache is not None and td.object_to_save in cache:
            (trackable, tensor_dict, object_proto) = cache[td.object_to_save]
            serialized_tensors[trackable] = tensor_dict
            object_graph_proto.nodes[td.node_id].attributes.MergeFrom(object_proto)
            continue
        legacy_name = saveable_compat.get_saveable_name(td.object_to_save) or ''
        if not saveable_object_util.trackable_has_serialize_to_tensor(td.object_to_save) or legacy_name:
            (trackable, tensor_dict) = _get_tensors_from_legacy_saveable(td, node_ids, call_with_mapped_captures, object_graph_proto)
        else:
            tensor_dict = _get_tensors_from_trackable(td, call_with_mapped_captures, object_graph_proto)
            trackable = td.object_to_save
        serialized_tensors[trackable] = tensor_dict
        if cache is not None and td.object_to_save not in cache:
            cache[td.object_to_save] = (trackable, tensor_dict, object_graph_proto.nodes[td.node_id].attributes)
    return serialized_tensors

def _get_tensors_from_legacy_saveable(trackable_data: _TrackableData, node_ids: Dict[base.Trackable, int], call_with_mapped_captures: Callable[..., Any], object_graph_proto: trackable_object_graph_pb2.TrackableObjectGraph) -> Tuple[base.Trackable, Dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    'Gets tensors to serialize from a Trackable with legacy SaveableObjects.'
    object_names = object_identity.ObjectIdentityDictionary()
    object_names[trackable_data.trackable] = trackable_data.object_name
    object_map = object_identity.ObjectIdentityDictionary()
    object_map[trackable_data.trackable] = trackable_data.object_to_save
    (checkpoint_factory_map, _) = save_util_v1.get_checkpoint_factories_and_keys(object_names, object_map)
    (named_saveable_objects, _) = save_util_v1.generate_saveable_objects(checkpoint_factory_map, object_graph_proto, node_ids, object_map, call_with_mapped_captures, saveables_cache=None)
    trackable = saveable_object_util.SaveableCompatibilityConverter(trackable_data.object_to_save, named_saveable_objects)
    return (trackable, trackable._serialize_to_tensors())

def _get_tensors_from_trackable(trackable_data: _TrackableData, call_with_mapped_captures: Union[Callable[..., Any], None], object_graph_proto: trackable_object_graph_pb2.TrackableObjectGraph) -> Dict[str, Any]:
    if False:
        return 10
    'Gets tensors to serialize from a Trackable.'
    trackable = trackable_data.object_to_save
    save_fn = trackable._serialize_to_tensors
    if call_with_mapped_captures and isinstance(save_fn, core.ConcreteFunction):
        ret_tensor_dict = call_with_mapped_captures(save_fn, [])
    else:
        ret_tensor_dict = save_fn()
    tensor_dict = {}
    for (tensor_name, maybe_tensor) in ret_tensor_dict.items():
        local_name = trackable_utils.escape_local_name(tensor_name)
        checkpoint_key = trackable_utils.checkpoint_key(trackable_data.object_name, local_name)
        tensor_dict[checkpoint_key] = maybe_tensor
        if isinstance(maybe_tensor, saveable_object_lib.SaveSpec):
            maybe_tensor.name = checkpoint_key
            maybe_tensor.slice_spec = ''
        if object_graph_proto is not None:
            object_graph_proto.nodes[trackable_data.node_id].attributes.add(name=local_name, checkpoint_key=checkpoint_key, full_name=util.get_full_name(trackable))
    return tensor_dict

def _get_and_write_pystate_feed_additions(pystate_trackables: List[_TrackableData], cache: Union[Dict[base.Trackable, Any], None], object_graph_proto=None) -> Tuple[Dict[base.Trackable, Any], Dict[base.Trackable, Any]]:
    if False:
        print('Hello World!')
    'Gets feed additions needed for checkpointing Python State.'
    serialized_tensors = object_identity.ObjectIdentityDictionary()
    feed_additions = {}
    for td in pystate_trackables:
        trackable = td.object_to_save
        checkpoint_key = trackable_utils.checkpoint_key(td.object_name, python_state.PYTHON_STATE)
        if trackable in cache:
            save_string = cache[td.object_to_save][python_state.PYTHON_STATE]
        else:
            with ops.device('/cpu:0'):
                save_string = constant_op.constant('', dtype=dtypes.string)
                cache[trackable] = {python_state.PYTHON_STATE: save_string}
        with ops.init_scope():
            value = trackable.serialize()
        feed_additions[save_string] = value
        serialized_tensors[trackable] = {checkpoint_key: save_string}
        object_graph_proto.nodes[td.node_id].attributes.add(name=python_state.PYTHON_STATE, checkpoint_key=checkpoint_key, full_name=util.get_full_name(trackable))
    return (serialized_tensors, feed_additions)

def _get_and_write_registered_savers(registered_trackables: Dict[str, List[_TrackableData]], object_graph_proto: trackable_object_graph_pb2.TrackableObjectGraph) -> Dict[str, Dict[str, base.Trackable]]:
    if False:
        while True:
            i = 10
    'Generates dictionary of registered savers and updates the proto.'
    registered_savers = collections.defaultdict(dict)
    for (saver_name, trackables) in registered_trackables.items():
        for td in trackables:
            registered_savers[saver_name][td.object_name] = td.object_to_save
            object_proto = object_graph_proto.nodes[td.node_id]
            object_proto.registered_saver.name = saver_name
            object_proto.registered_saver.object_name = td.object_name
    return registered_savers

def serialize_graph_view(graph_view: graph_view_lib.ObjectGraphView, object_map: Optional[Mapping[base.Trackable, base.Trackable]]=None, call_with_mapped_captures: Optional[Callable[..., Any]]=None, cache: Optional[Dict[base.Trackable, Any]]=None) -> ...:
    if False:
        i = 10
        return i + 15
    'Gathers serialization objects, and creates a TrackableObjectGraph proto.'
    (trackable_data, node_ids) = _gather_trackable_data(graph_view, object_map)
    (tensor_trackables, pystate_trackables, registered_trackables) = _split_trackables(trackable_data)
    object_graph_proto = _fill_object_graph_proto(trackable_data)
    serialized_tensors = _get_and_write_tensors_to_serialize(tensor_trackables, node_ids, call_with_mapped_captures, cache, object_graph_proto)
    registered_savers = _get_and_write_registered_savers(registered_trackables, object_graph_proto)
    if cache is None:
        feed_additions = None
        serialized_tensors.update(_get_and_write_tensors_to_serialize(pystate_trackables, node_ids, call_with_mapped_captures, cache, object_graph_proto))
    else:
        (new_serialized_tensors, feed_additions) = _get_and_write_pystate_feed_additions(pystate_trackables, cache, object_graph_proto)
        serialized_tensors.update(new_serialized_tensors)
    util.add_checkpoint_values_check(object_graph_proto)
    return (serialized_tensors, feed_additions, registered_savers, object_graph_proto)