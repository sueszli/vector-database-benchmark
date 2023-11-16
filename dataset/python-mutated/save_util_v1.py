"""Extracts tensors for checkpointing while updating a TrackableObjectGraph.

This is labelled "v1" because the methods here use SaveableObject, which will
soon be deprecated.
"""
import collections
from tensorflow.core.protobuf import trackable_object_graph_pb2
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
from tensorflow.python.util import object_identity
_CheckpointFactoryData = collections.namedtuple('_CheckpointFactoryData', ['factory', 'name', 'checkpoint_key'])

def get_checkpoint_factories_and_keys(object_names, object_map=None):
    if False:
        return 10
    'Gets a map of saveable factories and corresponding checkpoint keys.\n\n  Args:\n    object_names: a dictionary that maps `Trackable` objects to auto-generated\n      string names.\n    object_map: a dictionary mapping `Trackable` to copied `Trackable` objects.\n      The copied objects are generated from `Trackable.\n      _export_to_saved_model_graph()` which copies the object into another\n      graph. Generally only resource objects (e.g. Variables, Tables) will be\n      in this map.\n\n  Returns:\n    A tuple of (\n      Dictionary mapping trackable -> list of _CheckpointFactoryData,\n      Dictionary mapping registered saver name -> {object name -> trackable})\n  '
    checkpoint_factory_map = object_identity.ObjectIdentityDictionary()
    unmapped_registered_savers = collections.defaultdict(dict)
    for (trackable, object_name) in object_names.items():
        object_to_save = util.get_mapped_trackable(trackable, object_map)
        saver_name = registration.get_registered_saver_name(object_to_save)
        if saver_name:
            unmapped_registered_savers[saver_name][object_name] = trackable
        else:
            checkpoint_factory_map[trackable] = []
            for (name, saveable_factory) in saveable_object_util.saveable_objects_from_trackable(object_to_save).items():
                key_suffix = saveable_compat.get_saveable_name(object_to_save) or name
                checkpoint_key = trackable_utils.checkpoint_key(object_name, key_suffix)
                if not saveable_compat.force_checkpoint_conversion_enabled():
                    name = key_suffix
                checkpoint_factory_map[trackable].append(_CheckpointFactoryData(factory=saveable_factory, name=name, checkpoint_key=checkpoint_key))
    return (checkpoint_factory_map, unmapped_registered_savers)

def _add_attributes_to_object_graph(trackable_objects, object_graph_proto, node_ids, object_names, object_map, call_with_mapped_captures, saveables_cache):
    if False:
        for i in range(10):
            print('nop')
    'Create saveables/savers and corresponding protos in the object graph.'
    for (checkpoint_id, (trackable, unused_object_proto)) in enumerate(zip(trackable_objects, object_graph_proto.nodes)):
        assert node_ids[trackable] == checkpoint_id
    (checkpoint_factory_map, unmapped_registered_savers) = get_checkpoint_factories_and_keys(object_names, object_map)
    registered_savers = _add_attributes_to_object_graph_for_registered_savers(unmapped_registered_savers, object_graph_proto, node_ids, object_map)
    (named_saveable_objects, feed_additions) = generate_saveable_objects(checkpoint_factory_map, object_graph_proto, node_ids, object_map, call_with_mapped_captures, saveables_cache)
    return (named_saveable_objects, feed_additions, registered_savers)

def _add_attributes_to_object_graph_for_registered_savers(unmapped_registered_savers, object_graph_proto, node_ids, object_map):
    if False:
        print('Hello World!')
    'Fills the object graph proto with data about the registered savers.'
    registered_savers = collections.defaultdict(dict)
    for (saver_name, trackables) in unmapped_registered_savers.items():
        for (object_name, trackable) in trackables.items():
            object_proto = object_graph_proto.nodes[node_ids[trackable]]
            object_proto.registered_saver.name = saver_name
            object_proto.registered_saver.object_name = object_name
            object_to_save = util.get_mapped_trackable(trackable, object_map)
            registered_savers[saver_name][object_name] = object_to_save
    return registered_savers

def generate_saveable_objects(checkpoint_factory_map, object_graph_proto=None, node_ids=None, object_map=None, call_with_mapped_captures=None, saveables_cache=None):
    if False:
        i = 10
        return i + 15
    'Create SaveableObjects and corresponding SerializedTensor protos.'
    named_saveable_objects = []
    if saveables_cache is None:
        feed_additions = None
    else:
        feed_additions = {}
    for (trackable, factory_data_list) in checkpoint_factory_map.items():
        fill_object_proto = object_graph_proto is not None and node_ids is not None
        if fill_object_proto:
            object_proto = object_graph_proto.nodes[node_ids[trackable]]
        object_to_save = util.get_mapped_trackable(trackable, object_map)
        if saveables_cache is not None:
            cached_attributes = saveables_cache.setdefault(object_to_save, {})
        else:
            cached_attributes = None
        for factory_data in factory_data_list:
            name = factory_data.name
            key = factory_data.checkpoint_key
            saveable_factory = factory_data.factory
            saveables = cached_attributes.get(name) if cached_attributes else None
            if saveables is not None:
                for saveable in saveables:
                    if key not in saveable.name:
                        saveables = None
                        del cached_attributes[name]
                        break
            if saveables is None:
                if callable(saveable_factory):
                    maybe_saveable = saveable_object_util.create_saveable_object(name, key, saveable_factory, call_with_mapped_captures)
                else:
                    maybe_saveable = saveable_factory
                if isinstance(maybe_saveable, saveable_object_lib.SaveableObject):
                    saveables = (maybe_saveable,)
                else:
                    saveables = tuple(saveable_object_util.saveable_objects_for_op(op=maybe_saveable, name=key))
                for saveable in saveables:
                    if key not in saveable.name:
                        raise AssertionError(f"The object {trackable} produced a SaveableObject with name '{saveable.name}' for attribute '{name}'. Expected a name containing '{key}'.")
                if cached_attributes is not None:
                    cached_attributes[name] = saveables
            if isinstance(object_to_save, python_state.PythonState):
                assert len(saveables) == 1
                saveable = saveables[0]
                if feed_additions is None:
                    assert saveables_cache is None
                    saveables = (saveable.freeze(),)
                else:
                    feed_additions.update(saveable.feed_dict_additions())
            named_saveable_objects.extend(saveables)
            if not fill_object_proto:
                continue
            if isinstance(saveables[0], saveable_object_util.TrackableSaveable) and (saveable_compat.force_checkpoint_conversion_enabled() or saveable_compat.get_saveable_name(object_to_save) is None):
                for (local_name, local_key) in saveables[0].get_proto_names_and_checkpoint_keys():
                    object_proto.attributes.add(name=local_name, checkpoint_key=local_key, full_name=util.get_full_name(object_to_save))
            else:
                object_proto.attributes.add(name=name, checkpoint_key=key, full_name=util.get_full_name(object_to_save))
    return (named_saveable_objects, feed_additions)

def _fill_object_graph_proto(graph_view, trackable_objects, node_ids, slot_variables):
    if False:
        i = 10
        return i + 15
    'Name non-slot `Trackable`s and add them to `object_graph_proto`.'
    object_graph_proto = trackable_object_graph_pb2.TrackableObjectGraph()
    for (checkpoint_id, trackable) in enumerate(trackable_objects):
        assert node_ids[trackable] == checkpoint_id
        object_proto = object_graph_proto.nodes.add(slot_variables=slot_variables.get(trackable, ()))
        for child in graph_view.list_children(trackable):
            object_proto.children.add(node_id=node_ids[child.ref], local_name=child.name)
    return object_graph_proto

def serialize_gathered_objects(graph_view, object_map=None, call_with_mapped_captures=None, saveables_cache=None):
    if False:
        for i in range(10):
            print('nop')
    'Create SaveableObjects and protos for gathered objects.'
    (trackable_objects, node_paths) = graph_view.breadth_first_traversal()
    object_names = object_identity.ObjectIdentityDictionary()
    for (obj, path) in node_paths.items():
        object_names[obj] = trackable_utils.object_path_to_string(path)
    node_ids = object_identity.ObjectIdentityDictionary()
    for (node_id, node) in enumerate(trackable_objects):
        node_ids[node] = node_id
    slot_variables = util.serialize_slot_variables(trackable_objects=trackable_objects, node_ids=node_ids, object_names=object_names)
    object_graph_proto = _fill_object_graph_proto(graph_view=graph_view, trackable_objects=trackable_objects, node_ids=node_ids, slot_variables=slot_variables)
    (named_saveable_objects, feed_additions, registered_savers) = _add_attributes_to_object_graph(trackable_objects=trackable_objects, object_graph_proto=object_graph_proto, node_ids=node_ids, object_names=object_names, object_map=object_map, call_with_mapped_captures=call_with_mapped_captures, saveables_cache=saveables_cache)
    util.add_checkpoint_values_check(object_graph_proto)
    return (named_saveable_objects, object_graph_proto, feed_additions, registered_savers)

def serialize_object_graph_with_registered_savers(graph_view, saveables_cache):
    if False:
        return 10
    'Determine checkpoint keys for variables and build a serialized graph.'
    return serialize_gathered_objects(graph_view, saveables_cache=saveables_cache)

def frozen_saveables_and_savers(graph_view, object_map=None, to_graph=None, call_with_mapped_captures=None, saveables_cache=None):
    if False:
        i = 10
        return i + 15
    'Generates SaveableObjects and registered savers in the frozen graph.'
    if to_graph:
        target_context = to_graph.as_default
    else:
        target_context = ops.NullContextmanager
    with target_context():
        (named_saveable_objects, graph_proto, _, registered_savers) = serialize_gathered_objects(graph_view, object_map, call_with_mapped_captures, saveables_cache)
        with ops.device('/cpu:0'):
            object_graph_tensor = constant_op.constant(graph_proto.SerializeToString(), dtype=dtypes.string)
        named_saveable_objects.append(base.NoRestoreSaveable(tensor=object_graph_tensor, name=base.OBJECT_GRAPH_PROTO_KEY))
    return (named_saveable_objects, registered_savers)