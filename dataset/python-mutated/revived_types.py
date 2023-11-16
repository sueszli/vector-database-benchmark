"""Handles types registrations for tf.saved_model.load."""
import operator
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.trackable import data_structures
from tensorflow.python.util.tf_export import tf_export

@tf_export('__internal__.saved_model.load.VersionedTypeRegistration', v1=[])
class VersionedTypeRegistration(object):
    """Holds information about one version of a revived type."""

    def __init__(self, object_factory, version, min_producer_version, min_consumer_version, bad_consumers=None, setter=setattr):
        if False:
            while True:
                i = 10
        'Identify a revived type version.\n\n    Args:\n      object_factory: A callable which takes a SavedUserObject proto and returns\n        a trackable object. Dependencies are added later via `setter`.\n      version: An integer, the producer version of this wrapper type. When\n        making incompatible changes to a wrapper, add a new\n        `VersionedTypeRegistration` with an incremented `version`. The most\n        recent version will be saved, and all registrations with a matching\n        identifier will be searched for the highest compatible version to use\n        when loading.\n      min_producer_version: The minimum producer version number required to use\n        this `VersionedTypeRegistration` when loading a proto.\n      min_consumer_version: `VersionedTypeRegistration`s with a version number\n        less than `min_consumer_version` will not be used to load a proto saved\n        with this object. `min_consumer_version` should be set to the lowest\n        version number which can successfully load protos saved by this\n        object. If no matching registration is available on load, the object\n        will be revived with a generic trackable type.\n\n        `min_consumer_version` and `bad_consumers` are a blunt tool, and using\n        them will generally break forward compatibility: previous versions of\n        TensorFlow will revive newly saved objects as opaque trackable\n        objects rather than wrapped objects. When updating wrappers, prefer\n        saving new information but preserving compatibility with previous\n        wrapper versions. They are, however, useful for ensuring that\n        previously-released buggy wrapper versions degrade gracefully rather\n        than throwing exceptions when presented with newly-saved SavedModels.\n      bad_consumers: A list of consumer versions which are incompatible (in\n        addition to any version less than `min_consumer_version`).\n      setter: A callable with the same signature as `setattr` to use when adding\n        dependencies to generated objects.\n    '
        self.setter = setter
        self.identifier = None
        self._object_factory = object_factory
        self.version = version
        self._min_consumer_version = min_consumer_version
        self._min_producer_version = min_producer_version
        if bad_consumers is None:
            bad_consumers = []
        self._bad_consumers = bad_consumers

    def to_proto(self):
        if False:
            while True:
                i = 10
        'Create a SavedUserObject proto.'
        return saved_object_graph_pb2.SavedUserObject(identifier=self.identifier, version=versions_pb2.VersionDef(producer=self.version, min_consumer=self._min_consumer_version, bad_consumers=self._bad_consumers))

    def from_proto(self, proto):
        if False:
            return 10
        'Recreate a trackable object from a SavedUserObject proto.'
        return self._object_factory(proto)

    def should_load(self, proto):
        if False:
            while True:
                i = 10
        'Checks if this object should load the SavedUserObject `proto`.'
        if proto.identifier != self.identifier:
            return False
        if self.version < proto.version.min_consumer:
            return False
        if proto.version.producer < self._min_producer_version:
            return False
        for bad_version in proto.version.bad_consumers:
            if self.version == bad_version:
                return False
        return True
_REVIVED_TYPE_REGISTRY = {}
_TYPE_IDENTIFIERS = []

@tf_export('__internal__.saved_model.load.register_revived_type', v1=[])
def register_revived_type(identifier, predicate, versions):
    if False:
        for i in range(10):
            print('nop')
    'Register a type for revived objects.\n\n  Args:\n    identifier: A unique string identifying this class of objects.\n    predicate: A Boolean predicate for this registration. Takes a\n      trackable object as an argument. If True, `type_registration` may be\n      used to save and restore the object.\n    versions: A list of `VersionedTypeRegistration` objects.\n  '
    versions.sort(key=lambda reg: reg.version, reverse=True)
    if not versions:
        raise AssertionError('Need at least one version of a registered type.')
    version_numbers = set()
    for registration in versions:
        registration.identifier = identifier
        if registration.version in version_numbers:
            raise AssertionError(f'Got multiple registrations with version {registration.version} for type {identifier}.')
        version_numbers.add(registration.version)
    _REVIVED_TYPE_REGISTRY[identifier] = (predicate, versions)
    _TYPE_IDENTIFIERS.append(identifier)

def serialize(obj):
    if False:
        for i in range(10):
            print('nop')
    'Create a SavedUserObject from a trackable object.'
    for identifier in _TYPE_IDENTIFIERS:
        (predicate, versions) = _REVIVED_TYPE_REGISTRY[identifier]
        if predicate(obj):
            return versions[0].to_proto()
    return None

def deserialize(proto):
    if False:
        for i in range(10):
            print('nop')
    'Create a trackable object from a SavedUserObject proto.\n\n  Args:\n    proto: A SavedUserObject to deserialize.\n\n  Returns:\n    A tuple of (trackable, assignment_fn) where assignment_fn has the same\n    signature as setattr and should be used to add dependencies to\n    `trackable` when they are available.\n  '
    (_, type_registrations) = _REVIVED_TYPE_REGISTRY.get(proto.identifier, (None, None))
    if type_registrations is not None:
        for type_registration in type_registrations:
            if type_registration.should_load(proto):
                return (type_registration.from_proto(proto), type_registration.setter)
    return None

@tf_export('__internal__.saved_model.load.registered_identifiers', v1=[])
def registered_identifiers():
    if False:
        return 10
    'Return all the current registered revived object identifiers.\n\n  Returns:\n    A set of strings.\n  '
    return _REVIVED_TYPE_REGISTRY.keys()

@tf_export('__internal__.saved_model.load.get_setter', v1=[])
def get_setter(proto):
    if False:
        return 10
    'Gets the registered setter function for the SavedUserObject proto.\n\n  See VersionedTypeRegistration for info about the setter function.\n\n  Args:\n    proto: SavedUserObject proto\n\n  Returns:\n    setter function\n  '
    (_, type_registrations) = _REVIVED_TYPE_REGISTRY.get(proto.identifier, (None, None))
    if type_registrations is not None:
        for type_registration in type_registrations:
            if type_registration.should_load(proto):
                return type_registration.setter
    return None
register_revived_type('trackable_dict_wrapper', lambda obj: isinstance(obj, data_structures._DictWrapper), versions=[VersionedTypeRegistration(object_factory=lambda proto: data_structures._DictWrapper({}), version=1, min_producer_version=1, min_consumer_version=1, setter=operator.setitem)])
register_revived_type('trackable_list_wrapper', lambda obj: isinstance(obj, data_structures.ListWrapper), versions=[VersionedTypeRegistration(object_factory=lambda proto: data_structures.ListWrapper([]), version=1, min_producer_version=1, min_consumer_version=1, setter=data_structures.set_list_item)])
register_revived_type('trackable_tuple_wrapper', lambda obj: isinstance(obj, data_structures._TupleWrapper), versions=[VersionedTypeRegistration(object_factory=lambda proto: data_structures.ListWrapper([]), version=1, min_producer_version=1, min_consumer_version=1, setter=data_structures.set_tuple_item)])