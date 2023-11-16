"""Helper functions for beam validators and one-off jobs."""
from __future__ import annotations
from core import feconf
from core.platform import models
from apache_beam.io.gcp.datastore.v1new import types as beam_datastore_types
from google.cloud.ndb import model as ndb_model
from google.cloud.ndb import query as ndb_query
from typing import Any, List, Optional, Tuple, Type, Union
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import datastore_services
datastore_services = models.Registry.import_datastore_services()
(base_models,) = models.Registry.import_models([models.Names.BASE_MODEL])

def clone_model(model: datastore_services.TYPE_MODEL_SUBCLASS, **new_values: Any) -> datastore_services.TYPE_MODEL_SUBCLASS:
    if False:
        while True:
            i = 10
    "Clones the entity, adding or overriding constructor attributes.\n\n    The cloned entity will have exactly the same property values as the\n    original entity, except where overridden. By default, it will have no\n    parent entity or key name, unless supplied.\n\n    IMPORTANT: This function should be used in EVERY DoFn, beacse one of Apache\n    Beam's invariants is that all input values are IMMUTABLE.\n    TODO(#12449): Use a metaclass to wrap DoFn.process() with a function that\n    clones inputs, so that contributors don't need to remember to.\n\n    Args:\n        model: datastore_services.Model. Model to clone.\n        **new_values: dict(str: *). Keyword arguments to override when\n            invoking the cloned entity's constructor.\n\n    Returns:\n        datastore_services.Model. A cloned, and possibly modified, copy of self.\n        Subclasses of BaseModel will return a clone with the same type.\n    "
    model_id = new_values.pop('id', None) or get_model_id(model)
    cls = model.__class__
    props = {k: v.__get__(model, cls) for (k, v) in cls._properties.items()}
    props.update(new_values)
    with datastore_services.get_ndb_context():
        return cls(id=model_id, **props)

def get_model_class(kind: Optional[str]) -> Type[datastore_services.Model]:
    if False:
        return 10
    'Returns the model class corresponding to the given kind.\n\n    NOTE: A model\'s kind is usually, but not always, the same as a model\'s class\n    name. Specifically, the kind is different when a model overrides the\n    _get_kind() class method. Although Oppia never does this, the Apache Beam\n    framework uses "kind" to refer to models _extensively_, so we follow the\n    same convention and take special care to always return the correct value.\n\n    Args:\n        kind: str. The model\'s kind.\n\n    Returns:\n        type(datastore_services.Model). The corresponding class.\n\n    Raises:\n        KindError. Internally raised by _lookup_model when the kind is invalid.\n    '
    models.Registry.get_all_storage_model_classes()
    return datastore_services.Model._lookup_model(kind)

def get_model_kind(model: Union[datastore_services.Model, Type[datastore_services.Model]]) -> str:
    if False:
        i = 10
        return i + 15
    'Returns the "kind" of the given model.\n\n    NOTE: A model\'s kind is usually, but not always, the same as a model\'s class\n    name. Specifically, the kind is different when a model overwrites the\n    _get_kind() class method. Although Oppia never does this, the Apache Beam\n    framework uses "kind" to refer to models extensively, so we follow the same\n    convention and take special care to always return the correct value.\n\n    Args:\n        model: datastore_services.Model. The model to inspect.\n\n    Returns:\n        str. The model\'s kind.\n\n    Raises:\n        TypeError. When the argument is not a model.\n    '
    if isinstance(model, datastore_services.Model) or (isinstance(model, type) and issubclass(model, datastore_services.Model)):
        return model._get_kind()
    else:
        raise TypeError('%r is not a model type or instance' % model)

def get_model_id(model: datastore_services.Model) -> Optional[str]:
    if False:
        print('Hello World!')
    "Returns the given model's ID.\n\n    Args:\n        model: datastore_services.Model. The model to inspect.\n\n    Returns:\n        bytes. The model's ID.\n\n    Raises:\n        TypeError. When the argument is not a model.\n    "
    if isinstance(model, base_models.BaseModel):
        return model.id
    elif isinstance(model, datastore_services.Model):
        return None if model.key is None else model.key.id()
    else:
        raise TypeError('%r is not a model instance' % model)

def get_model_property(model: datastore_services.Model, property_name: str) -> Any:
    if False:
        for i in range(10):
            print('nop')
    "Returns the given property from a model.\n\n    Args:\n        model: datastore_services.Model. The model to inspect.\n        property_name: str. The name of the property to extract.\n\n    Returns:\n        *. The property's value.\n\n    Raises:\n        TypeError. When the argument is not a model.\n    "
    if property_name == 'id':
        return get_model_id(model)
    elif isinstance(model, datastore_services.Model):
        return getattr(model, property_name)
    else:
        raise TypeError('%r is not a model instance' % model)

def get_beam_entity_from_ndb_model(model: datastore_services.TYPE_MODEL_SUBCLASS) -> beam_datastore_types.Entity:
    if False:
        print('Hello World!')
    'Returns an Apache Beam entity equivalent to the given NDB model.\n\n    Args:\n        model: datastore_services.Model. The NDB model.\n\n    Returns:\n        beam_datastore_types.Entity. The Apache Beam entity.\n    '
    with datastore_services.get_ndb_context():
        model_to_put = ndb_model._entity_to_ds_entity(model)
    return beam_datastore_types.Entity.from_client_entity(model_to_put)

def get_ndb_model_from_beam_entity(beam_entity: beam_datastore_types.Entity) -> datastore_services.Model:
    if False:
        while True:
            i = 10
    'Returns an NDB model equivalent to the given Apache Beam entity.\n\n    Args:\n        beam_entity: beam_datastore_types.Entity. The Apache Beam entity.\n\n    Returns:\n        datastore_services.Model. The NDB model.\n    '
    ndb_key = get_ndb_key_from_beam_key(beam_entity.key)
    ndb_model_class = get_model_class(ndb_key.kind())
    return ndb_model._entity_from_ds_entity(beam_entity.to_client_entity(), model_class=ndb_model_class)

def get_ndb_key_from_beam_key(beam_key: beam_datastore_types.Key) -> datastore_services.Key:
    if False:
        i = 10
        return i + 15
    'Returns an NDB key equivalent to the given Apache Beam key.\n\n    Args:\n        beam_key: beam_datastore_types.Key. The Apache Beam key.\n\n    Returns:\n        datastore_services.Key. The NDB key.\n    '
    return datastore_services.Key._from_ds_key(beam_key.to_client_key())

def get_beam_key_from_ndb_key(ndb_key: datastore_services.Key) -> beam_datastore_types.Key:
    if False:
        for i in range(10):
            print('nop')
    'Returns an Apache Beam key equivalent to the given NDB key.\n\n    Args:\n        ndb_key: datastore_services.Key. The NDB key.\n\n    Returns:\n        beam_datastore_types.Key. The Apache Beam key.\n    '
    return beam_datastore_types.Key(ndb_key.flat(), project=ndb_key.project(), namespace=ndb_key.namespace())

def get_beam_query_from_ndb_query(query: datastore_services.Query, namespace: Optional[str]=None) -> beam_datastore_types.Query:
    if False:
        print('Hello World!')
    'Returns an equivalent Apache Beam query from the given NDB query.\n\n    This function helps developers avoid learning two types of query syntaxes.\n    Specifically, the datastoreio module offered by the Apache Beam SDK only\n    accepts Beam datastore queries, and are implemented very differently from\n    NDB queries. This function adapts the two patterns to make job code easier\n    to write.\n\n    Args:\n        query: datastore_services.Query. The NDB query to convert.\n        namespace: str|None. Namespace for isolating the NDB operations of unit\n            tests. IMPORTANT: Do not use this argument outside of unit tests.\n\n    Returns:\n        beam_datastore_types.Query. The equivalent Apache Beam query.\n    '
    kind = query.kind
    namespace = namespace or query.namespace
    filters = _get_beam_filters_from_ndb_node(query.filters) if query.filters else ()
    order = _get_beam_order_from_ndb_order(query.order_by) if query.order_by else ()
    if not kind and (not order):
        order = ('__key__',)
    return beam_datastore_types.Query(kind=kind, namespace=namespace, project=feconf.OPPIA_PROJECT_ID, filters=filters, order=order)

def _get_beam_filters_from_ndb_node(node: ndb_query.Node) -> Tuple[Tuple[str, str, Any], ...]:
    if False:
        return 10
    'Returns an equivalent Apache Beam filter from the given NDB filter node.\n\n    Args:\n        node: datastore_services.FilterNode. The filter node to convert.\n\n    Returns:\n        tuple(tuple(str, str, *)). The equivalent Apache Beam filters. Items\n        are: (property name, comparison operator, property value).\n\n    Raises:\n        TypeError. These `!=`, `IN`, and `OR` are forbidden filters.\n    '
    beam_filters: List[Tuple[str, str, Any]] = []
    if isinstance(node, ndb_query.ConjunctionNode):
        for n in node:
            beam_filters.extend(_get_beam_filters_from_ndb_node(n))
    elif isinstance(node, ndb_query.FilterNode):
        beam_filters.append((node._name, node._opsymbol, node._value))
    else:
        raise TypeError('`!=`, `IN`, and `OR` are forbidden filters. To emulate their behavior, use multiple AND queries and flatten them into a single PCollection.')
    return tuple(beam_filters)

def _get_beam_order_from_ndb_order(orders: List[ndb_query.PropertyOrder]) -> Tuple[str, ...]:
    if False:
        print('Hello World!')
    'Returns an equivalent Apache Beam order from the given datastore Order.\n\n    Args:\n        orders: list(datastore_query.Order). The datastore order to convert.\n\n    Returns:\n        tuple(str). The equivalent Apache Beam order.\n    '
    return tuple(('%s%s' % ('-' if o.reverse else '', o.name) for o in orders))