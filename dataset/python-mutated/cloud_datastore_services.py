"""Provides a seam for datastore services."""
from __future__ import annotations
import contextlib
import logging
from core.platform import models
from google.cloud import ndb
from typing import Any, ContextManager, Dict, List, Optional, Sequence, Tuple, TypeVar
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import transaction_services
transaction_services = models.Registry.import_transaction_services()
Cursor = ndb.Cursor
Model = ndb.Model
Key = ndb.Key
Property = ndb.Property
Query = ndb.Query
RedisCache = ndb.RedisCache
BooleanProperty = ndb.BooleanProperty
DateProperty = ndb.DateProperty
DateTimeProperty = ndb.DateTimeProperty
FloatProperty = ndb.FloatProperty
IntegerProperty = ndb.IntegerProperty
JsonProperty = ndb.JsonProperty
StringProperty = ndb.StringProperty
TextProperty = ndb.TextProperty
TYPE_MODEL_SUBCLASS = TypeVar('TYPE_MODEL_SUBCLASS', bound=Model)
MAX_GET_RETRIES = 3
CLIENT = ndb.Client()

def get_ndb_context(namespace: Optional[str]=None, global_cache: Optional[RedisCache]=None) -> ContextManager[ndb.context.Context]:
    if False:
        for i in range(10):
            print('nop')
    'Get the context of the Cloud NDB. This context needs to be entered in\n    order to do any Cloud NDB operations.\n\n    Returns:\n        ndb.context.Context. Cloud NDB context.\n    '
    context = ndb.get_context(raise_context_error=False)
    return CLIENT.context(namespace=namespace, global_cache=global_cache) if context is None else contextlib.nullcontext(enter_result=context)

def get_multi(keys: List[Key]) -> List[Optional[TYPE_MODEL_SUBCLASS]]:
    if False:
        for i in range(10):
            print('nop')
    "Fetches models corresponding to a sequence of keys.\n\n    Args:\n        keys: list(str). The keys to look up.\n\n    Returns:\n        list(datastore_services.Model | None). List whose items are either a\n        Model instance or None if the corresponding key wasn't found.\n\n    Raises:\n        Exception. If ndb.get_multi fails for MAX_GET_RETRIES.\n    "
    for unused_i in range(0, MAX_GET_RETRIES):
        try:
            return ndb.get_multi(keys)
        except Exception as e:
            logging.exception('Exception raised: %s', e)
            continue
    raise Exception('get_multi failed after %s retries' % MAX_GET_RETRIES)

def update_timestamps_multi(entities: Sequence[base_models.BaseModel], update_last_updated_time: bool=True) -> None:
    if False:
        while True:
            i = 10
    'Update the created_on and last_updated fields of all given entities.\n\n    Args:\n        entities: list(datastore_services.Model). List of model instances to\n            be stored.\n        update_last_updated_time: bool. Whether to update the\n            last_updated field of the model.\n    '
    for entity in entities:
        entity.update_timestamps(update_last_updated_time=update_last_updated_time)

def put_multi(entities: Sequence[Model]) -> List[str]:
    if False:
        while True:
            i = 10
    'Stores a sequence of Model instances.\n\n    Args:\n        entities: list(datastore_services.Model). A list of Model instances.\n\n    Returns:\n        list(str). A list with the stored keys.\n    '
    return ndb.put_multi(list(entities))

@transaction_services.run_in_transaction_wrapper
def delete_multi_transactional(keys: List[Key]) -> List[None]:
    if False:
        return 10
    'Deletes models corresponding to a sequence of keys and runs it through\n    a transaction. Either all models are deleted, or none of them in the case\n    when the transaction fails.\n\n    Args:\n        keys: list(str). A list of keys.\n\n    Returns:\n        list(None). A list of Nones, one per deleted model.\n    '
    return ndb.delete_multi(keys)

def delete_multi(keys: Sequence[Key]) -> List[None]:
    if False:
        print('Hello World!')
    'Deletes models corresponding to a sequence of keys.\n\n    Args:\n        keys: list(str). A list of keys.\n\n    Returns:\n        list(None). A list of Nones, one per deleted model.\n    '
    return ndb.delete_multi(keys)

def query_everything(**kwargs: Dict[str, Any]) -> Query:
    if False:
        i = 10
        return i + 15
    'Returns a query that targets every single entity in the datastore.\n\n    IMPORTANT: DO NOT USE THIS FUNCTION OUTSIDE OF UNIT TESTS. Querying\n    everything in the datastore is almost always a bad idea, ESPECIALLY in\n    production. Always prefer querying for specific models and combining them\n    afterwards.\n    '
    return ndb.Query(**kwargs)

def all_of(*nodes: ndb.Node) -> ndb.Node:
    if False:
        return 10
    'Returns a query node which performs a boolean AND on their conditions.\n\n    Args:\n        *nodes: datastore_services.Node. The nodes to combine.\n\n    Returns:\n        datastore_services.Node. A node combining the conditions using boolean\n        AND.\n    '
    return ndb.AND(*nodes)

def any_of(*nodes: ndb.Node) -> ndb.Node:
    if False:
        print('Hello World!')
    'Returns a query node which performs a boolean OR on their conditions.\n\n    Args:\n        *nodes: datastore_services.Node. The nodes to combine.\n\n    Returns:\n        datastore_services.Node. A node combining the conditions using boolean\n        OR.\n    '
    return ndb.OR(*nodes)

def make_cursor(urlsafe_cursor: Optional[str]=None) -> Cursor:
    if False:
        for i in range(10):
            print('nop')
    'Makes an immutable cursor that points to a relative position in a query.\n\n    The position denoted by a Cursor is relative to the result of a query, even\n    if the result is removed later on. Usually, the position points to whatever\n    immediately follows the last result of a batch.\n\n    A cursor should only be used on a query with an identical signature to the\n    one that produced it, or on a query with its sort order reversed.\n\n    A Cursor constructed with no arguments points to the first result of any\n    query. If such a Cursor is used as an end_cursor, no results will be\n    returned.\n\n    Args:\n        urlsafe_cursor: str | None. The base64-encoded serialization of a\n            cursor. When None, the cursor returned will point to the first\n            result of any query.\n\n    Returns:\n        Cursor. A cursor into an arbitrary query.\n    '
    return Cursor(urlsafe=urlsafe_cursor)

def fetch_multiple_entities_by_ids_and_models(ids_and_models: List[Tuple[str, List[str]]]) -> List[List[Optional[TYPE_MODEL_SUBCLASS]]]:
    if False:
        print('Hello World!')
    'Fetches the entities from the datastore corresponding to the given ids\n    and models.\n\n    Args:\n        ids_and_models: list(tuple(str, list(str))). The ids and their\n            corresponding model names for which we have to fetch entities.\n\n    Raises:\n        Exception. Model names should not be duplicated in input list.\n\n    Returns:\n        list(list(datastore_services.Model)). The model instances corresponding\n        to the ids and models. The models corresponding to the same tuple in the\n        input are grouped together.\n    '
    entity_keys: List[Key] = []
    model_names = [model_name for (model_name, _) in ids_and_models]
    if len(model_names) != len(list(set(model_names))):
        raise Exception('Model names should not be duplicated in input list.')
    for (model_name, entity_ids) in ids_and_models:
        entity_keys = entity_keys + [ndb.Key(model_name, entity_id) for entity_id in entity_ids]
    all_models: List[Optional[TYPE_MODEL_SUBCLASS]] = ndb.get_multi(entity_keys)
    all_models_grouped_by_model_type: List[List[Optional[TYPE_MODEL_SUBCLASS]]] = []
    start_index = 0
    for (_, entity_ids) in ids_and_models:
        all_models_grouped_by_model_type.append(all_models[start_index:start_index + len(entity_ids)])
        start_index = start_index + len(entity_ids)
    return all_models_grouped_by_model_type