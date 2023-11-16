"""Commands that can be used to operate on collections.

All functions here should be agnostic of how CollectionModel objects are
stored in the database. In particular, the various query methods should
delegate to the Collection model class. This will enable the collection
storage model to be changed without affecting this module and others above it.
"""
from __future__ import annotations
import collections
import copy
import logging
import os
from core import feconf
from core import utils
from core.constants import constants
from core.domain import activity_services
from core.domain import caching_services
from core.domain import change_domain
from core.domain import collection_domain
from core.domain import exp_fetchers
from core.domain import exp_services
from core.domain import rights_domain
from core.domain import rights_manager
from core.domain import search_services
from core.domain import subscription_services
from core.domain import user_domain
from core.domain import user_services
from core.platform import models
from typing import Dict, Final, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, TypedDict, cast, overload
MYPY = False
if MYPY:
    from mypy_imports import collection_models
    from mypy_imports import datastore_services
    from mypy_imports import user_models
(collection_models, user_models) = models.Registry.import_models([models.Names.COLLECTION, models.Names.USER])
datastore_services = models.Registry.import_datastore_services()
CMD_CREATE_NEW: Final = 'create_new'
SEARCH_INDEX_COLLECTIONS: Final = 'collections'
MAX_ITERATIONS: Final = 10

class SnapshotsMetadataDict(TypedDict):
    """Dictionary representing the snapshot metadata for collection models."""
    committer_id: str
    commit_message: str
    commit_cmds: List[Dict[str, change_domain.AcceptableChangeDictTypes]]
    commit_type: str
    version_number: int
    created_on_ms: float

def _migrate_collection_contents_to_latest_schema(versioned_collection_contents: collection_domain.VersionedCollectionDict) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Holds the responsibility of performing a step-by-step, sequential update\n    of the collection structure based on the schema version of the input\n    collection dictionary. This is very similar to the exploration migration\n    process seen in exp_services. If any of the current collection schemas\n    change, a new conversion function must be added and some code appended to\n    this function to account for that new version.\n\n    Args:\n        versioned_collection_contents: dict. A dict with two keys:\n          - schema_version: int. The schema version for the collection.\n          - collection_contents: dict. The dict comprising the collection\n              contents.\n\n    Raises:\n        Exception. The schema version of the collection is outside of what is\n            supported at present.\n    '
    collection_schema_version = versioned_collection_contents['schema_version']
    if not 1 <= collection_schema_version <= feconf.CURRENT_COLLECTION_SCHEMA_VERSION:
        raise Exception('Sorry, we can only process v1-v%d collection schemas at present.' % feconf.CURRENT_COLLECTION_SCHEMA_VERSION)
    while collection_schema_version < feconf.CURRENT_COLLECTION_SCHEMA_VERSION:
        collection_domain.Collection.update_collection_contents_from_model(versioned_collection_contents, collection_schema_version)
        collection_schema_version += 1

def get_collection_from_model(collection_model: collection_models.CollectionModel) -> collection_domain.Collection:
    if False:
        print('Hello World!')
    'Returns a Collection domain object given a collection model loaded\n    from the datastore.\n\n    Args:\n        collection_model: CollectionModel. The collection model loaded from the\n            datastore.\n\n    Returns:\n        Collection. A Collection domain object corresponding to the given\n        collection model.\n    '
    versioned_collection_contents: collection_domain.VersionedCollectionDict = {'schema_version': collection_model.schema_version, 'collection_contents': copy.deepcopy(collection_model.collection_contents)}
    if collection_model.schema_version == 2:
        versioned_collection_contents['collection_contents'] = {'nodes': copy.deepcopy(collection_model.nodes)}
    if collection_model.schema_version != feconf.CURRENT_COLLECTION_SCHEMA_VERSION:
        _migrate_collection_contents_to_latest_schema(versioned_collection_contents)
    return collection_domain.Collection(collection_model.id, collection_model.title, collection_model.category, collection_model.objective, collection_model.language_code, collection_model.tags, versioned_collection_contents['schema_version'], [collection_domain.CollectionNode.from_dict(collection_node_dict) for collection_node_dict in versioned_collection_contents['collection_contents']['nodes']], collection_model.version, collection_model.created_on, collection_model.last_updated)

def get_collection_summary_from_model(collection_summary_model: collection_models.CollectionSummaryModel) -> collection_domain.CollectionSummary:
    if False:
        for i in range(10):
            print('nop')
    'Returns a domain object for an Oppia collection summary given a\n    collection summary model.\n\n    Args:\n        collection_summary_model: CollectionSummaryModel. The model object\n            to extract domain object for oppia collection summary.\n\n    Returns:\n        CollectionSummary. The collection summary domain object extracted\n        from collection summary model.\n    '
    return collection_domain.CollectionSummary(collection_summary_model.id, collection_summary_model.title, collection_summary_model.category, collection_summary_model.objective, collection_summary_model.language_code, collection_summary_model.tags, collection_summary_model.status, collection_summary_model.community_owned, collection_summary_model.owner_ids, collection_summary_model.editor_ids, collection_summary_model.viewer_ids, collection_summary_model.contributor_ids, collection_summary_model.contributors_summary, collection_summary_model.version, collection_summary_model.node_count, collection_summary_model.collection_model_created_on, collection_summary_model.collection_model_last_updated)

@overload
def get_collection_by_id(collection_id: str) -> collection_domain.Collection:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_collection_by_id(collection_id: str, *, version: Optional[int]=None) -> collection_domain.Collection:
    if False:
        print('Hello World!')
    ...

@overload
def get_collection_by_id(collection_id: str, *, strict: Literal[True], version: Optional[int]=None) -> collection_domain.Collection:
    if False:
        print('Hello World!')
    ...

@overload
def get_collection_by_id(collection_id: str, *, strict: Literal[False], version: Optional[int]=None) -> Optional[collection_domain.Collection]:
    if False:
        while True:
            i = 10
    ...

@overload
def get_collection_by_id(collection_id: str, *, strict: bool, version: Optional[int]=None) -> Optional[collection_domain.Collection]:
    if False:
        print('Hello World!')
    ...

def get_collection_by_id(collection_id: str, strict: bool=True, version: Optional[int]=None) -> Optional[collection_domain.Collection]:
    if False:
        while True:
            i = 10
    'Returns a domain object representing a collection.\n\n    Args:\n        collection_id: str. ID of the collection.\n        strict: bool. Whether to fail noisily if no collection with the given\n            id exists in the datastore.\n        version: int or None. The version number of the collection to be\n            retrieved. If it is None, the latest version will be retrieved.\n\n    Returns:\n        Collection or None. The domain object representing a collection with the\n        given id, or None if it does not exist.\n    '
    sub_namespace = str(version) if version else None
    cached_collection = caching_services.get_multi(caching_services.CACHE_NAMESPACE_COLLECTION, sub_namespace, [collection_id]).get(collection_id)
    if cached_collection is not None:
        return cached_collection
    else:
        collection_model = collection_models.CollectionModel.get(collection_id, strict=strict, version=version)
        if collection_model:
            collection = get_collection_from_model(collection_model)
            caching_services.set_multi(caching_services.CACHE_NAMESPACE_COLLECTION, sub_namespace, {collection_id: collection})
            return collection
        else:
            return None

def get_collection_summary_by_id(collection_id: str) -> Optional[collection_domain.CollectionSummary]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a domain object representing a collection summary.\n\n    Args:\n        collection_id: str. ID of the collection summary.\n\n    Returns:\n        CollectionSummary|None. The collection summary domain object\n        corresponding to a collection with the given collection_id or\n        None if no CollectionSummaryModel exists for the given ID.\n    '
    collection_summary_model = collection_models.CollectionSummaryModel.get(collection_id, strict=False)
    if collection_summary_model is None:
        return None
    collection_summary = get_collection_summary_from_model(collection_summary_model)
    return collection_summary

def get_multiple_collections_by_id(collection_ids: List[str], strict: bool=True) -> Dict[str, collection_domain.Collection]:
    if False:
        i = 10
        return i + 15
    "Returns a dict of domain objects representing collections with the\n    given ids as keys.\n\n    Args:\n        collection_ids: list(str). A list of collection ids of collections to\n            be retrieved.\n        strict: bool. Whether to fail noisily if no collection with a given id\n            exists in the datastore.\n\n    Returns:\n        dict. A dict of domain objects representing collections with\n        the given ids as keys.\n\n    Raises:\n        ValueError. The 'strict' is True, and one or more of the given\n            collection ids are invalid.\n    "
    result = {}
    uncached = []
    cache_result = caching_services.get_multi(caching_services.CACHE_NAMESPACE_COLLECTION, None, collection_ids)
    for collection_obj in cache_result.values():
        result[collection_obj.id] = collection_obj
    for _id in collection_ids:
        if _id not in result:
            uncached.append(_id)
    db_collection_models = collection_models.CollectionModel.get_multi(uncached)
    db_results_dict = {}
    not_found = []
    for (index, cid) in enumerate(uncached):
        model = db_collection_models[index]
        if model:
            collection = get_collection_from_model(model)
            db_results_dict[cid] = collection
        else:
            logging.info('Tried to fetch collection with id %s, but no such collection exists in the datastore' % cid)
            not_found.append(cid)
    if strict and not_found:
        raise ValueError("Couldn't find collections with the following ids:\n%s" % '\n'.join(not_found))
    cache_update = {cid: val for (cid, val) in db_results_dict.items() if val is not None}
    if cache_update:
        caching_services.set_multi(caching_services.CACHE_NAMESPACE_COLLECTION, None, cache_update)
    result.update(db_results_dict)
    return result

def get_collection_and_collection_rights_by_id(collection_id: str) -> Tuple[Optional[collection_domain.Collection], Optional[rights_domain.ActivityRights]]:
    if False:
        i = 10
        return i + 15
    'Returns a tuple for collection domain object and collection rights\n    object.\n\n    Args:\n        collection_id: str. Id of the collection.\n\n    Returns:\n        tuple(Collection|None, CollectionRights|None). The collection and\n        collection rights domain object, respectively.\n    '
    collection_and_rights = datastore_services.fetch_multiple_entities_by_ids_and_models([('CollectionModel', [collection_id]), ('CollectionRightsModel', [collection_id])])
    collection = None
    if collection_and_rights[0][0] is not None:
        collection = get_collection_from_model(collection_and_rights[0][0])
    collection_rights = None
    if collection_and_rights[1][0] is not None:
        collection_rights = rights_manager.get_activity_rights_from_model(collection_and_rights[1][0], constants.ACTIVITY_TYPE_COLLECTION)
    return (collection, collection_rights)

def get_new_collection_id() -> str:
    if False:
        for i in range(10):
            print('nop')
    'Returns a new collection id.\n\n    Returns:\n        str. A new collection id.\n    '
    return collection_models.CollectionModel.get_new_id('')

def get_collection_titles_and_categories(collection_ids: List[str]) -> Dict[str, Dict[str, str]]:
    if False:
        i = 10
        return i + 15
    "Returns collection titles and categories for the given ids.\n\n    Args:\n        collection_ids: list(str). IDs of the collections whose titles and\n            categories are to be retrieved.\n\n    Returns:\n        A dict with collection ids as keys. The corresponding values\n        are dicts with the keys 'title' and 'category'.\n\n        Any invalid collection_ids will not be included in the return dict. No\n        error will be raised.\n    "
    collection_list = [get_collection_from_model(e) if e else None for e in collection_models.CollectionModel.get_multi(collection_ids)]
    result = {}
    for collection in collection_list:
        if collection is None:
            logging.error('Could not find collection corresponding to id')
        else:
            result[collection.id] = {'title': collection.title, 'category': collection.category}
    return result

def get_completed_exploration_ids(user_id: str, collection_id: str) -> List[str]:
    if False:
        print('Hello World!')
    "Returns a list of explorations the user has completed within the context\n    of the provided collection.\n\n    Args:\n        user_id: str. ID of the given user.\n        collection_id: str. ID of the collection.\n\n    Returns:\n        list(str). A list of exploration ids that the user with the given\n        user id has completed within the context of the provided collection with\n        the given collection id. The list is empty if the user has not yet\n        completed any explorations within the collection, or if either the\n        collection and/or user do not exist.\n\n        A progress model isn't added until the first exploration of a collection\n        is completed, so, if a model is missing, there isn't enough information\n        to infer whether that means the collection doesn't exist, the user\n        doesn't exist, or if they just haven't mdae any progress in that\n        collection yet. Thus, we just assume the user and collection exist for\n        the sake of this call, so it returns an empty list, indicating that no\n        progress has yet been made.\n    "
    progress_model = user_models.CollectionProgressModel.get(user_id, collection_id)
    if progress_model:
        exploration_ids: List[str] = progress_model.completed_explorations
        return exploration_ids
    else:
        return []

def get_explorations_completed_in_collections(user_id: str, collection_ids: List[str]) -> List[List[str]]:
    if False:
        print('Hello World!')
    'Returns the ids of the explorations completed in each of the collections.\n\n    Args:\n        user_id: str. ID of the given user.\n        collection_ids: list(str). IDs of the collections.\n\n    Returns:\n        list(list(str)). List of the exploration ids completed in each\n        collection.\n    '
    progress_models = user_models.CollectionProgressModel.get_multi(user_id, collection_ids)
    exploration_ids_completed_in_collections = []
    for progress_model in progress_models:
        if progress_model:
            exploration_ids_completed_in_collections.append(progress_model.completed_explorations)
        else:
            exploration_ids_completed_in_collections.append([])
    return exploration_ids_completed_in_collections

def get_valid_completed_exploration_ids(user_id: str, collection: collection_domain.Collection) -> List[str]:
    if False:
        print('Hello World!')
    'Returns a filtered version of the return value of\n    get_completed_exploration_ids, which only includes explorations found within\n    the current version of the collection.\n\n    Args:\n        user_id: str. ID of the given user.\n        collection: Collection. The collection to fetch exploration from.\n\n    Returns:\n        list(str). A filtered version of the return value of\n        get_completed_exploration_ids which only includes explorations found\n        within the current version of the collection.\n    '
    completed_exploration_ids = get_completed_exploration_ids(user_id, collection.id)
    return [exp_id for exp_id in completed_exploration_ids if collection.get_node(exp_id)]

def get_next_exploration_id_to_complete_by_user(user_id: str, collection_id: str) -> Optional[str]:
    if False:
        while True:
            i = 10
    "Returns the first exploration ID in the specified collection that the\n    given user has not yet attempted.\n\n    Args:\n        user_id: str. ID of the user.\n        collection_id: str. ID of the collection.\n\n    Returns:\n        str|None. The first exploration ID in the specified collection that\n        the given user has not completed. Returns the collection's initial\n        exploration if the user has yet to complete any explorations\n        within the collection or None if the collection is completed.\n    "
    completed_exploration_ids = get_completed_exploration_ids(user_id, collection_id)
    collection = get_collection_by_id(collection_id)
    if completed_exploration_ids:
        return collection.get_next_exploration_id(completed_exploration_ids)
    else:
        return collection.first_exploration_id

def record_played_exploration_in_collection_context(user_id: str, collection_id: str, exploration_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Records a exploration by a given user in a given collection\n    context as having been played.\n\n    Args:\n        user_id: str. ID of the given user.\n        collection_id: str. ID of the given collection.\n        exploration_id: str. ID of the given exploration.\n    '
    progress_model = user_models.CollectionProgressModel.get_or_create(user_id, collection_id)
    if exploration_id not in progress_model.completed_explorations:
        progress_model.completed_explorations.append(exploration_id)
        progress_model.update_timestamps()
        progress_model.put()

def get_collection_summary_dicts_from_models(collection_summary_models: Iterable[collection_models.CollectionSummaryModel]) -> Dict[str, collection_domain.CollectionSummary]:
    if False:
        i = 10
        return i + 15
    'Given an iterable of CollectionSummaryModel instances, create a dict\n    containing corresponding collection summary domain objects, keyed by id.\n\n    Args:\n        collection_summary_models: iterable(CollectionSummaryModel). An\n            iterable of CollectionSummaryModel instances.\n\n    Returns:\n        dict. A dict containing corresponding collection summary domain objects,\n        keyed by id.\n    '
    collection_summaries = [get_collection_summary_from_model(collection_summary_model) for collection_summary_model in collection_summary_models]
    result = {}
    for collection_summary in collection_summaries:
        result[collection_summary.id] = collection_summary
    return result

def get_collection_summaries_matching_ids(collection_ids: List[str]) -> List[Optional[collection_domain.CollectionSummary]]:
    if False:
        return 10
    'Given a list of collection ids, return a list with the corresponding\n    summary domain objects (or None if the corresponding summary does not\n    exist).\n\n    Args:\n        collection_ids: list(str). A list of collection ids.\n\n    Returns:\n        list(CollectionSummary). A list with the corresponding summary domain\n        objects or None if the corresponding summary does not exist.\n    '
    return [get_collection_summary_from_model(model) if model else None for model in collection_models.CollectionSummaryModel.get_multi(collection_ids)]

def get_collection_summaries_subscribed_to(user_id: str) -> List[collection_domain.CollectionSummary]:
    if False:
        return 10
    'Returns a list of CollectionSummary domain objects that the user\n    subscribes to.\n\n    Args:\n        user_id: str. The id of the user.\n\n    Returns:\n        list(CollectionSummary). List of CollectionSummary domain objects that\n        the user subscribes to.\n    '
    return [summary for summary in get_collection_summaries_matching_ids(subscription_services.get_collection_ids_subscribed_to(user_id)) if summary is not None]

def get_collection_summaries_where_user_has_role(user_id: str) -> List[collection_domain.CollectionSummary]:
    if False:
        while True:
            i = 10
    'Returns a list of CollectionSummary domain objects where the user has\n    some role.\n\n    Args:\n        user_id: str. The id of the user.\n\n    Returns:\n        list(CollectionSummary). List of CollectionSummary domain objects\n        where the user has some role.\n    '
    col_summary_models: Sequence[collection_models.CollectionSummaryModel] = collection_models.CollectionSummaryModel.query(datastore_services.any_of(collection_models.CollectionSummaryModel.owner_ids == user_id, collection_models.CollectionSummaryModel.editor_ids == user_id, collection_models.CollectionSummaryModel.viewer_ids == user_id, collection_models.CollectionSummaryModel.contributor_ids == user_id)).fetch()
    return [get_collection_summary_from_model(col_summary_model) for col_summary_model in col_summary_models]

def get_collection_ids_matching_query(query_string: str, categories: List[str], language_codes: List[str], offset: Optional[int]=None) -> Tuple[List[str], Optional[int]]:
    if False:
        i = 10
        return i + 15
    'Returns a list with all collection ids matching the given search query\n    string, as well as a search offset for future fetches.\n\n    Args:\n        query_string: str. The search query string.\n        categories: list(str). The list of categories to query for. If it is\n            empty, no category filter is applied to the results. If it is not\n            empty, then a result is considered valid if it matches at least one\n            of these categories.\n        language_codes: list(str). The list of language codes to query for. If\n            it is empty, no language code filter is applied to the results. If\n            it is not empty, then a result is considered valid if it matches at\n            least one of these language codes.\n        offset: int or None. Offset indicating where, in the list of\n            collections, to start the search from.\n\n    Returns:\n        2-tuple of (returned_collection_ids, search_offset). Where:\n            returned_collection_ids : list(str). A list with all collection ids\n                matching the given search query string, as well as a search\n                offset for future fetches. The list contains exactly\n                feconf.SEARCH_RESULTS_PAGE_SIZE results if there are at least\n                that many, otherwise it contains all remaining results. (If this\n                behaviour does not occur, an error will be logged.)\n            search_offset: int. Search offset for future fetches.\n    '
    returned_collection_ids: List[str] = []
    search_offset = offset
    for _ in range(MAX_ITERATIONS):
        remaining_to_fetch = feconf.SEARCH_RESULTS_PAGE_SIZE - len(returned_collection_ids)
        (collection_ids, search_offset) = search_services.search_collections(query_string, categories, language_codes, remaining_to_fetch, offset=search_offset)
        for (ind, _) in enumerate(collection_models.CollectionSummaryModel.get_multi(collection_ids)):
            returned_collection_ids.append(collection_ids[ind])
        if len(returned_collection_ids) == feconf.SEARCH_RESULTS_PAGE_SIZE or search_offset is None:
            break
    return (returned_collection_ids, search_offset)

def apply_change_list(collection_id: str, change_list: Sequence[Mapping[str, change_domain.AcceptableChangeDictTypes]]) -> collection_domain.Collection:
    if False:
        print('Hello World!')
    'Applies a changelist to a pristine collection and returns the result.\n\n    Args:\n        collection_id: str. ID of the given collection.\n        change_list: list(dict). A change list to be applied to the given\n            collection. Each entry is a dict that represents a CollectionChange\n            object.\n\n    Returns:\n        Collection. The resulting collection domain object.\n\n    Raises:\n        Exception. The change list is not applicable on the given collection.\n    '
    collection = get_collection_by_id(collection_id)
    try:
        changes = [collection_domain.CollectionChange(change_dict) for change_dict in change_list]
        for change in changes:
            if change.cmd == collection_domain.CMD_ADD_COLLECTION_NODE:
                add_collection_node_cmd = cast(collection_domain.AddCollectionNodeCmd, change)
                collection.add_node(add_collection_node_cmd.exploration_id)
            elif change.cmd == collection_domain.CMD_DELETE_COLLECTION_NODE:
                delete_collection_node_cmd = cast(collection_domain.DeleteCollectionNodeCmd, change)
                collection.delete_node(delete_collection_node_cmd.exploration_id)
            elif change.cmd == collection_domain.CMD_SWAP_COLLECTION_NODES:
                swap_collection_nodes_cmd = cast(collection_domain.SwapCollectionNodesCmd, change)
                collection.swap_nodes(swap_collection_nodes_cmd.first_index, swap_collection_nodes_cmd.second_index)
            elif change.cmd == collection_domain.CMD_EDIT_COLLECTION_PROPERTY:
                if change.property_name == collection_domain.COLLECTION_PROPERTY_TITLE:
                    edit_collection_property_title_cmd = cast(collection_domain.EditCollectionPropertyTitleCmd, change)
                    collection.update_title(edit_collection_property_title_cmd.new_value)
                elif change.property_name == collection_domain.COLLECTION_PROPERTY_CATEGORY:
                    edit_collection_property_category_cmd = cast(collection_domain.EditCollectionPropertyCategoryCmd, change)
                    collection.update_category(edit_collection_property_category_cmd.new_value)
                elif change.property_name == collection_domain.COLLECTION_PROPERTY_OBJECTIVE:
                    edit_collection_property_objective_cmd = cast(collection_domain.EditCollectionPropertyObjectiveCmd, change)
                    collection.update_objective(edit_collection_property_objective_cmd.new_value)
                elif change.property_name == collection_domain.COLLECTION_PROPERTY_LANGUAGE_CODE:
                    edit_collection_property_language_code_cmd = cast(collection_domain.EditCollectionPropertyLanguageCodeCmd, change)
                    collection.update_language_code(edit_collection_property_language_code_cmd.new_value)
                elif change.property_name == collection_domain.COLLECTION_PROPERTY_TAGS:
                    edit_collection_property_tags_cmd = cast(collection_domain.EditCollectionPropertyTagsCmd, change)
                    collection.update_tags(edit_collection_property_tags_cmd.new_value)
            elif change.cmd == collection_domain.CMD_MIGRATE_SCHEMA_TO_LATEST_VERSION:
                continue
        return collection
    except Exception as e:
        logging.error('%s %s %s %s' % (e.__class__.__name__, e, collection_id, change_list))
        raise e

def validate_exps_in_collection_are_public(collection: collection_domain.Collection) -> None:
    if False:
        return 10
    'Validates that explorations in a given collection are public.\n\n    Args:\n        collection: Collection. Collection to be validated.\n\n    Raises:\n        ValidationError. The collection contains at least one private\n            exploration.\n    '
    for exploration_id in collection.exploration_ids:
        if rights_manager.is_exploration_private(exploration_id):
            raise utils.ValidationError('Cannot reference a private exploration within a public collection, exploration ID: %s' % exploration_id)

def _save_collection(committer_id: str, collection: collection_domain.Collection, commit_message: Optional[str], change_list: Sequence[Mapping[str, change_domain.AcceptableChangeDictTypes]]) -> None:
    if False:
        return 10
    'Validates a collection and commits it to persistent storage. If\n    successful, increments the version number of the incoming collection domain\n    object by 1.\n\n    Args:\n        committer_id: str. ID of the given committer.\n        collection: Collection. The collection domain object to be saved.\n        commit_message: str|None. The commit message or None if unpublished\n            collection is provided.\n        change_list: list(dict). List of changes applied to a collection. Each\n            entry in change_list is a dict that represents a CollectionChange.\n\n    Raises:\n        ValidationError. An invalid exploration was referenced in the\n            collection.\n        Exception. The collection model and the incoming collection domain\n            object have different version numbers.\n    '
    if not change_list:
        raise Exception('Unexpected error: received an invalid change list when trying to save collection %s: %s' % (collection.id, change_list))
    collection_rights = rights_manager.get_collection_rights(collection.id)
    if collection_rights.status != rights_domain.ACTIVITY_STATUS_PRIVATE:
        collection.validate(strict=True)
    else:
        collection.validate(strict=False)
    exp_ids = collection.exploration_ids
    exp_summaries = exp_fetchers.get_exploration_summaries_matching_ids(exp_ids)
    exp_summaries_dict = {exp_id: exp_summaries[ind] for (ind, exp_id) in enumerate(exp_ids)}
    for collection_node in collection.nodes:
        if not exp_summaries_dict[collection_node.exploration_id]:
            raise utils.ValidationError('Expected collection to only reference valid explorations, but found an exploration with ID: %s (was it deleted?)' % collection_node.exploration_id)
    if rights_manager.is_collection_public(collection.id):
        validate_exps_in_collection_are_public(collection)
    collection_model = collection_models.CollectionModel.get(collection.id, strict=True)
    if collection.version > collection_model.version:
        raise Exception('Unexpected error: trying to update version %s of collection from version %s. Please reload the page and try again.' % (collection_model.version, collection.version))
    if collection.version < collection_model.version:
        raise Exception('Trying to update version %s of collection from version %s, which is too old. Please reload the page and try again.' % (collection_model.version, collection.version))
    collection_model.category = collection.category
    collection_model.title = collection.title
    collection_model.objective = collection.objective
    collection_model.language_code = collection.language_code
    collection_model.tags = collection.tags
    collection_model.schema_version = collection.schema_version
    collection_model.collection_contents = {'nodes': [collection_node.to_dict() for collection_node in collection.nodes]}
    collection_model.commit(committer_id, commit_message, change_list)
    caching_services.delete_multi(caching_services.CACHE_NAMESPACE_COLLECTION, None, [collection.id])
    index_collections_given_ids([collection.id])
    collection.version += 1

def _create_collection(committer_id: str, collection: collection_domain.Collection, commit_message: str, commit_cmds: Sequence[Mapping[str, change_domain.AcceptableChangeDictTypes]]) -> None:
    if False:
        print('Hello World!')
    'Creates a new collection, and ensures that rights for a new collection\n    are saved first. This is because _save_collection() depends on the rights\n    object being present to tell it whether to do strict validation or not.\n\n    Args:\n        committer_id: str. ID of the committer.\n        collection: Collection. Collection domain object.\n        commit_message: str. A description of changes made to the collection.\n        commit_cmds: list(dict). A list of change commands made to the given\n            collection.\n    '
    collection.validate(strict=False)
    rights_manager.create_new_collection_rights(collection.id, committer_id)
    model = collection_models.CollectionModel(id=collection.id, category=collection.category, title=collection.title, objective=collection.objective, language_code=collection.language_code, tags=collection.tags, schema_version=collection.schema_version, collection_contents={'nodes': [collection_node.to_dict() for collection_node in collection.nodes]})
    model.commit(committer_id, commit_message, commit_cmds)
    collection.version += 1
    regenerate_collection_summary_with_new_contributor(collection.id, committer_id)

def save_new_collection(committer_id: str, collection: collection_domain.Collection) -> None:
    if False:
        return 10
    'Saves a new collection.\n\n    Args:\n        committer_id: str. ID of the committer.\n        collection: Collection. Collection to be saved.\n    '
    commit_message = "New collection created with title '%s'." % collection.title
    _create_collection(committer_id, collection, commit_message, [{'cmd': CMD_CREATE_NEW, 'title': collection.title, 'category': collection.category}])

def delete_collection(committer_id: str, collection_id: str, force_deletion: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    'Deletes the collection with the given collection_id.\n\n    IMPORTANT: Callers of this function should ensure that committer_id has\n    permissions to delete this collection, prior to calling this function.\n\n    Args:\n        committer_id: str. ID of the committer.\n        collection_id: str. ID of the collection to be deleted.\n        force_deletion: bool. If true, the collection and its history are fully\n            deleted and are unrecoverable. Otherwise, the collection and all\n            its history are marked as deleted, but the corresponding models are\n            still retained in the datastore. This last option is the preferred\n            one.\n    '
    delete_collections(committer_id, [collection_id], force_deletion=force_deletion)

def delete_collections(committer_id: str, collection_ids: List[str], force_deletion: bool=False) -> None:
    if False:
        print('Hello World!')
    'Deletes the collections with the given collection_ids.\n\n    IMPORTANT: Callers of this function should ensure that committer_id has\n    permissions to delete this collection, prior to calling this function.\n\n    Args:\n        committer_id: str. ID of the committer.\n        collection_ids: list(str). IDs of the collections to be deleted.\n        force_deletion: bool. If true, the collections and its histories are\n            fully deleted and are unrecoverable. Otherwise, the collections and\n            all its histories are marked as deleted, but the corresponding\n            models are still retained in the datastore.\n    '
    collection_models.CollectionRightsModel.delete_multi(collection_ids, committer_id, '', force_deletion=force_deletion)
    collection_models.CollectionModel.delete_multi(collection_ids, committer_id, feconf.COMMIT_MESSAGE_EXPLORATION_DELETED, force_deletion=force_deletion)
    caching_services.delete_multi(caching_services.CACHE_NAMESPACE_COLLECTION, None, collection_ids)
    search_services.delete_collections_from_search_index(collection_ids)
    delete_collection_summaries(collection_ids)
    activity_services.remove_featured_activities(constants.ACTIVITY_TYPE_COLLECTION, collection_ids)

def get_collection_snapshots_metadata(collection_id: str) -> List[SnapshotsMetadataDict]:
    if False:
        return 10
    'Returns the snapshots for this collection, as dicts.\n\n    Args:\n        collection_id: str. The id of the collection in question.\n\n    Returns:\n        list of dicts, each representing a recent snapshot. Each dict has the\n        following keys: committer_id, commit_message, commit_cmds, commit_type,\n        created_on_ms, version_number. The version numbers are consecutive and\n        in ascending order. There are collection.version_number items in the\n        returned list.\n    '
    collection = get_collection_by_id(collection_id)
    current_version = collection.version
    version_nums = list(range(1, current_version + 1))
    return collection_models.CollectionModel.get_snapshots_metadata(collection_id, version_nums)

def publish_collection_and_update_user_profiles(committer: user_domain.UserActionsInfo, collection_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Publishes the collection with publish_collection() function in\n    rights_manager.py, as well as updates first_contribution_msec.\n\n    It is the responsibility of the caller to check that the collection is\n    valid prior to publication.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the committer.\n        collection_id: str. ID of the collection to be published.\n\n    Raises:\n        Exception. No collection summary model exists for the given id.\n    '
    rights_manager.publish_collection(committer, collection_id)
    contribution_time_msec = utils.get_current_time_in_millisecs()
    collection_summary = get_collection_summary_by_id(collection_id)
    if collection_summary is None:
        raise Exception('No collection summary model exists for the given id: %s' % collection_id)
    contributor_ids = collection_summary.contributor_ids
    user_settings_models = []
    for contributor in contributor_ids:
        user_settings = user_services.get_user_settings(contributor, strict=False)
        if user_settings is not None:
            user_settings.update_first_contribution_msec(contribution_time_msec)
            user_settings_models.append(user_services.convert_to_user_settings_model(user_settings))
    datastore_services.update_timestamps_multi(user_settings_models)
    datastore_services.put_multi(user_settings_models)

def update_collection(committer_id: str, collection_id: str, change_list: Sequence[Mapping[str, change_domain.AcceptableChangeDictTypes]], commit_message: Optional[str]) -> None:
    if False:
        while True:
            i = 10
    'Updates a collection. Commits changes.\n\n    Args:\n        committer_id: str. The id of the user who is performing the update\n            action.\n        collection_id: str. The collection id.\n        change_list: list(dict). Each entry represents a CollectionChange\n            object. These changes are applied in sequence to produce the\n            resulting collection.\n        commit_message: str or None. A description of changes made to the\n            collection. For published collections, this must be present; for\n            unpublished collections, it may be equal to None.\n\n    Raises:\n        ValueError. The collection is public but no commit message received.\n    '
    is_public = rights_manager.is_collection_public(collection_id)
    if is_public and (not commit_message):
        raise ValueError('Collection is public so expected a commit message but received none.')
    collection = apply_change_list(collection_id, change_list)
    _save_collection(committer_id, collection, commit_message, change_list)
    regenerate_collection_summary_with_new_contributor(collection.id, committer_id)
    if not rights_manager.is_collection_private(collection.id) and committer_id != feconf.MIGRATION_BOT_USER_ID:
        user_settings = user_services.get_user_settings(committer_id)
        if user_settings is not None:
            user_settings.update_first_contribution_msec(utils.get_current_time_in_millisecs())
            user_services.save_user_settings(user_settings)

def regenerate_collection_summary_with_new_contributor(collection_id: str, contributor_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Regenerate a summary of the given collection and add a new contributor to\n    the contributors summary. If the summary does not exist, this function\n    generates a new one.\n\n    Args:\n        collection_id: str. ID of the collection.\n        contributor_id: str. ID of the contributor to be added to the collection\n            summary.\n    '
    collection = get_collection_by_id(collection_id)
    collection_summary = _compute_summary_of_collection(collection)
    collection_summary.add_contribution_by_user(contributor_id)
    save_collection_summary(collection_summary)

def regenerate_collection_and_contributors_summaries(collection_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Regenerate a summary of the given collection and also regenerate\n    the contributors summary from the snapshots. If the summary does not exist,\n    this function generates a new one.\n\n    Args:\n        collection_id: str. ID of the collection.\n    '
    collection = get_collection_by_id(collection_id)
    collection_summary = _compute_summary_of_collection(collection)
    collection_summary.contributors_summary = compute_collection_contributors_summary(collection_summary.id)
    save_collection_summary(collection_summary)

def _compute_summary_of_collection(collection: collection_domain.Collection) -> collection_domain.CollectionSummary:
    if False:
        i = 10
        return i + 15
    'Create a CollectionSummary domain object for a given Collection domain\n    object and return it.\n\n    Args:\n        collection: Collection. The domain object.\n\n    Returns:\n        CollectionSummary. The computed summary for the given collection.\n\n    Raises:\n        Exception. No data available for when the collection was last_updated.\n        Exception. No data available for when the collection was created.\n    '
    collection_rights = collection_models.CollectionRightsModel.get_by_id(collection.id)
    collection_summary_model = collection_models.CollectionSummaryModel.get_by_id(collection.id)
    contributors_summary = collection_summary_model.contributors_summary if collection_summary_model else {}
    contributor_ids = list(contributors_summary.keys())
    collection_model_last_updated = collection.last_updated
    collection_model_created_on = collection.created_on
    collection_model_node_count = len(collection.nodes)
    if collection_model_last_updated is None:
        raise Exception('No data available for when the collection was last_updated.')
    if collection_model_created_on is None:
        raise Exception('No data available for when the collection was created.')
    collection_summary = collection_domain.CollectionSummary(collection.id, collection.title, collection.category, collection.objective, collection.language_code, collection.tags, collection_rights.status, collection_rights.community_owned, collection_rights.owner_ids, collection_rights.editor_ids, collection_rights.viewer_ids, contributor_ids, contributors_summary, collection.version, collection_model_node_count, collection_model_created_on, collection_model_last_updated)
    return collection_summary

def compute_collection_contributors_summary(collection_id: str) -> Dict[str, int]:
    if False:
        while True:
            i = 10
    "Computes the contributors' summary for a given collection.\n\n    Args:\n        collection_id: str. ID of the collection.\n\n    Returns:\n        dict. A dict whose keys are user_ids and whose values are the number of\n        (non-revert) commits made to the given collection by that user_id.\n        This does not count commits which have since been reverted.\n    "
    snapshots_metadata = get_collection_snapshots_metadata(collection_id)
    current_version = len(snapshots_metadata)
    contributors_summary: Dict[str, int] = collections.defaultdict(int)
    while True:
        snapshot_metadata = snapshots_metadata[current_version - 1]
        committer_id = snapshot_metadata['committer_id']
        if committer_id not in constants.SYSTEM_USER_IDS:
            contributors_summary[committer_id] += 1
        if current_version == 1:
            break
        current_version -= 1
    contributor_ids = list(contributors_summary)
    users_settings = user_services.get_users_settings(contributor_ids)
    for (contributor_id, user_settings) in zip(contributor_ids, users_settings):
        if user_settings is None:
            del contributors_summary[contributor_id]
    return contributors_summary

def save_collection_summary(collection_summary: collection_domain.CollectionSummary) -> None:
    if False:
        i = 10
        return i + 15
    'Save a collection summary domain object as a CollectionSummaryModel\n    entity in the datastore.\n\n    Args:\n        collection_summary: CollectionSummary. The collection summary\n            object to be saved in the datastore.\n    '
    collection_summary_dict = {'title': collection_summary.title, 'category': collection_summary.category, 'objective': collection_summary.objective, 'language_code': collection_summary.language_code, 'tags': collection_summary.tags, 'status': collection_summary.status, 'community_owned': collection_summary.community_owned, 'owner_ids': collection_summary.owner_ids, 'editor_ids': collection_summary.editor_ids, 'viewer_ids': collection_summary.viewer_ids, 'contributor_ids': list(collection_summary.contributors_summary.keys()), 'contributors_summary': collection_summary.contributors_summary, 'version': collection_summary.version, 'node_count': collection_summary.node_count, 'collection_model_last_updated': collection_summary.collection_model_last_updated, 'collection_model_created_on': collection_summary.collection_model_created_on}
    collection_summary_model = collection_models.CollectionSummaryModel.get_by_id(collection_summary.id)
    if collection_summary_model is not None:
        collection_summary_model.populate(**collection_summary_dict)
        collection_summary_model.update_timestamps()
        collection_summary_model.put()
    else:
        collection_summary_dict['id'] = collection_summary.id
        model = collection_models.CollectionSummaryModel(**collection_summary_dict)
        model.update_timestamps()
        model.put()

def delete_collection_summaries(collection_ids: List[str]) -> None:
    if False:
        while True:
            i = 10
    'Delete multiple collection summary models.\n\n    Args:\n        collection_ids: list(str). IDs of the collections whose collection\n            summaries are to be deleted.\n    '
    summary_models = collection_models.CollectionSummaryModel.get_multi(collection_ids)
    existing_summary_models = [summary_model for summary_model in summary_models if summary_model is not None]
    collection_models.CollectionSummaryModel.delete_multi(existing_summary_models)

def save_new_collection_from_yaml(committer_id: str, yaml_content: str, collection_id: str) -> collection_domain.Collection:
    if False:
        print('Hello World!')
    'Saves a new collection from a yaml content string.\n\n    Args:\n        committer_id: str. ID of the committer.\n        yaml_content: str. The yaml content string specifying a collection.\n        collection_id: str. ID of the saved collection.\n\n    Returns:\n        Collection. The domain object.\n    '
    collection = collection_domain.Collection.from_yaml(collection_id, yaml_content)
    commit_message = "New collection created from YAML file with title '%s'." % collection.title
    _create_collection(committer_id, collection, commit_message, [{'cmd': CMD_CREATE_NEW, 'title': collection.title, 'category': collection.category}])
    return collection

def delete_demo(collection_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Deletes a single demo collection.\n\n    Args:\n        collection_id: str. ID of the demo collection to be deleted.\n\n    Raises:\n        Exception. Invalid demo collection ID.\n    '
    if not collection_domain.Collection.is_demo_collection_id(collection_id):
        raise Exception('Invalid demo collection id %s' % collection_id)
    collection = get_collection_by_id(collection_id, strict=False)
    if not collection:
        logging.info('Collection with id %s was not deleted, because it does not exist.' % collection_id)
    else:
        delete_collection(feconf.SYSTEM_COMMITTER_ID, collection_id, force_deletion=True)

def load_demo(collection_id: str) -> None:
    if False:
        print('Hello World!')
    'Loads a demo collection.\n\n    The resulting collection will have version 2 (one for its initial\n    creation and one for its subsequent modification).\n\n    Args:\n        collection_id: str. ID of the collection to be loaded.\n    '
    delete_demo(collection_id)
    demo_filepath = os.path.join(feconf.SAMPLE_COLLECTIONS_DIR, feconf.DEMO_COLLECTIONS[collection_id])
    yaml_content = utils.get_file_contents(demo_filepath)
    collection = save_new_collection_from_yaml(feconf.SYSTEM_COMMITTER_ID, yaml_content, collection_id)
    system_user = user_services.get_system_user()
    publish_collection_and_update_user_profiles(system_user, collection_id)
    index_collections_given_ids([collection_id])
    for collection_node in collection.nodes:
        exp_id = collection_node.exploration_id
        if exp_fetchers.get_exploration_by_id(exp_id, strict=False) is None:
            exp_services.load_demo(exp_id)
    logging.info('Collection with id %s was loaded.' % collection_id)

def index_collections_given_ids(collection_ids: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Adds the given collections to the search index.\n\n    Args:\n        collection_ids: list(str). List of collection ids whose collections are\n            to be indexed.\n    '
    collection_summaries = get_collection_summaries_matching_ids(collection_ids)
    search_services.index_collection_summaries([collection_summary for collection_summary in collection_summaries if collection_summary is not None])