"""Commands for operating on lists of activity references."""
from __future__ import annotations
import logging
from core import feconf
from core.constants import constants
from core.domain import activity_domain
from core.platform import models
from typing import List, Tuple
MYPY = False
if MYPY:
    from mypy_imports import activity_models
(activity_models,) = models.Registry.import_models([models.Names.ACTIVITY])

def get_featured_activity_references() -> List[activity_domain.ActivityReference]:
    if False:
        return 10
    'Gets a list of ActivityReference domain models.\n\n    Returns:\n        list(ActivityReference). A list of all ActivityReference domain objects\n        that are currently featured.\n    '
    featured_model_instance = activity_models.ActivityReferencesModel.get_or_create(feconf.ACTIVITY_REFERENCE_LIST_FEATURED)
    return [activity_domain.ActivityReference(reference['type'], reference['id']) for reference in featured_model_instance.activity_references]

def update_featured_activity_references(featured_activity_references: List[activity_domain.ActivityReference]) -> None:
    if False:
        while True:
            i = 10
    "Updates the current list of featured activity references.\n\n    Args:\n        featured_activity_references: list(ActivityReference). A list of\n            ActivityReference domain objects representing the full list of\n            'featured' activities.\n\n    Raises:\n        Exception. The input list of ActivityReference domain objects has\n            duplicates.\n    "
    for activity_reference in featured_activity_references:
        activity_reference.validate()
    activity_hashes = [reference.get_hash() for reference in featured_activity_references]
    if len(activity_hashes) != len(set(activity_hashes)):
        raise Exception('The activity reference list should not have duplicates.')
    featured_model_instance = activity_models.ActivityReferencesModel.get_or_create(feconf.ACTIVITY_REFERENCE_LIST_FEATURED)
    featured_model_instance.activity_references = [reference.to_dict() for reference in featured_activity_references]
    featured_model_instance.update_timestamps()
    featured_model_instance.put()

def remove_featured_activity(activity_type: str, activity_id: str) -> None:
    if False:
        return 10
    'Removes the specified activity reference from the list of featured\n    activity references.\n\n    Args:\n        activity_type: str. The type of the activity to remove.\n        activity_id: str. The id of the activity to remove.\n    '
    remove_featured_activities(activity_type, [activity_id])

def remove_featured_activities(activity_type: str, activity_ids: list[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Removes the specified activity references from the list of featured\n    activity references.\n\n    Args:\n        activity_type: str. The type of the activities to remove.\n        activity_ids: list(str). The ids of the activities to remove.\n    '
    featured_references = get_featured_activity_references()
    activity_references_ids_found = []
    new_activity_references = []
    for reference in featured_references:
        if reference.type != activity_type or reference.id not in activity_ids:
            new_activity_references.append(reference)
        else:
            activity_references_ids_found.append(reference.id)
    if activity_references_ids_found:
        for activity_id in activity_references_ids_found:
            logging.info('The %s with id %s was removed from the featured list.' % (activity_type, activity_id))
        update_featured_activity_references(new_activity_references)

def split_by_type(activity_references: List[activity_domain.ActivityReference]) -> Tuple[List[str], List[str]]:
    if False:
        while True:
            i = 10
    'Given a list of activity references, returns two lists: the first list\n    contains the exploration ids, and the second contains the collection ids.\n    The elements in each of the returned lists are in the same order as those\n    in the input list.\n\n    Args:\n        activity_references: list(ActivityReference). The domain object\n            containing exploration ids and collection ids.\n\n    Returns:\n        tuple(list(str), list(str)). A 2-tuple whose first element is a list of\n        all exploration ids represented in the input list, and whose second\n        element is a list of all collection ids represented in the input list.\n\n    Raises:\n        Exception. The activity reference type is invalid.\n    '
    (exploration_ids, collection_ids) = ([], [])
    for activity_reference in activity_references:
        if activity_reference.type == constants.ACTIVITY_TYPE_EXPLORATION:
            exploration_ids.append(activity_reference.id)
        elif activity_reference.type == constants.ACTIVITY_TYPE_COLLECTION:
            collection_ids.append(activity_reference.id)
        else:
            raise Exception('Invalid activity reference: (%s, %s)' % (activity_reference.type, activity_reference.id))
    return (exploration_ids, collection_ids)