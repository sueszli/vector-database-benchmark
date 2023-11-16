"""Commands for operations on subtopic pages, and related models."""
from __future__ import annotations
import copy
from core import feconf
from core.domain import change_domain
from core.domain import classroom_config_services
from core.domain import learner_group_services
from core.domain import skill_services
from core.domain import subtopic_page_domain
from core.domain import topic_fetchers
from core.platform import models
from typing import Dict, List, Literal, Optional, Sequence, overload
MYPY = False
if MYPY:
    from mypy_imports import subtopic_models
(subtopic_models,) = models.Registry.import_models([models.Names.SUBTOPIC])

def _migrate_page_contents_to_latest_schema(versioned_page_contents: subtopic_page_domain.VersionedSubtopicPageContentsDict) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Holds the responsibility of performing a step-by-step, sequential update\n    of the page contents structure based on the schema version of the input\n    page contents dictionary. If the current page_contents schema changes, a\n    new conversion function must be added and some code appended to this\n    function to account for that new version.\n\n    Args:\n        versioned_page_contents: dict. A dict with two keys:\n          - schema_version: int. The schema version for the page_contents dict.\n          - page_contents: dict. The dict comprising the page contents.\n\n    Raises:\n        Exception. The schema version of the page_contents is outside of what\n            is supported at present.\n    '
    page_contents_schema_version = versioned_page_contents['schema_version']
    if not 1 <= page_contents_schema_version <= feconf.CURRENT_SUBTOPIC_PAGE_CONTENTS_SCHEMA_VERSION:
        raise Exception('Sorry, we can only process v1-v%d page schemas at present.' % feconf.CURRENT_SUBTOPIC_PAGE_CONTENTS_SCHEMA_VERSION)
    while page_contents_schema_version < feconf.CURRENT_SUBTOPIC_PAGE_CONTENTS_SCHEMA_VERSION:
        subtopic_page_domain.SubtopicPage.update_page_contents_from_model(versioned_page_contents, page_contents_schema_version)
        page_contents_schema_version += 1

def get_subtopic_page_from_model(subtopic_page_model: subtopic_models.SubtopicPageModel) -> subtopic_page_domain.SubtopicPage:
    if False:
        for i in range(10):
            print('nop')
    'Returns a domain object for an SubtopicPage given a subtopic page model.\n\n    Args:\n        subtopic_page_model: SubtopicPageModel. The subtopic page model to get\n            the corresponding domain object.\n\n    Returns:\n        SubtopicPage. The domain object corresponding to the given model object.\n    '
    versioned_page_contents: subtopic_page_domain.VersionedSubtopicPageContentsDict = {'schema_version': subtopic_page_model.page_contents_schema_version, 'page_contents': copy.deepcopy(subtopic_page_model.page_contents)}
    if subtopic_page_model.page_contents_schema_version != feconf.CURRENT_SUBTOPIC_PAGE_CONTENTS_SCHEMA_VERSION:
        _migrate_page_contents_to_latest_schema(versioned_page_contents)
    return subtopic_page_domain.SubtopicPage(subtopic_page_model.id, subtopic_page_model.topic_id, subtopic_page_domain.SubtopicPageContents.from_dict(versioned_page_contents['page_contents']), versioned_page_contents['schema_version'], subtopic_page_model.language_code, subtopic_page_model.version)

@overload
def get_subtopic_page_by_id(topic_id: str, subtopic_id: int) -> subtopic_page_domain.SubtopicPage:
    if False:
        return 10
    ...

@overload
def get_subtopic_page_by_id(topic_id: str, subtopic_id: int, *, version: int) -> subtopic_page_domain.SubtopicPage:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_subtopic_page_by_id(topic_id: str, subtopic_id: int, *, strict: Literal[True], version: Optional[int]=...) -> subtopic_page_domain.SubtopicPage:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_subtopic_page_by_id(topic_id: str, subtopic_id: int, *, strict: Literal[False], version: Optional[int]=...) -> Optional[subtopic_page_domain.SubtopicPage]:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_subtopic_page_by_id(topic_id: str, subtopic_id: int, *, strict: bool=..., version: Optional[int]=...) -> Optional[subtopic_page_domain.SubtopicPage]:
    if False:
        for i in range(10):
            print('nop')
    ...

def get_subtopic_page_by_id(topic_id: str, subtopic_id: int, strict: bool=True, version: Optional[int]=None) -> Optional[subtopic_page_domain.SubtopicPage]:
    if False:
        while True:
            i = 10
    'Returns a domain object representing a subtopic page.\n\n    Args:\n        topic_id: str. ID of the topic that the subtopic is a part of.\n        subtopic_id: int. The id of the subtopic.\n        strict: bool. Whether to fail noisily if no subtopic page with the given\n            id exists in the datastore.\n        version: str or None. The version number of the subtopic page.\n\n    Returns:\n        SubtopicPage or None. The domain object representing a subtopic page\n        with the given id, or None if it does not exist.\n    '
    subtopic_page_id = subtopic_page_domain.SubtopicPage.get_subtopic_page_id(topic_id, subtopic_id)
    subtopic_page_model = subtopic_models.SubtopicPageModel.get(subtopic_page_id, strict=strict, version=version)
    if subtopic_page_model:
        subtopic_page = get_subtopic_page_from_model(subtopic_page_model)
        return subtopic_page
    else:
        return None

def get_subtopic_pages_with_ids(topic_id: str, subtopic_ids: List[int]) -> List[Optional[subtopic_page_domain.SubtopicPage]]:
    if False:
        print('Hello World!')
    'Returns a list of domain objects with given ids.\n\n    Args:\n        topic_id: str. ID of the topic that the subtopics belong to.\n        subtopic_ids: list(int). The ids of the subtopics.\n\n    Returns:\n        list(SubtopicPage) or None. The list of domain objects representing the\n        subtopic pages corresponding to given ids list or None if none exist.\n    '
    subtopic_page_ids = []
    for subtopic_id in subtopic_ids:
        subtopic_page_ids.append(subtopic_page_domain.SubtopicPage.get_subtopic_page_id(topic_id, subtopic_id))
    subtopic_page_models = subtopic_models.SubtopicPageModel.get_multi(subtopic_page_ids)
    subtopic_pages: List[Optional[subtopic_page_domain.SubtopicPage]] = []
    for subtopic_page_model in subtopic_page_models:
        if subtopic_page_model is None:
            subtopic_pages.append(subtopic_page_model)
        else:
            subtopic_pages.append(get_subtopic_page_from_model(subtopic_page_model))
    return subtopic_pages

@overload
def get_subtopic_page_contents_by_id(topic_id: str, subtopic_id: int) -> subtopic_page_domain.SubtopicPageContents:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_subtopic_page_contents_by_id(topic_id: str, subtopic_id: int, *, strict: Literal[True]) -> subtopic_page_domain.SubtopicPageContents:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_subtopic_page_contents_by_id(topic_id: str, subtopic_id: int, *, strict: Literal[False]) -> Optional[subtopic_page_domain.SubtopicPageContents]:
    if False:
        i = 10
        return i + 15
    ...

def get_subtopic_page_contents_by_id(topic_id: str, subtopic_id: int, strict: bool=True) -> Optional[subtopic_page_domain.SubtopicPageContents]:
    if False:
        while True:
            i = 10
    'Returns the page contents of a subtopic\n\n    Args:\n        topic_id: str. ID of the topic that the subtopic belong to.\n        subtopic_id: int. The id of the subtopic.\n        strict: bool. Whether to fail noisily if no subtopic page with the given\n            id exists in the datastore.\n\n    Returns:\n        SubtopicPageContents or None. The page contents for a subtopic page,\n        or None if subtopic page does not exist.\n    '
    subtopic_page = get_subtopic_page_by_id(topic_id, subtopic_id, strict=strict)
    if subtopic_page is not None:
        return subtopic_page.page_contents
    else:
        return None

def save_subtopic_page(committer_id: str, subtopic_page: subtopic_page_domain.SubtopicPage, commit_message: Optional[str], change_list: Sequence[change_domain.BaseChange]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Validates a subtopic page and commits it to persistent storage. If\n    successful, increments the version number of the incoming subtopic page\n    domain object by 1.\n\n    Args:\n        committer_id: str. ID of the given committer.\n        subtopic_page: SubtopicPage. The subtopic page domain object to be\n            saved.\n        commit_message: str|None. The commit description message, for\n            unpublished topics, it may be equal to None.\n        change_list: list(SubtopicPageChange). List of changes applied to a\n            subtopic page.\n\n    Raises:\n        Exception. Received invalid change list.\n        Exception. The subtopic page model and the incoming subtopic page domain\n            object have different version numbers.\n    '
    if not change_list:
        raise Exception('Unexpected error: received an invalid change list when trying to save topic %s: %s' % (subtopic_page.id, change_list))
    subtopic_page.validate()
    subtopic_page_model = subtopic_models.SubtopicPageModel.get(subtopic_page.id, strict=False)
    if subtopic_page_model is None:
        subtopic_page_model = subtopic_models.SubtopicPageModel(id=subtopic_page.id)
    else:
        if subtopic_page.version > subtopic_page_model.version:
            raise Exception('Unexpected error: trying to update version %s of topic from version %s. Please reload the page and try again.' % (subtopic_page_model.version, subtopic_page.version))
        if subtopic_page.version < subtopic_page_model.version:
            raise Exception('Trying to update version %s of topic from version %s, which is too old. Please reload the page and try again.' % (subtopic_page_model.version, subtopic_page.version))
    subtopic_page_model.topic_id = subtopic_page.topic_id
    subtopic_page_model.page_contents = subtopic_page.page_contents.to_dict()
    subtopic_page_model.language_code = subtopic_page.language_code
    subtopic_page_model.page_contents_schema_version = subtopic_page.page_contents_schema_version
    change_dicts = [change.to_dict() for change in change_list]
    subtopic_page_model.commit(committer_id, commit_message, change_dicts)
    subtopic_page.version += 1

def delete_subtopic_page(committer_id: str, topic_id: str, subtopic_id: int, force_deletion: bool=False) -> None:
    if False:
        print('Hello World!')
    'Delete a topic summary model.\n\n    Args:\n        committer_id: str. The user who is deleting the subtopic page.\n        topic_id: str. The ID of the topic that this subtopic belongs to.\n        subtopic_id: int. ID of the subtopic which was removed.\n        force_deletion: bool. If true, the subtopic page and its history are\n            fully deleted and are unrecoverable. Otherwise, the subtopic page\n            and all its history are marked as deleted, but the corresponding\n            models are still retained in the datastore. This last option is the\n            preferred one.\n    '
    subtopic_page_id = subtopic_page_domain.SubtopicPage.get_subtopic_page_id(topic_id, subtopic_id)
    subtopic_models.SubtopicPageModel.get(subtopic_page_id).delete(committer_id, feconf.COMMIT_MESSAGE_SUBTOPIC_PAGE_DELETED, force_deletion=force_deletion)
    learner_group_services.remove_subtopic_page_reference_from_learner_groups(topic_id, subtopic_id)

def get_topic_ids_from_subtopic_page_ids(subtopic_page_ids: List[str]) -> List[str]:
    if False:
        return 10
    'Returns the topic ids corresponding to the given set of subtopic page\n    ids.\n\n    Args:\n        subtopic_page_ids: list(str). The ids of the subtopic pages.\n\n    Returns:\n        list(str). The topic ids corresponding to the given subtopic page ids.\n        The returned list of topic ids is deduplicated and ordered\n        alphabetically.\n    '
    return sorted(list({subtopic_page_id.split(':')[0] for subtopic_page_id in subtopic_page_ids}))

def get_multi_users_subtopic_pages_progress(user_ids: List[str], subtopic_page_ids: List[str]) -> Dict[str, List[subtopic_page_domain.SubtopicPageSummaryDict]]:
    if False:
        print('Hello World!')
    'Returns the progress of the given user on the given subtopic pages.\n\n    Args:\n        user_ids: list(str). The ids of the users.\n        subtopic_page_ids: list(str). The ids of the subtopic pages.\n\n    Returns:\n        dict(str, list(SubtopicPageSummaryDict)). User IDs as keys and Subtopic\n        Page Summary domain object dictionaries containing details of the\n        subtopic page and users mastery in it as values.\n    '
    topic_ids = get_topic_ids_from_subtopic_page_ids(subtopic_page_ids)
    topics = topic_fetchers.get_topics_by_ids(topic_ids, strict=True)
    all_skill_ids_lists = [topic.get_all_skill_ids() for topic in topics if topic]
    all_skill_ids = list({skill_id for skill_list in all_skill_ids_lists for skill_id in skill_list})
    all_users_skill_mastery_dicts = skill_services.get_multi_users_skills_mastery(user_ids, all_skill_ids)
    all_users_subtopic_prog_summaries: Dict[str, List[subtopic_page_domain.SubtopicPageSummaryDict]] = {user_id: [] for user_id in user_ids}
    for topic in topics:
        for subtopic in topic.subtopics:
            subtopic_page_id = '{}:{}'.format(topic.id, subtopic.id)
            if subtopic_page_id not in subtopic_page_ids:
                continue
            for (user_id, skills_mastery_dict) in all_users_skill_mastery_dicts.items():
                skill_mastery_dict = {skill_id: mastery for (skill_id, mastery) in skills_mastery_dict.items() if mastery is not None and skill_id in subtopic.skill_ids}
                subtopic_mastery: Optional[float] = None
                if skill_mastery_dict:
                    subtopic_mastery = sum(skill_mastery_dict.values()) / len(skill_mastery_dict)
                all_users_subtopic_prog_summaries[user_id].append({'subtopic_id': subtopic.id, 'subtopic_title': subtopic.title, 'parent_topic_id': topic.id, 'parent_topic_name': topic.name, 'thumbnail_filename': subtopic.thumbnail_filename, 'thumbnail_bg_color': subtopic.thumbnail_bg_color, 'subtopic_mastery': subtopic_mastery, 'parent_topic_url_fragment': topic.url_fragment, 'classroom_url_fragment': classroom_config_services.get_classroom_url_fragment_for_topic_id(topic.id)})
    return all_users_subtopic_prog_summaries

def get_learner_group_syllabus_subtopic_page_summaries(subtopic_page_ids: List[str]) -> List[subtopic_page_domain.SubtopicPageSummaryDict]:
    if False:
        return 10
    'Returns summary dicts corresponding to the given subtopic page ids.\n\n    Args:\n        subtopic_page_ids: list(str). The ids of the subtopic pages.\n\n    Returns:\n        list(SubtopicPageSummaryDict). The summary dicts corresponding to the\n        given subtopic page ids.\n    '
    topic_ids = get_topic_ids_from_subtopic_page_ids(subtopic_page_ids)
    topics = topic_fetchers.get_topics_by_ids(topic_ids, strict=True)
    all_learner_group_subtopic_page_summaries: List[subtopic_page_domain.SubtopicPageSummaryDict] = []
    for topic in topics:
        for subtopic in topic.subtopics:
            subtopic_page_id = '{}:{}'.format(topic.id, subtopic.id)
            if subtopic_page_id not in subtopic_page_ids:
                continue
            all_learner_group_subtopic_page_summaries.append({'subtopic_id': subtopic.id, 'subtopic_title': subtopic.title, 'parent_topic_id': topic.id, 'parent_topic_name': topic.name, 'thumbnail_filename': subtopic.thumbnail_filename, 'thumbnail_bg_color': subtopic.thumbnail_bg_color, 'subtopic_mastery': None, 'parent_topic_url_fragment': topic.url_fragment, 'classroom_url_fragment': None})
    return all_learner_group_subtopic_page_summaries

def populate_subtopic_page_model_fields(subtopic_page_model: subtopic_models.SubtopicPageModel, subtopic_page: subtopic_page_domain.SubtopicPage) -> subtopic_models.SubtopicPageModel:
    if False:
        print('Hello World!')
    'Populate subtopic page model with the data from subtopic page object.\n\n    Args:\n        subtopic_page_model: SubtopicPageModel. The model to populate.\n        subtopic_page: SubtopicPage. The subtopic page domain object which\n            should be used to populate the model.\n\n    Returns:\n        SubtopicPageModel. Populated model.\n    '
    subtopic_page_model.topic_id = subtopic_page.topic_id
    subtopic_page_model.page_contents = subtopic_page.page_contents.to_dict()
    subtopic_page_model.page_contents_schema_version = subtopic_page.page_contents_schema_version
    subtopic_page_model.language_code = subtopic_page.language_code
    return subtopic_page_model