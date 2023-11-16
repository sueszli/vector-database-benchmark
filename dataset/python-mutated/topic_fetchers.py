"""Getter commands for for topic models."""
from __future__ import annotations
import copy
from core import feconf
from core import utils
from core.domain import caching_services
from core.domain import classroom_config_services
from core.domain import story_domain
from core.domain import story_fetchers
from core.domain import topic_domain
from core.platform import models
from typing import Dict, List, Literal, Optional, Sequence, Set, TypedDict, overload
MYPY = False
if MYPY:
    from mypy_imports import topic_models
(topic_models,) = models.Registry.import_models([models.Names.TOPIC])

def _migrate_subtopics_to_latest_schema(versioned_subtopics: topic_domain.VersionedSubtopicsDict, topic_id: str) -> None:
    if False:
        while True:
            i = 10
    "Holds the responsibility of performing a step-by-step, sequential update\n    of the subtopics structure based on the schema version of the input\n    subtopics dictionary. If the current subtopics schema changes, a\n    new conversion function must be added and some code appended to this\n    function to account for that new version.\n\n    Args:\n        versioned_subtopics: dict. A dict with two keys:\n          - schema_version: int. The schema version for the subtopics dict.\n          - subtopics: list(dict). The list of dicts comprising the topic's\n              subtopics.\n        topic_id: str. The id of the topic to which the subtopics are part of.\n\n    Raises:\n        Exception. The schema version of subtopics is outside of what\n            is supported at present.\n    "
    subtopic_schema_version = versioned_subtopics['schema_version']
    if not 1 <= subtopic_schema_version <= feconf.CURRENT_SUBTOPIC_SCHEMA_VERSION:
        raise Exception('Sorry, we can only process v1-v%d subtopic schemas at present.' % feconf.CURRENT_SUBTOPIC_SCHEMA_VERSION)
    while subtopic_schema_version < feconf.CURRENT_SUBTOPIC_SCHEMA_VERSION:
        topic_domain.Topic.update_subtopics_from_model(versioned_subtopics, subtopic_schema_version, topic_id)
        subtopic_schema_version += 1

def _migrate_story_references_to_latest_schema(versioned_story_references: topic_domain.VersionedStoryReferencesDict) -> None:
    if False:
        return 10
    "Holds the responsibility of performing a step-by-step, sequential update\n    of the story reference structure based on the schema version of the input\n    story reference dictionary. If the current story reference schema changes, a\n    new conversion function must be added and some code appended to this\n    function to account for that new version.\n\n    Args:\n        versioned_story_references: dict. A dict with two keys:\n          - schema_version: int. The schema version for the story reference\n                dict.\n          - story_references: list(dict). The list of dicts comprising the\n                topic's story references.\n\n    Raises:\n        Exception. The schema version of story_references is outside of what\n            is supported at present.\n    "
    story_reference_schema_version = versioned_story_references['schema_version']
    if not 1 <= story_reference_schema_version <= feconf.CURRENT_STORY_REFERENCE_SCHEMA_VERSION:
        raise Exception('Sorry, we can only process v1-v%d story reference schemas at present.' % feconf.CURRENT_STORY_REFERENCE_SCHEMA_VERSION)
    while story_reference_schema_version < feconf.CURRENT_STORY_REFERENCE_SCHEMA_VERSION:
        topic_domain.Topic.update_story_references_from_model(versioned_story_references, story_reference_schema_version)
        story_reference_schema_version += 1

def get_topic_from_model(topic_model: topic_models.TopicModel) -> topic_domain.Topic:
    if False:
        for i in range(10):
            print('nop')
    'Returns a topic domain object given a topic model loaded\n    from the datastore.\n\n    Args:\n        topic_model: TopicModel. The topic model loaded from the\n            datastore.\n\n    Returns:\n        topic. A Topic domain object corresponding to the given\n        topic model.\n    '
    versioned_subtopics: topic_domain.VersionedSubtopicsDict = {'schema_version': topic_model.subtopic_schema_version, 'subtopics': copy.deepcopy(topic_model.subtopics)}
    versioned_canonical_story_references: topic_domain.VersionedStoryReferencesDict = {'schema_version': topic_model.story_reference_schema_version, 'story_references': topic_model.canonical_story_references}
    versioned_additional_story_references: topic_domain.VersionedStoryReferencesDict = {'schema_version': topic_model.story_reference_schema_version, 'story_references': topic_model.additional_story_references}
    if topic_model.subtopic_schema_version != feconf.CURRENT_SUBTOPIC_SCHEMA_VERSION:
        _migrate_subtopics_to_latest_schema(versioned_subtopics, topic_model.id)
    if topic_model.story_reference_schema_version != feconf.CURRENT_STORY_REFERENCE_SCHEMA_VERSION:
        _migrate_story_references_to_latest_schema(versioned_canonical_story_references)
        _migrate_story_references_to_latest_schema(versioned_additional_story_references)
    return topic_domain.Topic(topic_model.id, topic_model.name, topic_model.abbreviated_name, topic_model.url_fragment, topic_model.thumbnail_filename, topic_model.thumbnail_bg_color, topic_model.thumbnail_size_in_bytes, topic_model.description, [topic_domain.StoryReference.from_dict(reference) for reference in versioned_canonical_story_references['story_references']], [topic_domain.StoryReference.from_dict(reference) for reference in versioned_additional_story_references['story_references']], topic_model.uncategorized_skill_ids, [topic_domain.Subtopic.from_dict(subtopic) for subtopic in versioned_subtopics['subtopics']], versioned_subtopics['schema_version'], topic_model.next_subtopic_id, topic_model.language_code, topic_model.version, feconf.CURRENT_STORY_REFERENCE_SCHEMA_VERSION, topic_model.meta_tag_content, topic_model.practice_tab_is_displayed, topic_model.page_title_fragment_for_web, topic_model.skill_ids_for_diagnostic_test, topic_model.created_on, topic_model.last_updated)

@overload
def get_topic_by_id(topic_id: str) -> topic_domain.Topic:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_topic_by_id(topic_id: str, *, version: Optional[int]=None) -> topic_domain.Topic:
    if False:
        print('Hello World!')
    ...

@overload
def get_topic_by_id(topic_id: str, *, strict: Literal[True], version: Optional[int]=None) -> topic_domain.Topic:
    if False:
        while True:
            i = 10
    ...

@overload
def get_topic_by_id(topic_id: str, *, strict: Literal[False], version: Optional[int]=None) -> Optional[topic_domain.Topic]:
    if False:
        i = 10
        return i + 15
    ...

def get_topic_by_id(topic_id: str, strict: bool=True, version: Optional[int]=None) -> Optional[topic_domain.Topic]:
    if False:
        print('Hello World!')
    'Returns a domain object representing a topic.\n\n    Args:\n        topic_id: str. ID of the topic.\n        strict: bool. Whether to fail noisily if no topic with the given\n            id exists in the datastore.\n        version: int or None. The version number of the topic to be\n            retrieved. If it is None, the latest version will be retrieved.\n\n    Returns:\n        Topic or None. The domain object representing a topic with the\n        given id, or None if it does not exist.\n    '
    sub_namespace: Optional[str] = str(version) if version else None
    cached_topic = caching_services.get_multi(caching_services.CACHE_NAMESPACE_TOPIC, sub_namespace, [topic_id]).get(topic_id)
    if cached_topic is not None:
        return cached_topic
    else:
        topic_model = topic_models.TopicModel.get(topic_id, strict=strict, version=version)
        if topic_model:
            topic = get_topic_from_model(topic_model)
            caching_services.set_multi(caching_services.CACHE_NAMESPACE_TOPIC, sub_namespace, {topic_id: topic})
            return topic
        else:
            return None

@overload
def get_topics_by_ids(topic_ids: List[str], *, strict: Literal[True]) -> List[topic_domain.Topic]:
    if False:
        while True:
            i = 10
    ...

@overload
def get_topics_by_ids(topic_ids: List[str]) -> List[Optional[topic_domain.Topic]]:
    if False:
        while True:
            i = 10
    ...

@overload
def get_topics_by_ids(topic_ids: List[str], *, strict: Literal[False]) -> List[Optional[topic_domain.Topic]]:
    if False:
        while True:
            i = 10
    ...

def get_topics_by_ids(topic_ids: List[str], strict: bool=False) -> Sequence[Optional[topic_domain.Topic]]:
    if False:
        i = 10
        return i + 15
    'Returns a list of topics matching the IDs provided.\n\n    Args:\n        topic_ids: list(str). List of IDs to get topics for.\n        strict: bool. Whether to fail noisily if no topic model exists\n            with a given ID exists in the datastore.\n\n    Returns:\n        list(Topic|None). The list of topics corresponding to given ids\n        (with None in place of topic ids corresponding to deleted topics).\n\n    Raises:\n        Exception. No topic model exists for the given topic_id.\n    '
    all_topic_models: List[Optional[topic_models.TopicModel]] = topic_models.TopicModel.get_multi(topic_ids)
    topics: List[Optional[topic_domain.Topic]] = []
    for (index, topic_model) in enumerate(all_topic_models):
        if topic_model is None:
            if strict:
                raise Exception('No topic model exists for the topic_id: %s' % topic_ids[index])
            topics.append(topic_model)
        if topic_model is not None:
            topics.append(get_topic_from_model(topic_model))
    return topics

@overload
def get_topic_by_name(topic_name: str, *, strict: Literal[True]=...) -> topic_domain.Topic:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_topic_by_name(topic_name: str, *, strict: Literal[False]=...) -> Optional[topic_domain.Topic]:
    if False:
        print('Hello World!')
    ...

def get_topic_by_name(topic_name: str, strict: bool=False) -> Optional[topic_domain.Topic]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a domain object representing a topic.\n\n    Args:\n        topic_name: str. The name of the topic.\n        strict: bool. Whether to fail noisily if no Topic exists for\n            the given topic name.\n\n    Returns:\n        Topic or None. The domain object representing a topic with the\n        given id, or None if it does not exist.\n\n    Raises:\n        Exception. No Topic exists for the given topic name.\n    '
    topic_model: Optional[topic_models.TopicModel] = topic_models.TopicModel.get_by_name(topic_name)
    if topic_model is None:
        if strict:
            raise Exception('No Topic exists for the given topic name: %s' % topic_name)
        return None
    return get_topic_from_model(topic_model)

def get_topic_by_url_fragment(url_fragment: str) -> Optional[topic_domain.Topic]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a domain object representing a topic.\n\n    Args:\n        url_fragment: str. The url fragment of the topic.\n\n    Returns:\n        Topic or None. The domain object representing a topic with the\n        given id, or None if it does not exist.\n    '
    topic_model: Optional[topic_models.TopicModel] = topic_models.TopicModel.get_by_url_fragment(url_fragment)
    if topic_model is None:
        return None
    return get_topic_from_model(topic_model)

def get_all_topics() -> List[topic_domain.Topic]:
    if False:
        i = 10
        return i + 15
    'Returns all the topics present in the datastore.\n\n    Returns:\n        list(Topic). The list of topics present in the datastore.\n    '
    backend_topic_models = topic_models.TopicModel.get_all()
    topics: List[topic_domain.Topic] = [get_topic_from_model(topic) for topic in backend_topic_models]
    return topics

@overload
def get_topic_rights(topic_id: str) -> topic_domain.TopicRights:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_topic_rights(topic_id: str, *, strict: Literal[True]) -> topic_domain.TopicRights:
    if False:
        while True:
            i = 10
    ...

@overload
def get_topic_rights(topic_id: str, *, strict: Literal[False]) -> Optional[topic_domain.TopicRights]:
    if False:
        i = 10
        return i + 15
    ...

def get_topic_rights(topic_id: str, strict: bool=True) -> Optional[topic_domain.TopicRights]:
    if False:
        for i in range(10):
            print('nop')
    'Retrieves the rights object for the given topic.\n\n    Args:\n        topic_id: str. ID of the topic.\n        strict: bool. Whether to fail noisily if no topic with a given id\n            exists in the datastore.\n\n    Returns:\n        TopicRights or None. The rights object associated with the given topic,\n        or None if it does not exist.\n\n    Raises:\n        EntityNotFoundError. The TopicRights with ID topic_id was not\n            found in the datastore.\n    '
    model: Optional[topic_models.TopicRightsModel] = topic_models.TopicRightsModel.get(topic_id, strict=strict)
    if model is None:
        return None
    return get_topic_rights_from_model(model)

def get_topic_rights_from_model(topic_rights_model: topic_models.TopicRightsModel) -> topic_domain.TopicRights:
    if False:
        return 10
    'Constructs a TopicRights object from the given topic rights model.\n\n    Args:\n        topic_rights_model: TopicRightsModel. Topic rights from the\n            datastore.\n\n    Returns:\n        TopicRights. The rights object created from the model.\n    '
    return topic_domain.TopicRights(topic_rights_model.id, topic_rights_model.manager_ids, topic_rights_model.topic_is_published)

def get_all_topic_summaries() -> List[topic_domain.TopicSummary]:
    if False:
        for i in range(10):
            print('nop')
    'Returns the summaries of all topics present in the datastore.\n\n    Returns:\n        list(TopicSummary). The list of summaries of all topics present in the\n        datastore.\n    '
    topic_summaries_models = topic_models.TopicSummaryModel.get_all()
    topic_summaries: List[topic_domain.TopicSummary] = [get_topic_summary_from_model(summary) for summary in topic_summaries_models]
    return topic_summaries

def get_multi_topic_summaries(topic_ids: List[str]) -> List[Optional[topic_domain.TopicSummary]]:
    if False:
        for i in range(10):
            print('nop')
    'Returns the summaries of all topics whose topic ids are passed in.\n\n    Args:\n        topic_ids: list(str). The IDs of topics for which summaries are to be\n            returned.\n\n    Returns:\n        list(TopicSummary) or None. The list of summaries of all given topics\n        present in the datastore, or None if it does not exist.\n    '
    topic_summaries_models = topic_models.TopicSummaryModel.get_multi(topic_ids)
    topic_summaries = [get_topic_summary_from_model(summary) if summary else None for summary in topic_summaries_models]
    return topic_summaries

def get_published_topic_summaries() -> List[topic_domain.TopicSummary]:
    if False:
        i = 10
        return i + 15
    'Returns the summaries of all published topics present in the datastore.\n\n    Returns:\n        list(TopicSummary). The list of summaries of all published topics\n        present in the datastore.\n    '
    topic_id_to_topic_rights = get_all_topic_rights()
    published_topic_ids = [topic_id for (topic_id, topic_rights) in topic_id_to_topic_rights.items() if topic_rights.topic_is_published]
    topic_summaries_list = [topic_summary for topic_summary in get_multi_topic_summaries(published_topic_ids) if topic_summary is not None]
    return topic_summaries_list

def get_all_skill_ids_assigned_to_some_topic() -> Set[str]:
    if False:
        while True:
            i = 10
    'Returns the ids of all the skills that are linked to some topics.\n\n    Returns:\n        set([str]). The ids of all the skills linked to some topic.\n    '
    skill_ids: Set[str] = set()
    all_topic_models = topic_models.TopicModel.get_all()
    all_topics: List[topic_domain.Topic] = [get_topic_from_model(topic) for topic in all_topic_models]
    for topic in all_topics:
        skill_ids.update(topic.get_all_skill_ids())
    return skill_ids

def get_topic_summary_from_model(topic_summary_model: topic_models.TopicSummaryModel) -> topic_domain.TopicSummary:
    if False:
        for i in range(10):
            print('nop')
    'Returns a domain object for an Oppia topic summary given a\n    topic summary model.\n\n    Args:\n        topic_summary_model: TopicSummaryModel. The topic summary model object\n            to get the corresponding domain object.\n\n    Returns:\n        TopicSummary. The domain object corresponding to the given model object.\n    '
    return topic_domain.TopicSummary(topic_summary_model.id, topic_summary_model.name, topic_summary_model.canonical_name, topic_summary_model.language_code, topic_summary_model.description, topic_summary_model.version, topic_summary_model.canonical_story_count, topic_summary_model.additional_story_count, topic_summary_model.uncategorized_skill_count, topic_summary_model.subtopic_count, topic_summary_model.total_skill_count, topic_summary_model.total_published_node_count, topic_summary_model.thumbnail_filename, topic_summary_model.thumbnail_bg_color, topic_summary_model.url_fragment, topic_summary_model.topic_model_created_on, topic_summary_model.topic_model_last_updated)

@overload
def get_topic_summary_by_id(topic_id: str) -> topic_domain.TopicSummary:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_topic_summary_by_id(topic_id: str, *, strict: Literal[True]) -> topic_domain.TopicSummary:
    if False:
        print('Hello World!')
    ...

@overload
def get_topic_summary_by_id(topic_id: str, *, strict: Literal[False]) -> Optional[topic_domain.TopicSummary]:
    if False:
        return 10
    ...

def get_topic_summary_by_id(topic_id: str, strict: bool=True) -> Optional[topic_domain.TopicSummary]:
    if False:
        while True:
            i = 10
    'Returns a domain object representing a topic summary.\n\n    Args:\n        topic_id: str. ID of the topic summary.\n        strict: bool. Whether to fail noisily if no topic summary with the given\n            id exist in the datastore.\n\n    Returns:\n        TopicSummary or None. The topic summary domain object corresponding to\n        a topic with the given topic_id, if it exists, or else None.\n    '
    topic_summary_model: Optional[topic_models.TopicSummaryModel] = topic_models.TopicSummaryModel.get(topic_id, strict=strict)
    if topic_summary_model:
        topic_summary: topic_domain.TopicSummary = get_topic_summary_from_model(topic_summary_model)
        return topic_summary
    else:
        return None

def get_new_topic_id() -> str:
    if False:
        for i in range(10):
            print('nop')
    'Returns a new topic id.\n\n    Returns:\n        str. A new topic id.\n    '
    return topic_models.TopicModel.get_new_id('')

@overload
def get_multi_topic_rights(topic_ids: List[str], *, strict: Literal[True]) -> List[topic_domain.TopicRights]:
    if False:
        return 10
    ...

@overload
def get_multi_topic_rights(topic_ids: List[str]) -> List[Optional[topic_domain.TopicRights]]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_multi_topic_rights(topic_ids: List[str], *, strict: Literal[False]) -> List[Optional[topic_domain.TopicRights]]:
    if False:
        while True:
            i = 10
    ...

def get_multi_topic_rights(topic_ids: List[str], strict: bool=False) -> Sequence[Optional[topic_domain.TopicRights]]:
    if False:
        return 10
    'Returns the rights of all topics whose topic ids are passed in.\n\n    Args:\n        topic_ids: list(str). The IDs of topics for which rights are to be\n            returned.\n        strict: bool. Whether to fail noisily if no TopicRights exists for\n            the given topic id.\n\n    Returns:\n        Sequence[Optional[TopicRights]]. The list of rights of all given topics\n        present in the datastore.\n\n    Raises:\n        Exception. No topic_rights exists for the given topic_id.\n    '
    topic_rights_models: List[Optional[topic_models.TopicRightsModel]] = topic_models.TopicRightsModel.get_multi(topic_ids)
    topic_rights: List[Optional[topic_domain.TopicRights]] = []
    for (index, rights) in enumerate(topic_rights_models):
        if rights is None:
            if strict:
                raise Exception('No topic_rights exists for the given topic_id: %s' % topic_ids[index])
            topic_rights.append(rights)
        else:
            topic_rights.append(get_topic_rights_from_model(rights))
    return topic_rights

def get_topic_rights_with_user(user_id: str) -> List[topic_domain.TopicRights]:
    if False:
        for i in range(10):
            print('nop')
    'Retrieves the rights object for all topics assigned to given user.\n\n    Args:\n        user_id: str. ID of the user.\n\n    Returns:\n        list(TopicRights). The rights objects associated with the topics\n        assigned to given user.\n    '
    topic_rights_models: Sequence[topic_models.TopicRightsModel] = topic_models.TopicRightsModel.get_by_user(user_id)
    return [get_topic_rights_from_model(model) for model in topic_rights_models if model is not None]

def get_all_topic_rights() -> Dict[str, topic_domain.TopicRights]:
    if False:
        while True:
            i = 10
    'Returns the rights object of all topics present in the datastore.\n\n    Returns:\n        dict. The dict of rights objects of all topics present in the datastore\n        keyed by topic id.\n    '
    topic_rights_models = topic_models.TopicRightsModel.get_all()
    topic_rights: Dict[str, topic_domain.TopicRights] = {}
    for model in topic_rights_models:
        rights: topic_domain.TopicRights = get_topic_rights_from_model(model)
        topic_rights[rights.id] = rights
    return topic_rights

class CannonicalStoryDict(TypedDict):
    """Dictionary that represents cannonical stories."""
    id: str
    title: str
    description: str
    node_titles: List[str]
    thumbnail_bg_color: Optional[str]
    thumbnail_filename: Optional[str]
    url_fragment: str
    topic_url_fragment: str
    classroom_url_fragment: str
    story_is_published: bool
    completed_node_titles: List[str]
    all_node_dicts: List[story_domain.StoryNodeDict]

def get_canonical_story_dicts(user_id: str, topic: topic_domain.Topic) -> List[CannonicalStoryDict]:
    if False:
        while True:
            i = 10
    'Returns a list of canonical story dicts in the topic.\n\n    Args:\n        user_id: str. The ID of the user.\n        topic: Topic. The topic domain object.\n\n    Returns:\n        list(dict). A list of canonical story dicts in the given topic.\n    '
    canonical_story_ids: List[str] = topic.get_canonical_story_ids(include_only_published=True)
    canonical_story_summaries: List[story_domain.StorySummary] = [story_fetchers.get_story_summary_by_id(canonical_story_id) for canonical_story_id in canonical_story_ids]
    canonical_story_dicts = []
    for story_summary in canonical_story_summaries:
        pending_and_all_nodes_in_story = story_fetchers.get_pending_and_all_nodes_in_story(user_id, story_summary.id)
        all_nodes = pending_and_all_nodes_in_story['all_nodes']
        pending_nodes = pending_and_all_nodes_in_story['pending_nodes']
        pending_node_titles = [node.title for node in pending_nodes]
        completed_node_titles = utils.compute_list_difference(story_summary.node_titles, pending_node_titles)
        story_summary_dict: CannonicalStoryDict = story_summary.to_human_readable_dict()
        story_summary_dict['topic_url_fragment'] = topic.url_fragment
        story_summary_dict['classroom_url_fragment'] = classroom_config_services.get_classroom_url_fragment_for_topic_id(topic.id)
        story_summary_dict['story_is_published'] = True
        story_summary_dict['completed_node_titles'] = completed_node_titles
        story_summary_dict['all_node_dicts'] = [node.to_dict() for node in all_nodes]
        canonical_story_dicts.append(story_summary_dict)
    return canonical_story_dicts