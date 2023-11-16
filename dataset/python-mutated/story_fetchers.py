"""Commands that can be used to fetch story related models.

All functions here should be agnostic of how StoryModel objects are
stored in the database. In particular, the various query methods should
delegate to the Story model class. This will enable the story
storage model to be changed without affecting this module and others above it.
"""
from __future__ import annotations
import copy
import itertools
from core import feconf
from core.domain import caching_services
from core.domain import classroom_config_services
from core.domain import exp_fetchers
from core.domain import story_domain
from core.domain import topic_fetchers
from core.domain import user_services
from core.platform import models
from typing import Dict, List, Literal, Optional, Sequence, overload
MYPY = False
if MYPY:
    from mypy_imports import story_models
    from mypy_imports import user_models
(story_models, user_models) = models.Registry.import_models([models.Names.STORY, models.Names.USER])

def _migrate_story_contents_to_latest_schema(versioned_story_contents: story_domain.VersionedStoryContentsDict, story_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Holds the responsibility of performing a step-by-step, sequential update\n    of the story structure based on the schema version of the input\n    story dictionary. If the current story_contents schema changes, a new\n    conversion function must be added and some code appended to this function\n    to account for that new version.\n\n    Args:\n        versioned_story_contents: dict. A dict with two keys:\n          - schema_version: str. The schema version for the story_contents dict.\n          - story_contents: dict. The dict comprising the story\n              contents.\n        story_id: str. The unique ID of the story.\n\n    Raises:\n        Exception. The schema version of the story_contents is outside of what\n            is supported at present.\n    '
    story_contents_schema_version = versioned_story_contents['schema_version']
    if not 1 <= story_contents_schema_version <= feconf.CURRENT_STORY_CONTENTS_SCHEMA_VERSION:
        raise Exception('Sorry, we can only process v1-v%d story schemas at present.' % feconf.CURRENT_STORY_CONTENTS_SCHEMA_VERSION)
    while story_contents_schema_version < feconf.CURRENT_STORY_CONTENTS_SCHEMA_VERSION:
        story_domain.Story.update_story_contents_from_model(versioned_story_contents, story_contents_schema_version, story_id)
        story_contents_schema_version += 1

def get_story_from_model(story_model: story_models.StoryModel) -> story_domain.Story:
    if False:
        print('Hello World!')
    'Returns a story domain object given a story model loaded\n    from the datastore.\n\n    Args:\n        story_model: StoryModel. The story model loaded from the\n            datastore.\n\n    Returns:\n        story. A Story domain object corresponding to the given\n        story model.\n    '
    versioned_story_contents: story_domain.VersionedStoryContentsDict = {'schema_version': story_model.story_contents_schema_version, 'story_contents': copy.deepcopy(story_model.story_contents)}
    if story_model.story_contents_schema_version != feconf.CURRENT_STORY_CONTENTS_SCHEMA_VERSION:
        _migrate_story_contents_to_latest_schema(versioned_story_contents, story_model.id)
    return story_domain.Story(story_model.id, story_model.title, story_model.thumbnail_filename, story_model.thumbnail_bg_color, story_model.thumbnail_size_in_bytes, story_model.description, story_model.notes, story_domain.StoryContents.from_dict(versioned_story_contents['story_contents']), versioned_story_contents['schema_version'], story_model.language_code, story_model.corresponding_topic_id, story_model.version, story_model.url_fragment, story_model.meta_tag_content, story_model.created_on, story_model.last_updated)

def get_story_summary_from_model(story_summary_model: story_models.StorySummaryModel) -> story_domain.StorySummary:
    if False:
        return 10
    'Returns a domain object for an Oppia story summary given a\n    story summary model.\n\n    Args:\n        story_summary_model: StorySummaryModel. The story summary model object\n            to get the corresponding domain object.\n\n    Returns:\n        StorySummary. The corresponding domain object to the given story\n        summary model object.\n    '
    return story_domain.StorySummary(story_summary_model.id, story_summary_model.title, story_summary_model.description, story_summary_model.language_code, story_summary_model.version, story_summary_model.node_titles, story_summary_model.thumbnail_bg_color, story_summary_model.thumbnail_filename, story_summary_model.url_fragment, story_summary_model.story_model_created_on, story_summary_model.story_model_last_updated)

@overload
def get_story_by_id(story_id: str) -> story_domain.Story:
    if False:
        while True:
            i = 10
    ...

@overload
def get_story_by_id(story_id: str, *, version: Optional[int]=None) -> story_domain.Story:
    if False:
        print('Hello World!')
    ...

@overload
def get_story_by_id(story_id: str, *, strict: Literal[True], version: Optional[int]=None) -> story_domain.Story:
    if False:
        return 10
    ...

@overload
def get_story_by_id(story_id: str, *, strict: Literal[False], version: Optional[int]=None) -> Optional[story_domain.Story]:
    if False:
        for i in range(10):
            print('nop')
    ...

def get_story_by_id(story_id: str, strict: bool=True, version: Optional[int]=None) -> Optional[story_domain.Story]:
    if False:
        print('Hello World!')
    'Returns a domain object representing a story.\n\n    Args:\n        story_id: str. ID of the story.\n        strict: bool. Whether to fail noisily if no story with the given\n            id exists in the datastore.\n        version: str or None. The version number of the story to be\n            retrieved. If it is None, the latest version will be retrieved.\n\n    Returns:\n        Story or None. The domain object representing a story with the\n        given id, or None if it does not exist.\n    '
    sub_namespace = str(version) if version else None
    cached_story = caching_services.get_multi(caching_services.CACHE_NAMESPACE_STORY, sub_namespace, [story_id]).get(story_id)
    if cached_story is not None:
        return cached_story
    else:
        story_model = story_models.StoryModel.get(story_id, strict=strict, version=version)
        if story_model:
            story = get_story_from_model(story_model)
            caching_services.set_multi(caching_services.CACHE_NAMESPACE_STORY, sub_namespace, {story_id: story})
            return story
        else:
            return None

def get_story_by_url_fragment(url_fragment: str) -> Optional[story_domain.Story]:
    if False:
        while True:
            i = 10
    'Returns a domain object representing a story.\n\n    Args:\n        url_fragment: str. The url fragment of the story.\n\n    Returns:\n        Story or None. The domain object representing a story with the\n        given url_fragment, or None if it does not exist.\n    '
    story_model = story_models.StoryModel.get_by_url_fragment(url_fragment)
    if story_model is None:
        return None
    story = get_story_from_model(story_model)
    return story

@overload
def get_story_summary_by_id(story_id: str) -> story_domain.StorySummary:
    if False:
        print('Hello World!')
    ...

@overload
def get_story_summary_by_id(story_id: str, *, strict: Literal[True]) -> story_domain.StorySummary:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_story_summary_by_id(story_id: str, *, strict: Literal[False]) -> Optional[story_domain.StorySummary]:
    if False:
        print('Hello World!')
    ...

def get_story_summary_by_id(story_id: str, strict: bool=True) -> Optional[story_domain.StorySummary]:
    if False:
        return 10
    'Returns a domain object representing a story summary.\n\n    Args:\n        story_id: str. ID of the story summary.\n        strict: bool. Whether to fail noisily if no story summary with the given\n            id exists in the datastore.\n\n    Returns:\n        StorySummary. The story summary domain object corresponding to\n        a story with the given story_id.\n    '
    story_summary_model = story_models.StorySummaryModel.get(story_id, strict=strict)
    if story_summary_model:
        story_summary = get_story_summary_from_model(story_summary_model)
        return story_summary
    else:
        return None

@overload
def get_stories_by_ids(story_ids: List[str], *, strict: Literal[True]) -> List[story_domain.Story]:
    if False:
        return 10
    ...

@overload
def get_stories_by_ids(story_ids: List[str]) -> List[Optional[story_domain.Story]]:
    if False:
        print('Hello World!')
    ...

@overload
def get_stories_by_ids(story_ids: List[str], *, strict: Literal[False]) -> List[Optional[story_domain.Story]]:
    if False:
        return 10
    ...

def get_stories_by_ids(story_ids: List[str], strict: bool=False) -> Sequence[Optional[story_domain.Story]]:
    if False:
        return 10
    'Returns a list of stories matching the IDs provided.\n\n    Args:\n        story_ids: list(str). List of IDs to get stories for.\n        strict: bool. Whether to fail noisily if no story model exists\n            with a given ID exists in the datastore.\n\n    Returns:\n        list(Story|None). The list of stories corresponding to given ids.  If a\n        Story does not exist, the corresponding returned list element is None.\n\n    Raises:\n        Exception. No story model exists for the given story_id.\n    '
    all_story_models = story_models.StoryModel.get_multi(story_ids)
    stories: List[Optional[story_domain.Story]] = []
    for (index, story_model) in enumerate(all_story_models):
        if story_model is None:
            if strict:
                raise Exception('No story model exists for the story_id: %s' % story_ids[index])
            stories.append(story_model)
        elif story_model is not None:
            stories.append(get_story_from_model(story_model))
    return stories

def get_story_summaries_by_ids(story_ids: List[str]) -> List[story_domain.StorySummary]:
    if False:
        while True:
            i = 10
    'Returns the StorySummary objects corresponding the given story ids.\n\n    Args:\n        story_ids: list(str). The list of story ids for which the story\n            summaries are to be found.\n\n    Returns:\n        list(StorySummary). The story summaries corresponds to given story\n        ids.\n    '
    story_summary_models = story_models.StorySummaryModel.get_multi(story_ids)
    story_summaries = [get_story_summary_from_model(story_summary_model) for story_summary_model in story_summary_models if story_summary_model is not None]
    return story_summaries

def get_learner_group_syllabus_story_summaries(story_ids: List[str]) -> List[story_domain.LearnerGroupSyllabusStorySummaryDict]:
    if False:
        while True:
            i = 10
    'Returns the learner group syllabus story summary dicts\n    corresponding the given story ids.\n\n    Args:\n        story_ids: list(str). The list of story ids for which the story\n            summaries are to be returned.\n\n    Returns:\n        list(LearnerGroupSyllabusStorySummaryDict). The story summaries\n        corresponds to given story ids.\n    '
    all_stories = [story for story in get_stories_by_ids(story_ids) if story]
    topic_ids = list({story.corresponding_topic_id for story in all_stories})
    topics = topic_fetchers.get_topics_by_ids(topic_ids)
    topic_id_to_topic_map = {}
    for topic in topics:
        assert topic is not None
        topic_id_to_topic_map[topic.id] = topic
    story_summaries_dicts = [story_summary.to_dict() for story_summary in get_story_summaries_by_ids(story_ids)]
    return [{'id': story.id, 'title': story.title, 'description': story.description, 'language_code': story.language_code, 'version': story.version, 'node_titles': summary_dict['node_titles'], 'thumbnail_filename': story.thumbnail_filename, 'thumbnail_bg_color': story.thumbnail_bg_color, 'url_fragment': story.url_fragment, 'story_model_created_on': summary_dict['story_model_created_on'], 'story_model_last_updated': summary_dict['story_model_last_updated'], 'story_is_published': True, 'completed_node_titles': [], 'all_node_dicts': [node.to_dict() for node in story.story_contents.nodes], 'topic_name': topic_id_to_topic_map[story.corresponding_topic_id].name, 'topic_url_fragment': topic_id_to_topic_map[story.corresponding_topic_id].url_fragment, 'classroom_url_fragment': None} for (story, summary_dict) in zip(all_stories, story_summaries_dicts)]

def get_latest_completed_node_ids(user_id: str, story_id: str) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Returns the ids of the completed nodes that come latest in the story.\n\n    Args:\n        user_id: str. ID of the given user.\n        story_id: str. ID of the story.\n\n    Returns:\n        list(str). List of the completed node ids that come latest in the story.\n        If length is larger than 3, return the last three of them. If length is\n        smaller or equal to 3, return all of them.\n    '
    progress_model = user_models.StoryProgressModel.get(user_id, story_id, strict=False)
    if not progress_model:
        return []
    num_of_nodes = min(len(progress_model.completed_node_ids), 3)
    story = get_story_by_id(story_id, strict=True)
    ordered_node_ids = [node.id for node in story.story_contents.get_ordered_nodes()]
    ordered_completed_node_ids = [node_id for node_id in ordered_node_ids if node_id in progress_model.completed_node_ids]
    return ordered_completed_node_ids[-num_of_nodes:]

def get_completed_nodes_in_story(user_id: str, story_id: str) -> List[story_domain.StoryNode]:
    if False:
        print('Hello World!')
    'Returns nodes that are completed in a story\n\n    Args:\n        user_id: str. The user id of the user.\n        story_id: str. The id of the story.\n\n    Returns:\n        list(StoryNode). The list of the story nodes that the user has\n        completed.\n    '
    story = get_story_by_id(story_id, strict=True)
    completed_nodes = []
    completed_node_ids = get_completed_node_ids(user_id, story_id)
    for node in story.story_contents.nodes:
        if node.id in completed_node_ids:
            completed_nodes.append(node)
    return completed_nodes

def get_user_progress_in_story_chapters(user_id: str, story_ids: List[str]) -> List[story_domain.StoryChapterProgressSummaryDict]:
    if False:
        i = 10
        return i + 15
    'Returns the progress of multiple users in multiple chapters.\n\n    Args:\n        user_id: str. The user id of the user.\n        story_ids: list(str). The ids of the stories.\n\n    Returns:\n        list(StoryChapterProgressSummaryDict). The list of the progress\n        summaries of the user corresponding to all stories chapters.\n    '
    all_valid_story_nodes: List[story_domain.StoryNode] = []
    for story in get_stories_by_ids(story_ids):
        if story is not None:
            all_valid_story_nodes.extend(story.story_contents.nodes)
    exp_ids = [node.exploration_id for node in all_valid_story_nodes if node.exploration_id]
    exp_id_to_exp_map = exp_fetchers.get_multiple_explorations_by_id(exp_ids)
    user_id_exp_id_combinations = list(itertools.product([user_id], exp_ids))
    exp_user_data_models = user_models.ExplorationUserDataModel.get_multi(user_id_exp_id_combinations)
    all_chapters_progress: List[story_domain.StoryChapterProgressSummaryDict] = []
    for (i, user_id_exp_id_pair) in enumerate(user_id_exp_id_combinations):
        exp_id = user_id_exp_id_pair[1]
        exploration = exp_id_to_exp_map[exp_id]
        all_checkpoints = user_services.get_checkpoints_in_order(exploration.init_state_name, exploration.states)
        model = exp_user_data_models[i]
        visited_checkpoints = 0
        if model is not None:
            most_recently_visited_checkpoint = model.most_recently_reached_checkpoint_state_name
            if most_recently_visited_checkpoint is not None:
                visited_checkpoints = all_checkpoints.index(most_recently_visited_checkpoint) + 1
        all_chapters_progress.append({'exploration_id': exp_id, 'visited_checkpoints_count': visited_checkpoints, 'total_checkpoints_count': len(all_checkpoints)})
    return all_chapters_progress

def get_multi_users_progress_in_stories(user_ids: List[str], story_ids: List[str]) -> Dict[str, List[story_domain.LearnerGroupSyllabusStorySummaryDict]]:
    if False:
        while True:
            i = 10
    'Returns the progress of given users in all given stories.\n\n    Args:\n        user_ids: list(str). The user ids of the users.\n        story_ids: list(str). The list of story ids.\n\n    Returns:\n        Dict(str, list(StoryProgressDict)). Dictionary of user id and their\n        corresponding list of story progress dicts.\n    '
    all_valid_stories = [story for story in get_stories_by_ids(story_ids) if story]
    topic_ids = list({story.corresponding_topic_id for story in all_valid_stories})
    topics = topic_fetchers.get_topics_by_ids(topic_ids, strict=True)
    topic_id_to_topic_map = {}
    for topic in topics:
        topic_id_to_topic_map[topic.id] = topic
    story_id_to_story_map = {story.id: story for story in all_valid_stories}
    valid_story_ids = [story.id for story in all_valid_stories]
    all_story_summaries = get_story_summaries_by_ids(valid_story_ids)
    story_id_to_summary_map = {summary.id: summary for summary in all_story_summaries}
    all_posssible_combinations = itertools.product(user_ids, valid_story_ids)
    progress_models = user_models.StoryProgressModel.get_multi(user_ids, valid_story_ids)
    all_users_stories_progress: Dict[str, List[story_domain.LearnerGroupSyllabusStorySummaryDict]] = {user_id: [] for user_id in user_ids}
    for (i, (user_id, story_id)) in enumerate(all_posssible_combinations):
        progress_model = progress_models[i]
        completed_node_ids = []
        if progress_model is not None:
            completed_node_ids = progress_model.completed_node_ids
        story = story_id_to_story_map[story_id]
        completed_node_titles = [node.title for node in story.story_contents.nodes if node.id in completed_node_ids]
        topic = topic_id_to_topic_map[story.corresponding_topic_id]
        summary_dict = story_id_to_summary_map[story_id].to_dict()
        all_users_stories_progress[user_id].append({'id': summary_dict['id'], 'title': summary_dict['title'], 'description': summary_dict['description'], 'language_code': summary_dict['language_code'], 'version': summary_dict['version'], 'node_titles': summary_dict['node_titles'], 'thumbnail_filename': summary_dict['thumbnail_filename'], 'thumbnail_bg_color': summary_dict['thumbnail_bg_color'], 'url_fragment': summary_dict['url_fragment'], 'story_model_created_on': summary_dict['story_model_created_on'], 'story_model_last_updated': summary_dict['story_model_last_updated'], 'story_is_published': True, 'completed_node_titles': completed_node_titles, 'all_node_dicts': [node.to_dict() for node in story.story_contents.nodes], 'topic_name': topic.name, 'topic_url_fragment': topic.url_fragment, 'classroom_url_fragment': classroom_config_services.get_classroom_url_fragment_for_topic_id(topic.id)})
    return all_users_stories_progress

def get_pending_and_all_nodes_in_story(user_id: Optional[str], story_id: str) -> Dict[str, List[story_domain.StoryNode]]:
    if False:
        print('Hello World!')
    'Returns the nodes that are pending in a story\n\n    Args:\n        user_id: Optional[str]. The user id of the user, or None if\n            the user is not logged in.\n        story_id: str. The id of the story.\n\n    Returns:\n        Dict[str, List[story_domain.StoryNode]]. The list of story nodes,\n        pending for the user.\n    '
    story = get_story_by_id(story_id, strict=True)
    pending_nodes = []
    completed_node_ids = get_completed_node_ids(user_id, story_id) if user_id else []
    for node in story.story_contents.nodes:
        if node.id not in completed_node_ids:
            pending_nodes.append(node)
    return {'all_nodes': story.story_contents.nodes, 'pending_nodes': pending_nodes}

def get_completed_node_ids(user_id: str, story_id: str) -> List[str]:
    if False:
        return 10
    'Returns the ids of the nodes completed in the story.\n\n    Args:\n        user_id: str. ID of the given user.\n        story_id: str. ID of the story.\n\n    Returns:\n        list(str). List of the node ids completed in story.\n    '
    progress_model = user_models.StoryProgressModel.get(user_id, story_id, strict=False)
    if progress_model:
        completed_node_ids: List[str] = progress_model.completed_node_ids
        return completed_node_ids
    else:
        return []

def get_node_index_by_story_id_and_node_id(story_id: str, node_id: str) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Returns the index of the story node with the given story id\n    and node id.\n\n    Args:\n        story_id: str. ID of the story.\n        node_id: str. ID of the story node.\n\n    Returns:\n        int. The index of the corresponding node.\n\n    Raises:\n        Exception. The given story does not exist.\n    '
    story = get_story_by_id(story_id, strict=False)
    if story is None:
        raise Exception('Story with id %s does not exist.' % story_id)
    node_index = story.story_contents.get_node_index(node_id)
    return node_index