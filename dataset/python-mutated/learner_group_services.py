"""Services for the learner groups."""
from __future__ import annotations
from core import platform_feature_list
from core.constants import constants
from core.domain import config_domain
from core.domain import learner_group_domain
from core.domain import learner_group_fetchers
from core.domain import platform_feature_services
from core.domain import story_domain
from core.domain import story_fetchers
from core.domain import subtopic_page_domain
from core.domain import topic_domain
from core.domain import topic_fetchers
from core.platform import models
from typing import List, Optional, Sequence, Tuple
MYPY = False
if MYPY:
    from mypy_imports import datastore_services
    from mypy_imports import learner_group_models
    from mypy_imports import user_models
(learner_group_models, user_models) = models.Registry.import_models([models.Names.LEARNER_GROUP, models.Names.USER])
datastore_services = models.Registry.import_datastore_services()

def is_learner_group_feature_enabled() -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Checks if the learner group feature is enabled.\n\n    Returns:\n        bool. Whether the learner group feature is enabled.\n    '
    return bool(platform_feature_services.is_feature_enabled(platform_feature_list.ParamNames.LEARNER_GROUPS_ARE_ENABLED.value))

def create_learner_group(group_id: str, title: str, description: str, facilitator_user_ids: List[str], invited_learner_ids: List[str], subtopic_page_ids: List[str], story_ids: List[str]) -> learner_group_domain.LearnerGroup:
    if False:
        return 10
    'Creates a new learner group.\n\n    Args:\n        group_id: str. The id of the learner group to be created.\n        title: str. The title of the learner group.\n        description: str. The description of the learner group.\n        facilitator_user_ids: str. List of user ids of the facilitators of the\n            learner group.\n        invited_learner_ids: list(str). List of user ids of the learners who\n            have been invited to join the learner group.\n        subtopic_page_ids: list(str). The ids of the subtopics pages that are\n            part of the learner group syllabus. Each subtopic page id is\n            represented as a topicId:subtopicId string.\n        story_ids: list(str). The ids of the stories that are part of the\n            learner group syllabus.\n\n    Returns:\n        LearnerGroup. The domain object of the newly created learner group.\n    '
    learner_group = learner_group_domain.LearnerGroup(group_id, title, description, facilitator_user_ids, [], invited_learner_ids, subtopic_page_ids, story_ids)
    learner_group.validate()
    learner_group_model = learner_group_models.LearnerGroupModel(id=group_id, title=title, description=description, facilitator_user_ids=facilitator_user_ids, learner_user_ids=[], invited_learner_user_ids=invited_learner_ids, subtopic_page_ids=subtopic_page_ids, story_ids=story_ids)
    learner_group_model.update_timestamps()
    learner_group_model.put()
    if len(learner_group_model.invited_learner_user_ids) > 0:
        invite_learners_to_learner_group(group_id, learner_group_model.invited_learner_user_ids)
    return learner_group

def update_learner_group(group_id: str, title: str, description: str, facilitator_user_ids: List[str], learner_ids: List[str], invited_learner_ids: List[str], subtopic_page_ids: List[str], story_ids: List[str]) -> learner_group_domain.LearnerGroup:
    if False:
        return 10
    'Updates a learner group if it is present.\n\n    Args:\n        group_id: str. The id of the learner group to be updated.\n        title: str. The title of the learner group.\n        description: str. The description of the learner group.\n        facilitator_user_ids: str. List of user ids of the facilitators of the\n            learner group.\n        learner_ids: list(str). List of user ids of the learners of the\n            learner group.\n        invited_learner_ids: list(str). List of user ids of the learners who\n            have been invited to join the learner group.\n        subtopic_page_ids: list(str). The ids of the subtopics pages that are\n            part of the learner group syllabus. Each subtopic page id is\n            represented as a topicId:subtopicId string.\n        story_ids: list(str). The ids of the stories that are part of the\n            learner group syllabus.\n\n    Returns:\n        learner_group: learner_group_domain.LearnerGroup. The domain object\n        of the updated learner group.\n    '
    learner_group_model = learner_group_models.LearnerGroupModel.get(group_id, strict=True)
    old_invited_learner_ids = set(learner_group_model.invited_learner_user_ids)
    new_invited_learner_ids = set(invited_learner_ids)
    if new_invited_learner_ids != old_invited_learner_ids:
        newly_added_invites = list(new_invited_learner_ids - old_invited_learner_ids)
        newly_removed_invites = list(old_invited_learner_ids - new_invited_learner_ids)
        invite_learners_to_learner_group(group_id, newly_added_invites)
        remove_invited_learners_from_learner_group(group_id, newly_removed_invites, False)
    old_learner_ids = set(learner_group_model.learner_user_ids)
    new_learner_ids = set(learner_ids)
    if old_learner_ids != new_learner_ids:
        newly_removed_learners = list(old_learner_ids - new_learner_ids)
        remove_learners_from_learner_group(group_id, newly_removed_learners, False)
    learner_group_model.title = title
    learner_group_model.description = description
    learner_group_model.facilitator_user_ids = facilitator_user_ids
    learner_group_model.learner_user_ids = learner_ids
    learner_group_model.invited_learner_user_ids = invited_learner_ids
    learner_group_model.subtopic_page_ids = subtopic_page_ids
    learner_group_model.story_ids = story_ids
    learner_group = get_learner_group_from_model(learner_group_model)
    learner_group.validate()
    learner_group_model.update_timestamps()
    learner_group_model.put()
    return get_learner_group_from_model(learner_group_model)

def is_user_facilitator(user_id: str, group_id: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Checks if the user is a facilitator of the leaner group.\n\n    Args:\n        user_id: str. The id of the user.\n        group_id: str. The id of the learner group.\n\n    Returns:\n        bool. Whether the user is a facilitator of the learner group.\n    '
    learner_group_model = learner_group_models.LearnerGroupModel.get(group_id, strict=True)
    return user_id in learner_group_model.facilitator_user_ids

def is_user_learner(user_id: str, group_id: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Checks if the user is a learner of the learner group.\n\n    Args:\n        user_id: str. The id of the user.\n        group_id: str. The id of the learner group.\n\n    Returns:\n        bool. Whether the user is a learner of the learner group.\n    '
    learner_group_model = learner_group_models.LearnerGroupModel.get(group_id, strict=True)
    return user_id in learner_group_model.learner_user_ids

def remove_learner_group(group_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Removes the learner group with of given learner group ID.\n\n    Args:\n        group_id: str. The id of the learner group to be removed.\n    '
    learner_group_model = learner_group_models.LearnerGroupModel.get(group_id, strict=True)
    learner_group_model.delete()

def get_matching_learner_group_syllabus_to_add(learner_group_id: str, keyword: str, search_type: str, category: str, language_code: str) -> learner_group_domain.LearnerGroupSyllabusDict:
    if False:
        print('Hello World!')
    "Returns the syllabus of items matching the given filter arguments\n    that can be added to the learner group.\n\n    Args:\n        learner_group_id: str. The id of the learner group.\n        keyword: str. The keyword to search the syllabus. It is compared with\n            the title of the topics, stories and subtopics.\n        search_type: str. The type of the syllabus item to search. It can be\n            either 'Story' or 'Skill'.\n        category: str. The category of the syllabus items. It is the\n            classroom in which the stories and subtopics are to be searched.\n        language_code: str. The language of the topics in which the stories\n            and subtopics are to be searched.\n\n    Returns:\n        dict. The matching syllabus items to add to the learner group.\n    "
    group_subtopic_page_ids: List[str] = []
    group_story_ids: List[str] = []
    if learner_group_id:
        learner_group_model = learner_group_models.LearnerGroupModel.get(learner_group_id, strict=True)
        group_subtopic_page_ids = learner_group_model.subtopic_page_ids
        group_story_ids = learner_group_model.story_ids
    matching_topic_ids: List[str] = []
    all_classrooms_dict = config_domain.CLASSROOM_PAGES_DATA.value
    matching_subtopics_dicts: List[subtopic_page_domain.SubtopicPageSummaryDict] = []
    matching_story_syllabus_item_dicts: List[story_domain.LearnerGroupSyllabusStorySummaryDict] = []
    if category != constants.DEFAULT_ADD_SYLLABUS_FILTER:
        for classroom in all_classrooms_dict:
            if category and classroom['name'] == category:
                matching_topic_ids.extend(classroom['topic_ids'])
        matching_topics: List[topic_domain.Topic] = topic_fetchers.get_topics_by_ids(matching_topic_ids, strict=True)
    else:
        matching_topics = topic_fetchers.get_all_topics()
    keyword = keyword.lower()
    for topic in matching_topics:
        if language_code not in (constants.DEFAULT_ADD_SYLLABUS_FILTER, topic.language_code):
            continue
        if keyword in topic.canonical_name:
            if search_type in (constants.LEARNER_GROUP_ADD_STORY_FILTER, constants.DEFAULT_ADD_SYLLABUS_FILTER):
                matching_story_syllabus_item_dicts.extend(get_matching_story_syllabus_item_dicts(topic, group_story_ids))
            if search_type in (constants.LEARNER_GROUP_ADD_SKILL_FILTER, constants.DEFAULT_ADD_SYLLABUS_FILTER):
                matching_subtopics_dicts.extend(get_matching_subtopic_syllabus_item_dicts(topic, group_subtopic_page_ids))
        else:
            if search_type in (constants.LEARNER_GROUP_ADD_SKILL_FILTER, constants.DEFAULT_ADD_SYLLABUS_FILTER):
                matching_subtopics_dicts.extend(get_matching_subtopic_syllabus_item_dicts(topic, group_subtopic_page_ids, keyword))
            if search_type in (constants.LEARNER_GROUP_ADD_STORY_FILTER, constants.DEFAULT_ADD_SYLLABUS_FILTER):
                matching_story_syllabus_item_dicts.extend(get_matching_story_syllabus_item_dicts(topic, group_story_ids, keyword))
    return {'story_summary_dicts': matching_story_syllabus_item_dicts, 'subtopic_summary_dicts': matching_subtopics_dicts}

def get_matching_subtopic_syllabus_item_dicts(topic: topic_domain.Topic, group_subtopic_page_ids: List[str], keyword: Optional[str]=None) -> List[subtopic_page_domain.SubtopicPageSummaryDict]:
    if False:
        i = 10
        return i + 15
    'Returns the matching subtopics syllabus item dicts of the given topic\n    that can be added to the learner group syllabus.\n\n    Args:\n        topic: Topic. The topic whose subtopic subtopic items are to be\n            searched.\n        group_subtopic_page_ids: list(str). The ids of the subtopic pages of\n            the learner group.\n        keyword: Optional[str]. The keyword to search the subtopic syllabus\n            items. It is compared with the title of the subtopics if passed\n            in arguments.\n\n    Returns:\n        list(dict). The matching subtopic syllabus items of the given topic.\n    '
    matching_subtopic_syllabus_item_dicts: List[subtopic_page_domain.SubtopicPageSummaryDict] = []
    for subtopic in topic.subtopics:
        subtopic_page_id = '{}:{}'.format(topic.id, subtopic.id)
        if subtopic_page_id not in group_subtopic_page_ids:
            if keyword is None or keyword in subtopic.title.lower():
                matching_subtopic_syllabus_item_dicts.append({'subtopic_id': subtopic.id, 'subtopic_title': subtopic.title, 'parent_topic_id': topic.id, 'parent_topic_name': topic.name, 'thumbnail_filename': subtopic.thumbnail_filename, 'thumbnail_bg_color': subtopic.thumbnail_bg_color, 'subtopic_mastery': None, 'parent_topic_url_fragment': topic.url_fragment, 'classroom_url_fragment': None})
    return matching_subtopic_syllabus_item_dicts

def get_matching_story_syllabus_item_dicts(topic: topic_domain.Topic, group_story_ids: List[str], keyword: Optional[str]=None) -> List[story_domain.LearnerGroupSyllabusStorySummaryDict]:
    if False:
        return 10
    'Returns the matching story syllabus item dicts of the given topic\n    that can be added to the learner group syllabus.\n\n    Args:\n        topic: Topic. The topic whose stories are to be searched.\n        group_story_ids: list(str). The story ids of the learner group.\n        keyword: Optional[str]. The keyword to search the stories. It is\n            compared with the title of the story if passed in arguments.\n\n    Returns:\n        list(dict). The matching story syllabus item dicts of the given topic.\n    '
    story_ids = [story.story_id for story in topic.canonical_story_references if story.story_id not in group_story_ids and story.story_is_published is True]
    matching_stories = story_fetchers.get_story_summaries_by_ids(story_ids)
    stories = story_fetchers.get_stories_by_ids(story_ids, strict=True)
    matching_story_syllabus_item_dicts: List[story_domain.LearnerGroupSyllabusStorySummaryDict] = []
    for (ind, story_summary) in enumerate(matching_stories):
        if keyword is None or keyword in story_summary.title.lower():
            story = stories[ind]
            summary_dict = story_summary.to_dict()
            matching_story_syllabus_item_dicts.append({'id': summary_dict['id'], 'title': summary_dict['title'], 'description': summary_dict['description'], 'language_code': summary_dict['language_code'], 'version': summary_dict['version'], 'node_titles': summary_dict['node_titles'], 'thumbnail_filename': summary_dict['thumbnail_filename'], 'thumbnail_bg_color': summary_dict['thumbnail_bg_color'], 'url_fragment': summary_dict['url_fragment'], 'story_model_created_on': summary_dict['story_model_created_on'], 'story_model_last_updated': summary_dict['story_model_last_updated'], 'story_is_published': True, 'completed_node_titles': [], 'all_node_dicts': [node.to_dict() for node in story.story_contents.nodes], 'topic_name': topic.name, 'topic_url_fragment': topic.url_fragment, 'classroom_url_fragment': None})
    return matching_story_syllabus_item_dicts

def add_learner_to_learner_group(group_id: str, user_id: str, progress_sharing_permission: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Adds the given learner to the given learner group.\n\n    Args:\n        group_id: str. The id of the learner group.\n        user_id: str. The id of the learner.\n        progress_sharing_permission: bool. The progress sharing permission of\n            the learner group. True if progress sharing is allowed, False\n            otherwise.\n\n    Raises:\n        Exception. Learner was not invited to join the learner group.\n    '
    learner_group_model = learner_group_models.LearnerGroupModel.get(group_id, strict=True)
    if user_id not in learner_group_model.invited_learner_user_ids:
        raise Exception('Learner was not invited to join the learner group.')
    learner_group_model.invited_learner_user_ids.remove(user_id)
    learner_group_model.learner_user_ids.append(user_id)
    details_of_learner_group = {'group_id': group_id, 'progress_sharing_is_turned_on': progress_sharing_permission}
    learner_grps_user_model = user_models.LearnerGroupsUserModel.get(user_id, strict=True)
    learner_grps_user_model.invited_to_learner_groups_ids.remove(group_id)
    learner_grps_user_model.learner_groups_user_details.append(details_of_learner_group)
    learner_grps_user_model.update_timestamps()
    learner_grps_user_model.put()
    learner_group_model.update_timestamps()
    learner_group_model.put()

def remove_learners_from_learner_group(group_id: str, user_ids: List[str], update_group: bool) -> None:
    if False:
        return 10
    'Removes the given learner from the given learner group.\n\n    Args:\n        group_id: str. The id of the learner group.\n        user_ids: List[str]. The id of the learners to be removed.\n        update_group: bool. Flag indicating whether to update the\n            learner group or not.\n    '
    if update_group:
        learner_group_model = learner_group_models.LearnerGroupModel.get(group_id, strict=True)
        learner_group_model.learner_user_ids = [user_id for user_id in learner_group_model.learner_user_ids if user_id not in user_ids]
        learner_group_model.update_timestamps()
        learner_group_model.put()
    learner_grps_users_models = learner_group_fetchers.get_learner_group_models_by_ids(user_ids, strict=True)
    models_to_put = []
    for learner_grps_user_model in learner_grps_users_models:
        learner_grps_user_model.learner_groups_user_details = [details for details in learner_grps_user_model.learner_groups_user_details if details['group_id'] != group_id]
        models_to_put.append(learner_grps_user_model)
    user_models.LearnerGroupsUserModel.update_timestamps_multi(models_to_put)
    user_models.LearnerGroupsUserModel.put_multi(models_to_put)

def invite_learners_to_learner_group(group_id: str, invited_learner_ids: List[str]) -> None:
    if False:
        while True:
            i = 10
    'Invites the given learners to the given learner group.\n\n    Args:\n        group_id: str. The id of the learner group.\n        invited_learner_ids: list(str). The ids of the learners to invite.\n    '
    learner_groups_user_models = user_models.LearnerGroupsUserModel.get_multi(invited_learner_ids)
    models_to_put = []
    for (index, learner_id) in enumerate(invited_learner_ids):
        learner_groups_user_model = learner_groups_user_models[index]
        if learner_groups_user_model:
            learner_groups_user_model.invited_to_learner_groups_ids.append(group_id)
        else:
            learner_groups_user_model = user_models.LearnerGroupsUserModel(id=learner_id, invited_to_learner_groups_ids=[group_id], learner_groups_user_details=[])
        models_to_put.append(learner_groups_user_model)
    user_models.LearnerGroupsUserModel.update_timestamps_multi(models_to_put)
    user_models.LearnerGroupsUserModel.put_multi(models_to_put)

def remove_invited_learners_from_learner_group(group_id: str, learner_ids: List[str], update_group: bool) -> None:
    if False:
        i = 10
        return i + 15
    'Removes the given invited learners from the given learner group.\n\n    Args:\n        group_id: str. The id of the learner group.\n        learner_ids: list(str). The ids of the learners to remove.\n        update_group: bool. Flag indicating whether to update the\n            learner group or not.\n    '
    if update_group:
        learner_group_model = learner_group_models.LearnerGroupModel.get(group_id, strict=True)
        learner_group_model.invited_learner_user_ids = [learner_id for learner_id in learner_group_model.invited_learner_user_ids if learner_id not in learner_ids]
        learner_group_model.update_timestamps()
        learner_group_model.put()
    found_models = learner_group_fetchers.get_learner_group_models_by_ids(learner_ids, strict=True)
    models_to_put = []
    for model in found_models:
        if group_id in model.invited_to_learner_groups_ids:
            model.invited_to_learner_groups_ids.remove(group_id)
            models_to_put.append(model)
    user_models.LearnerGroupsUserModel.update_timestamps_multi(models_to_put)
    user_models.LearnerGroupsUserModel.put_multi(models_to_put)

def get_learner_group_from_model(learner_group_model: learner_group_models.LearnerGroupModel) -> learner_group_domain.LearnerGroup:
    if False:
        print('Hello World!')
    'Returns the learner group domain object given the learner group\n    model loaded from the datastore.\n\n    Args:\n        learner_group_model: LearnerGroupModel. The learner group model\n            from the datastore.\n\n    Returns:\n        LearnerGroup. The learner group domain object corresponding to the\n        given model.\n    '
    return learner_group_domain.LearnerGroup(learner_group_model.id, learner_group_model.title, learner_group_model.description, learner_group_model.facilitator_user_ids, learner_group_model.learner_user_ids, learner_group_model.invited_learner_user_ids, learner_group_model.subtopic_page_ids, learner_group_model.story_ids)

def can_user_be_invited(user_id: str, username: str, group_id: str) -> Tuple[bool, str]:
    if False:
        print('Hello World!')
    'Checks if the user can be invited to the learner group.\n\n    Args:\n        user_id: str. The id of the user.\n        username: str. The username of the user.\n        group_id: str. The id of the learner group.\n\n    Returns:\n        bool. True if the user can be invited to the learner group. False\n        otherwise.\n        str. Error message if the user cannot be invited to the learner group.\n    '
    if not group_id:
        return (True, '')
    learner_group_model = learner_group_models.LearnerGroupModel.get(group_id, strict=True)
    if user_id in learner_group_model.learner_user_ids:
        return (False, 'User with username %s is already a learner.' % username)
    elif user_id in learner_group_model.invited_learner_user_ids:
        return (False, 'User with username %s has been already invited to join the group' % username)
    elif user_id in learner_group_model.facilitator_user_ids:
        return (False, 'User with username %s is already a facilitator.' % username)
    return (True, '')

def remove_story_reference_from_learner_groups(story_id: str) -> None:
    if False:
        i = 10
        return i + 15
    "Removes a given story id from all learner groups that have it's\n    reference.\n\n    Args:\n        story_id: str. Story id to remove.\n    "
    found_models: Sequence[learner_group_models.LearnerGroupModel] = learner_group_models.LearnerGroupModel.get_all().filter(datastore_services.any_of(learner_group_models.LearnerGroupModel.story_ids == story_id)).fetch()
    models_to_put = []
    for model in found_models:
        model.story_ids.remove(story_id)
        models_to_put.append(model)
    learner_group_models.LearnerGroupModel.update_timestamps_multi(models_to_put)
    learner_group_models.LearnerGroupModel.put_multi(models_to_put)

def remove_subtopic_page_reference_from_learner_groups(topic_id: str, subtopic_id: int) -> None:
    if False:
        return 10
    "Removes a given subtopic page from all learner groups that have it's\n    reference.\n\n    Args:\n        topic_id: str. Id of the topic of the subtopic page.\n        subtopic_id: int. Id of the subtopic of the subtopic page.\n    "
    subtopic_page_id = '{}:{}'.format(topic_id, subtopic_id)
    learner_group_model_cls = learner_group_models.LearnerGroupModel
    found_models: Sequence[learner_group_models.LearnerGroupModel] = learner_group_model_cls.get_all().filter(datastore_services.any_of(learner_group_model_cls.subtopic_page_ids == subtopic_page_id)).fetch()
    models_to_put = []
    for model in found_models:
        model.subtopic_page_ids.remove(subtopic_page_id)
        models_to_put.append(model)
    learner_group_models.LearnerGroupModel.update_timestamps_multi(models_to_put)
    learner_group_models.LearnerGroupModel.put_multi(models_to_put)

def update_progress_sharing_permission(user_id: str, group_id: str, new_progress_sharing_permission: bool) -> None:
    if False:
        i = 10
        return i + 15
    'Updates the progress sharing permission of the learner group.\n\n    Args:\n        user_id: str. The id of the user.\n        group_id: str. The id of the learner group.\n        new_progress_sharing_permission: bool. The new progress sharing\n            permission of the learner group.\n    '
    learner_grps_user_model = user_models.LearnerGroupsUserModel.get(user_id, strict=True)
    old_user_details = learner_grps_user_model.learner_groups_user_details
    learner_grps_user_model.learner_groups_user_details = []
    for group_details in old_user_details:
        if group_details['group_id'] == group_id:
            learner_grps_user_model.learner_groups_user_details.append({'group_id': group_id, 'progress_sharing_is_turned_on': new_progress_sharing_permission})
        else:
            learner_grps_user_model.learner_groups_user_details.append(group_details)
    learner_grps_user_model.update_timestamps()
    learner_grps_user_model.put()