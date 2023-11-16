"""Commands that can be used to operate on skills."""
from __future__ import annotations
import collections
import itertools
import logging
from core import feconf
from core.constants import constants
from core.domain import caching_services
from core.domain import config_domain
from core.domain import html_cleaner
from core.domain import opportunity_services
from core.domain import role_services
from core.domain import skill_domain
from core.domain import skill_fetchers
from core.domain import state_domain
from core.domain import suggestion_services
from core.domain import taskqueue_services
from core.domain import topic_domain
from core.domain import topic_fetchers
from core.domain import topic_services
from core.domain import user_services
from core.platform import models
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, cast, overload
MYPY = False
if MYPY:
    from mypy_imports import question_models
    from mypy_imports import skill_models
    from mypy_imports import topic_models
    from mypy_imports import user_models
(skill_models, user_models, question_models, topic_models) = models.Registry.import_models([models.Names.SKILL, models.Names.USER, models.Names.QUESTION, models.Names.TOPIC])

def get_merged_skill_ids() -> List[str]:
    if False:
        print('Hello World!')
    'Returns the skill IDs of skills that have been merged.\n\n    Returns:\n        list(str). List of skill IDs of merged skills.\n    '
    return [skill.id for skill in skill_models.SkillModel.get_merged_skills()]

def get_all_skill_summaries() -> List[skill_domain.SkillSummary]:
    if False:
        return 10
    'Returns the summaries of all skills present in the datastore.\n\n    Returns:\n        list(SkillSummary). The list of summaries of all skills present in the\n        datastore.\n    '
    skill_summaries_models = skill_models.SkillSummaryModel.get_all()
    skill_summaries = [get_skill_summary_from_model(summary) for summary in skill_summaries_models]
    return skill_summaries

def _get_skill_summaries_in_batches(num_skills_to_fetch: int, urlsafe_start_cursor: Optional[str], sort_by: Optional[str]) -> Tuple[List[skill_domain.SkillSummary], Optional[str], bool]:
    if False:
        while True:
            i = 10
    'Returns the summaries of skills present in the datastore.\n\n    Args:\n        num_skills_to_fetch: int. Number of skills to fetch.\n        urlsafe_start_cursor: str or None. The cursor to the next page.\n        sort_by: str|None. A string indicating how to sort the result, or None\n            if no sort is required.\n\n    Returns:\n        3-tuple(skill_summaries, new_urlsafe_start_cursor, more). where:\n            skill_summaries: list(SkillSummary). The list of skill summaries.\n                The number of returned skill summaries might include more than\n                the requested number. Hence, the cursor returned will represent\n                the point to which those results were fetched (and not the\n                "num_skills_to_fetch" point).\n            urlsafe_start_cursor: str or None. A query cursor pointing to the\n                next batch of results. If there are no more results, this might\n                be None.\n            more: bool. If True, there are (probably) more results after this\n                batch. If False, there are no further results after this batch.\n    '
    (skill_summaries_models, new_urlsafe_start_cursor, more) = skill_models.SkillSummaryModel.fetch_page(2 * num_skills_to_fetch, urlsafe_start_cursor, sort_by)
    skill_summaries = [get_skill_summary_from_model(summary) for summary in skill_summaries_models]
    return (skill_summaries, new_urlsafe_start_cursor, more)

def get_filtered_skill_summaries(num_skills_to_fetch: int, status: Optional[str], classroom_name: Optional[str], keywords: List[str], sort_by: Optional[str], urlsafe_start_cursor: Optional[str]) -> Tuple[List[skill_domain.AugmentedSkillSummary], Optional[str], bool]:
    if False:
        print('Hello World!')
    'Returns all the skill summary dicts after filtering.\n\n    Args:\n        num_skills_to_fetch: int. Number of skills to fetch.\n        status: str|None. The status of the skill, or None if no status is\n            provided to filter skills id.\n        classroom_name: str|None. The classroom_name of the topic to which\n            the skill is assigned to.\n        keywords: list(str). The keywords to look for\n            in the skill description.\n        sort_by: str|None. A string indicating how to sort the result, or None\n            if no sorting is required.\n        urlsafe_start_cursor: str or None. The cursor to the next page.\n\n    Returns:\n        3-tuple(augmented_skill_summaries, new_urlsafe_start_cursor, more).\n        Where:\n            augmented_skill_summaries: list(AugmentedSkillSummary). The list of\n                augmented skill summaries. The number of returned skills might\n                include more than the requested number. Hence, the cursor\n                returned will represent the point to which those results were\n                fetched (and not the "num_skills_to_fetch" point).\n            new_urlsafe_start_cursor: str or None. A query cursor pointing to\n                the next batch of results. If there are no more results, this\n                might be None.\n            more: bool. If True, there are (probably) more results after this\n                batch. If False, there are no further results after this batch.\n    '
    augmented_skill_summaries: List[skill_domain.AugmentedSkillSummary] = []
    new_urlsafe_start_cursor = urlsafe_start_cursor
    more = True
    while len(augmented_skill_summaries) < num_skills_to_fetch and more:
        (augmented_skill_summaries_batch, new_urlsafe_start_cursor, more) = _get_augmented_skill_summaries_in_batches(num_skills_to_fetch, new_urlsafe_start_cursor, sort_by)
        filtered_augmented_skill_summaries = _filter_skills_by_status(augmented_skill_summaries_batch, status)
        filtered_augmented_skill_summaries = _filter_skills_by_classroom(filtered_augmented_skill_summaries, classroom_name)
        filtered_augmented_skill_summaries = _filter_skills_by_keywords(filtered_augmented_skill_summaries, keywords)
        augmented_skill_summaries.extend(filtered_augmented_skill_summaries)
    return (augmented_skill_summaries, new_urlsafe_start_cursor, more)

def _get_augmented_skill_summaries_in_batches(num_skills_to_fetch: int, urlsafe_start_cursor: Optional[str], sort_by: Optional[str]) -> Tuple[List[skill_domain.AugmentedSkillSummary], Optional[str], bool]:
    if False:
        return 10
    'Returns all the Augmented skill summaries after attaching\n    topic and classroom.\n\n    Returns:\n        3-tuple(augmented_skill_summaries, urlsafe_start_cursor, more). Where:\n            augmented_skill_summaries: list(AugmentedSkillSummary). The list of\n                skill summaries.\n            urlsafe_start_cursor: str or None. A query cursor pointing to the\n                next batch of results. If there are no more results, this might\n                be None.\n            more: bool. If True, there are (probably) more results after this\n                batch. If False, there are no further results after this batch.\n    '
    (skill_summaries, new_urlsafe_start_cursor, more) = _get_skill_summaries_in_batches(num_skills_to_fetch, urlsafe_start_cursor, sort_by)
    assigned_skill_ids: Dict[str, Dict[str, List[str]]] = collections.defaultdict(lambda : {'topic_names': [], 'classroom_names': []})
    all_topic_models = topic_models.TopicModel.get_all()
    all_topics = [topic_fetchers.get_topic_from_model(topic_model) for topic_model in all_topic_models if topic_model is not None]
    topic_classroom_dict = {}
    all_classrooms_dict = config_domain.CLASSROOM_PAGES_DATA.value
    for classroom in all_classrooms_dict:
        for topic_id in classroom['topic_ids']:
            topic_classroom_dict[topic_id] = classroom['name']
    for topic in all_topics:
        for skill_id in topic.get_all_skill_ids():
            assigned_skill_ids[skill_id]['topic_names'].append(topic.name)
            assigned_skill_ids[skill_id]['classroom_names'].append(topic_classroom_dict.get(topic.id, None))
    augmented_skill_summaries = []
    for skill_summary in skill_summaries:
        topic_names = []
        classroom_names = []
        if skill_summary.id in assigned_skill_ids:
            topic_names = assigned_skill_ids[skill_summary.id]['topic_names']
            classroom_names = assigned_skill_ids[skill_summary.id]['classroom_names']
        augmented_skill_summary = skill_domain.AugmentedSkillSummary(skill_summary.id, skill_summary.description, skill_summary.language_code, skill_summary.version, skill_summary.misconception_count, skill_summary.worked_examples_count, topic_names, classroom_names, skill_summary.skill_model_created_on, skill_summary.skill_model_last_updated)
        augmented_skill_summaries.append(augmented_skill_summary)
    return (augmented_skill_summaries, new_urlsafe_start_cursor, more)

def _filter_skills_by_status(augmented_skill_summaries: List[skill_domain.AugmentedSkillSummary], status: Optional[str]) -> List[skill_domain.AugmentedSkillSummary]:
    if False:
        while True:
            i = 10
    'Returns the skill summary dicts after filtering by status.\n\n    Args:\n        augmented_skill_summaries: list(AugmentedSkillSummary). The list\n            of augmented skill summaries.\n        status: str|None. The status of the skill, or None if no status is\n            provided to filter skills id.\n\n    Returns:\n        list(AugmentedSkillSummary). The list of AugmentedSkillSummaries\n        matching the given status.\n    '
    if status is None or status == constants.SKILL_STATUS_OPTIONS['ALL']:
        return augmented_skill_summaries
    elif status == constants.SKILL_STATUS_OPTIONS['UNASSIGNED']:
        unassigned_augmented_skill_summaries = []
        for augmented_skill_summary in augmented_skill_summaries:
            if not augmented_skill_summary.topic_names:
                unassigned_augmented_skill_summaries.append(augmented_skill_summary)
        return unassigned_augmented_skill_summaries
    elif status == constants.SKILL_STATUS_OPTIONS['ASSIGNED']:
        assigned_augmented_skill_summaries = []
        for augmented_skill_summary in augmented_skill_summaries:
            if augmented_skill_summary.topic_names:
                assigned_augmented_skill_summaries.append(augmented_skill_summary)
        return assigned_augmented_skill_summaries
    return []

def _filter_skills_by_classroom(augmented_skill_summaries: List[skill_domain.AugmentedSkillSummary], classroom_name: Optional[str]) -> List[skill_domain.AugmentedSkillSummary]:
    if False:
        return 10
    'Returns the skill summary dicts after filtering by classroom_name.\n\n    Args:\n        augmented_skill_summaries: list(AugmentedSkillSummary).\n            The list of augmented skill summaries.\n        classroom_name: str|None. The classroom_name of the topic to which\n            the skill is assigned to.\n\n    Returns:\n        list(AugmentedSkillSummary). The list of augmented skill summaries with\n        the given classroom name.\n    '
    if classroom_name is None or classroom_name == 'All':
        return augmented_skill_summaries
    augmented_skill_summaries_with_classroom_name = []
    for augmented_skill_summary in augmented_skill_summaries:
        if classroom_name in augmented_skill_summary.classroom_names:
            augmented_skill_summaries_with_classroom_name.append(augmented_skill_summary)
    return augmented_skill_summaries_with_classroom_name

def _filter_skills_by_keywords(augmented_skill_summaries: List[skill_domain.AugmentedSkillSummary], keywords: List[str]) -> List[skill_domain.AugmentedSkillSummary]:
    if False:
        return 10
    'Returns whether the keywords match the skill description.\n\n    Args:\n        augmented_skill_summaries: list(AugmentedSkillSummary). The augmented\n            skill summaries.\n        keywords: list(str). The keywords to match.\n\n    Returns:\n        list(AugmentedSkillSummary). The list of augmented skill summaries\n        matching the given keywords.\n    '
    if not keywords:
        return augmented_skill_summaries
    filtered_augmented_skill_summaries = []
    for augmented_skill_summary in augmented_skill_summaries:
        if any((augmented_skill_summary.description.lower().find(keyword.lower()) != -1 for keyword in keywords)):
            filtered_augmented_skill_summaries.append(augmented_skill_summary)
    return filtered_augmented_skill_summaries

def get_multi_skill_summaries(skill_ids: List[str]) -> List[skill_domain.SkillSummary]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of skill summaries matching the skill IDs provided.\n\n    Args:\n        skill_ids: list(str). List of skill IDs to get skill summaries for.\n\n    Returns:\n        list(SkillSummary). The list of summaries of skills matching the\n        provided IDs.\n    '
    skill_summaries_models = skill_models.SkillSummaryModel.get_multi(skill_ids)
    skill_summaries = [get_skill_summary_from_model(skill_summary_model) for skill_summary_model in skill_summaries_models if skill_summary_model is not None]
    return skill_summaries

def get_rubrics_of_skills(skill_ids: List[str]) -> Tuple[Dict[str, Optional[List[skill_domain.RubricDict]]], List[str]]:
    if False:
        return 10
    'Returns a list of rubrics corresponding to given skills.\n\n    Args:\n        skill_ids: list(str). The list of skill IDs.\n\n    Returns:\n        dict, list(str). The skill rubrics of skills keyed by their\n        corresponding ids and the list of deleted skill ids, if any.\n    '
    skills = skill_fetchers.get_multi_skills(skill_ids, strict=False)
    skill_id_to_rubrics_dict: Dict[str, Optional[List[skill_domain.RubricDict]]] = {}
    for skill in skills:
        if skill is not None:
            rubric_dicts = [rubric.to_dict() for rubric in skill.rubrics]
            skill_id_to_rubrics_dict[skill.id] = rubric_dicts
    deleted_skill_ids = []
    for skill_id in skill_ids:
        if skill_id not in skill_id_to_rubrics_dict:
            skill_id_to_rubrics_dict[skill_id] = None
            deleted_skill_ids.append(skill_id)
    return (skill_id_to_rubrics_dict, deleted_skill_ids)

def get_descriptions_of_skills(skill_ids: List[str]) -> Tuple[Dict[str, str], List[str]]:
    if False:
        print('Hello World!')
    'Returns a list of skill descriptions corresponding to the given skills.\n\n    Args:\n        skill_ids: list(str). The list of skill ids.\n\n    Returns:\n        dict, list(str). The skill descriptions of skills keyed by their\n        corresponding ids and the list of deleted skill ids, if any.\n    '
    skill_summaries = get_multi_skill_summaries(skill_ids)
    skill_id_to_description_dict: Dict[str, str] = {}
    for skill_summary in skill_summaries:
        if skill_summary is not None:
            skill_id_to_description_dict[skill_summary.id] = skill_summary.description
    deleted_skill_ids = []
    for skill_id in skill_ids:
        if skill_id not in skill_id_to_description_dict:
            deleted_skill_ids.append(skill_id)
    return (skill_id_to_description_dict, deleted_skill_ids)

def get_skill_summary_from_model(skill_summary_model: skill_models.SkillSummaryModel) -> skill_domain.SkillSummary:
    if False:
        return 10
    'Returns a domain object for an Oppia skill summary given a\n    skill summary model.\n\n    Args:\n        skill_summary_model: SkillSummaryModel. The skill summary model object\n            to get corresponding domain object.\n\n    Returns:\n        SkillSummary. The domain object corresponding to given skill summmary\n        model.\n    '
    return skill_domain.SkillSummary(skill_summary_model.id, skill_summary_model.description, skill_summary_model.language_code, skill_summary_model.version, skill_summary_model.misconception_count, skill_summary_model.worked_examples_count, skill_summary_model.skill_model_created_on, skill_summary_model.skill_model_last_updated)

def get_image_filenames_from_skill(skill: skill_domain.Skill) -> List[str]:
    if False:
        print('Hello World!')
    'Get the image filenames from the skill.\n\n    Args:\n        skill: Skill. The skill itself.\n\n    Returns:\n        list(str). List containing the name of the image files in skill.\n    '
    html_list = skill.get_all_html_content_strings()
    return html_cleaner.get_image_filenames_from_html_strings(html_list)

def get_all_topic_assignments_for_skill(skill_id: str) -> List[skill_domain.TopicAssignment]:
    if False:
        i = 10
        return i + 15
    'Returns a list containing all the topics to which the given skill is\n    assigned along with topic details.\n\n    Args:\n        skill_id: str. ID of the skill.\n\n    Returns:\n        list(TopicAssignment). A list of TopicAssignment domain objects.\n    '
    topic_assignments = []
    topics = topic_fetchers.get_all_topics()
    for topic in topics:
        if skill_id in topic.get_all_skill_ids():
            subtopic_id = None
            for subtopic in topic.subtopics:
                if skill_id in subtopic.skill_ids:
                    subtopic_id = subtopic.id
                    break
            topic_assignments.append(skill_domain.TopicAssignment(topic.id, topic.name, topic.version, subtopic_id))
    return topic_assignments

def get_topic_names_with_given_skill_in_diagnostic_test(skill_id: str) -> List[str]:
    if False:
        i = 10
        return i + 15
    "Returns a list of topic names for which the given skill is assigned\n    to that topic's diagnostic test.\n\n    Args:\n        skill_id: str. ID of the skill.\n\n    Returns:\n        list(str). A list of topic names for which the given skill is assigned\n        to that topic's diagnostic test.\n    "
    topics = topic_fetchers.get_all_topics()
    topic_names = []
    for topic in topics:
        if skill_id in topic.skill_ids_for_diagnostic_test:
            topic_names.append(topic.name)
    return topic_names

def replace_skill_id_in_all_topics(user_id: str, old_skill_id: str, new_skill_id: str) -> None:
    if False:
        print('Hello World!')
    'Replaces the old skill id with the new one in all the associated topics.\n\n    Args:\n        user_id: str. The unique user ID of the user.\n        old_skill_id: str. The old skill id.\n        new_skill_id: str. The new skill id.\n\n    Raises:\n        Exception. The new skill already present.\n    '
    all_topics = topic_fetchers.get_all_topics()
    for topic in all_topics:
        change_list = []
        if old_skill_id in topic.get_all_skill_ids():
            if new_skill_id in topic.get_all_skill_ids():
                raise Exception("Found topic '%s' contains the two skills to be merged. Please unassign one of these skills from topic and retry this operation." % topic.name)
            if old_skill_id in topic.uncategorized_skill_ids:
                change_list.extend([topic_domain.TopicChange({'cmd': 'remove_uncategorized_skill_id', 'uncategorized_skill_id': old_skill_id}), topic_domain.TopicChange({'cmd': 'add_uncategorized_skill_id', 'new_uncategorized_skill_id': new_skill_id})])
            for subtopic in topic.subtopics:
                if old_skill_id in subtopic.skill_ids:
                    change_list.extend([topic_domain.TopicChange({'cmd': topic_domain.CMD_REMOVE_SKILL_ID_FROM_SUBTOPIC, 'subtopic_id': subtopic.id, 'skill_id': old_skill_id}), topic_domain.TopicChange({'cmd': 'remove_uncategorized_skill_id', 'uncategorized_skill_id': old_skill_id}), topic_domain.TopicChange({'cmd': 'add_uncategorized_skill_id', 'new_uncategorized_skill_id': new_skill_id}), topic_domain.TopicChange({'cmd': topic_domain.CMD_MOVE_SKILL_ID_TO_SUBTOPIC, 'old_subtopic_id': None, 'new_subtopic_id': subtopic.id, 'skill_id': new_skill_id})])
                    break
            topic_services.update_topic_and_subtopic_pages(user_id, topic.id, change_list, 'Replace skill id %s with skill id %s in the topic' % (old_skill_id, new_skill_id))

def remove_skill_from_all_topics(user_id: str, skill_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Deletes the skill with the given id from all the associated topics.\n\n    Args:\n        user_id: str. The unique user ID of the user.\n        skill_id: str. ID of the skill.\n    '
    all_topics = topic_fetchers.get_all_topics()
    for topic in all_topics:
        change_list = []
        if skill_id in topic.get_all_skill_ids():
            for subtopic in topic.subtopics:
                if skill_id in subtopic.skill_ids:
                    change_list.append(topic_domain.TopicChange({'cmd': 'remove_skill_id_from_subtopic', 'subtopic_id': subtopic.id, 'skill_id': skill_id}))
                    break
            change_list.append(topic_domain.TopicChange({'cmd': 'remove_uncategorized_skill_id', 'uncategorized_skill_id': skill_id}))
            skill_name = get_skill_summary_by_id(skill_id).description
            topic_services.update_topic_and_subtopic_pages(user_id, topic.id, change_list, 'Removed skill with id %s and name %s from the topic' % (skill_id, skill_name))

@overload
def get_skill_summary_by_id(skill_id: str) -> skill_domain.SkillSummary:
    if False:
        while True:
            i = 10
    ...

@overload
def get_skill_summary_by_id(skill_id: str, *, strict: Literal[True]) -> skill_domain.SkillSummary:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_skill_summary_by_id(skill_id: str, *, strict: Literal[False]) -> Optional[skill_domain.SkillSummary]:
    if False:
        while True:
            i = 10
    ...

def get_skill_summary_by_id(skill_id: str, strict: bool=True) -> Optional[skill_domain.SkillSummary]:
    if False:
        return 10
    'Returns a domain object representing a skill summary.\n\n    Args:\n        skill_id: str. ID of the skill summary.\n        strict: bool. Whether to fail noisily if no skill summary with the given\n            id exists in the datastore.\n\n    Returns:\n        SkillSummary. The skill summary domain object corresponding to a skill\n        with the given skill_id.\n    '
    skill_summary_model = skill_models.SkillSummaryModel.get(skill_id, strict=strict)
    if skill_summary_model:
        skill_summary = get_skill_summary_from_model(skill_summary_model)
        return skill_summary
    else:
        return None

def get_new_skill_id() -> str:
    if False:
        while True:
            i = 10
    'Returns a new skill id.\n\n    Returns:\n        str. A new skill id.\n    '
    return skill_models.SkillModel.get_new_id('')

def _create_skill(committer_id: str, skill: skill_domain.Skill, commit_message: str, commit_cmds: List[skill_domain.SkillChange]) -> None:
    if False:
        return 10
    'Creates a new skill.\n\n    Args:\n        committer_id: str. ID of the committer.\n        skill: Skill. The skill domain object.\n        commit_message: str. A description of changes made to the skill.\n        commit_cmds: list(SkillChange). A list of change commands made to the\n            given skill.\n    '
    skill.validate()
    model = skill_models.SkillModel(id=skill.id, description=skill.description, language_code=skill.language_code, misconceptions=[misconception.to_dict() for misconception in skill.misconceptions], rubrics=[rubric.to_dict() for rubric in skill.rubrics], skill_contents=skill.skill_contents.to_dict(), next_misconception_id=skill.next_misconception_id, misconceptions_schema_version=skill.misconceptions_schema_version, rubric_schema_version=skill.rubric_schema_version, skill_contents_schema_version=skill.skill_contents_schema_version, superseding_skill_id=skill.superseding_skill_id, all_questions_merged=skill.all_questions_merged, prerequisite_skill_ids=skill.prerequisite_skill_ids)
    commit_cmd_dicts = [commit_cmd.to_dict() for commit_cmd in commit_cmds]
    model.commit(committer_id, commit_message, commit_cmd_dicts)
    skill.version += 1
    create_skill_summary(skill.id)
    opportunity_services.create_skill_opportunity(skill.id, skill.description)

def does_skill_with_description_exist(description: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Checks if skill with provided description exists.\n\n    Args:\n        description: str. The description for the skill.\n\n    Returns:\n        bool. Whether the the description for the skill exists.\n    '
    existing_skill = skill_fetchers.get_skill_by_description(description)
    return existing_skill is not None

def save_new_skill(committer_id: str, skill: skill_domain.Skill) -> None:
    if False:
        i = 10
        return i + 15
    'Saves a new skill.\n\n    Args:\n        committer_id: str. ID of the committer.\n        skill: Skill. Skill to be saved.\n    '
    commit_message = 'New skill created.'
    _create_skill(committer_id, skill, commit_message, [skill_domain.SkillChange({'cmd': skill_domain.CMD_CREATE_NEW})])

def apply_change_list(skill_id: str, change_list: List[skill_domain.SkillChange], committer_id: str) -> skill_domain.Skill:
    if False:
        while True:
            i = 10
    'Applies a changelist to a skill and returns the result.\n\n    Args:\n        skill_id: str. ID of the given skill.\n        change_list: list(SkillChange). A change list to be applied to the given\n            skill.\n        committer_id: str. The ID of the committer of this change list.\n\n    Returns:\n        Skill. The resulting skill domain object.\n\n    Raises:\n        Exception. The user does not have enough rights to edit the\n            skill description.\n        Exception. Invalid change dict.\n    '
    skill = skill_fetchers.get_skill_by_id(skill_id)
    user = user_services.get_user_actions_info(committer_id)
    try:
        for change in change_list:
            if change.cmd == skill_domain.CMD_UPDATE_SKILL_PROPERTY:
                if change.property_name == skill_domain.SKILL_PROPERTY_DESCRIPTION:
                    if role_services.ACTION_EDIT_SKILL_DESCRIPTION not in user.actions:
                        raise Exception('The user does not have enough rights to edit the skill description.')
                    update_description_cmd = cast(skill_domain.UpdateSkillPropertyDescriptionCmd, change)
                    skill.update_description(update_description_cmd.new_value)
                    opportunity_services.update_skill_opportunity_skill_description(skill.id, update_description_cmd.new_value)
                elif change.property_name == skill_domain.SKILL_PROPERTY_LANGUAGE_CODE:
                    update_language_code_cmd = cast(skill_domain.UpdateSkillPropertyLanguageCodeCmd, change)
                    skill.update_language_code(update_language_code_cmd.new_value)
                elif change.property_name == skill_domain.SKILL_PROPERTY_SUPERSEDING_SKILL_ID:
                    update_superseding_skill_id_cmd = cast(skill_domain.UpdateSkillPropertySupersedingSkillIdCmd, change)
                    skill.update_superseding_skill_id(update_superseding_skill_id_cmd.new_value)
                elif change.property_name == skill_domain.SKILL_PROPERTY_ALL_QUESTIONS_MERGED:
                    update_all_questions_merged_cmd = cast(skill_domain.UpdateSkillPropertyAllQuestionsMergedCmd, change)
                    skill.record_that_all_questions_are_merged(update_all_questions_merged_cmd.new_value)
            elif change.cmd == skill_domain.CMD_UPDATE_SKILL_CONTENTS_PROPERTY:
                if change.property_name == skill_domain.SKILL_CONTENTS_PROPERTY_EXPLANATION:
                    update_explanation_cmd = cast(skill_domain.UpdateSkillContentsPropertyExplanationCmd, change)
                    explanation = state_domain.SubtitledHtml.from_dict(update_explanation_cmd.new_value)
                    explanation.validate()
                    skill.update_explanation(explanation)
                elif change.property_name == skill_domain.SKILL_CONTENTS_PROPERTY_WORKED_EXAMPLES:
                    update_worked_examples_cmd = cast(skill_domain.UpdateSkillContentsPropertyWorkedExamplesCmd, change)
                    worked_examples_list: List[skill_domain.WorkedExample] = []
                    for worked_example in update_worked_examples_cmd.new_value:
                        worked_examples_list.append(skill_domain.WorkedExample.from_dict(worked_example))
                    skill.update_worked_examples(worked_examples_list)
            elif change.cmd == skill_domain.CMD_ADD_SKILL_MISCONCEPTION:
                add_skill_misconception_cmd = cast(skill_domain.AddSkillMisconceptionCmd, change)
                misconception = skill_domain.Misconception.from_dict(add_skill_misconception_cmd.new_misconception_dict)
                skill.add_misconception(misconception)
            elif change.cmd == skill_domain.CMD_DELETE_SKILL_MISCONCEPTION:
                delete_misconception_cmd = cast(skill_domain.DeleteSkillMisconceptionCmd, change)
                skill.delete_misconception(delete_misconception_cmd.misconception_id)
            elif change.cmd == skill_domain.CMD_ADD_PREREQUISITE_SKILL:
                add_prerequisite_skill_cmd = cast(skill_domain.AddPrerequisiteSkillCmd, change)
                skill.add_prerequisite_skill(add_prerequisite_skill_cmd.skill_id)
            elif change.cmd == skill_domain.CMD_DELETE_PREREQUISITE_SKILL:
                delete_prerequisite_skill_cmd = cast(skill_domain.DeletePrerequisiteSkillCmd, change)
                skill.delete_prerequisite_skill(delete_prerequisite_skill_cmd.skill_id)
            elif change.cmd == skill_domain.CMD_UPDATE_RUBRICS:
                update_rubric_cmd = cast(skill_domain.UpdateRubricsCmd, change)
                skill.update_rubric(update_rubric_cmd.difficulty, update_rubric_cmd.explanations)
            elif change.cmd == skill_domain.CMD_UPDATE_SKILL_MISCONCEPTIONS_PROPERTY:
                if change.property_name == skill_domain.SKILL_MISCONCEPTIONS_PROPERTY_NAME:
                    update_property_name_cmd = cast(skill_domain.UpdateSkillMisconceptionPropertyNameCmd, change)
                    skill.update_misconception_name(update_property_name_cmd.misconception_id, update_property_name_cmd.new_value)
                elif change.property_name == skill_domain.SKILL_MISCONCEPTIONS_PROPERTY_NOTES:
                    update_property_notes_cmd = cast(skill_domain.UpdateSkillMisconceptionPropertyNotesCmd, change)
                    skill.update_misconception_notes(update_property_notes_cmd.misconception_id, update_property_notes_cmd.new_value)
                elif change.property_name == skill_domain.SKILL_MISCONCEPTIONS_PROPERTY_FEEDBACK:
                    update_property_feedback_cmd = cast(skill_domain.UpdateSkillMisconceptionPropertyFeedbackCmd, change)
                    skill.update_misconception_feedback(update_property_feedback_cmd.misconception_id, update_property_feedback_cmd.new_value)
                elif change.property_name == skill_domain.SKILL_MISCONCEPTIONS_PROPERTY_MUST_BE_ADDRESSED:
                    update_property_must_be_addressed_cmd = cast(skill_domain.UpdateSkillMisconceptionPropertyMustBeAddressedCmd, change)
                    skill.update_misconception_must_be_addressed(update_property_must_be_addressed_cmd.misconception_id, update_property_must_be_addressed_cmd.new_value)
                else:
                    raise Exception('Invalid change dict.')
            elif change.cmd in (skill_domain.CMD_MIGRATE_CONTENTS_SCHEMA_TO_LATEST_VERSION, skill_domain.CMD_MIGRATE_MISCONCEPTIONS_SCHEMA_TO_LATEST_VERSION, skill_domain.CMD_MIGRATE_RUBRICS_SCHEMA_TO_LATEST_VERSION):
                continue
        return skill
    except Exception as e:
        logging.error('%s %s %s %s' % (e.__class__.__name__, e, skill_id, change_list))
        raise e

def populate_skill_model_fields(skill_model: skill_models.SkillModel, skill: skill_domain.Skill) -> skill_models.SkillModel:
    if False:
        while True:
            i = 10
    'Populate skill model with the data from skill object.\n\n    Args:\n        skill_model: SkillModel. The model to populate.\n        skill: Skill. The skill domain object which should be used to\n            populate the model.\n\n    Returns:\n        SkillModel. Populated model.\n    '
    skill_model.description = skill.description
    skill_model.language_code = skill.language_code
    skill_model.superseding_skill_id = skill.superseding_skill_id
    skill_model.all_questions_merged = skill.all_questions_merged
    skill_model.prerequisite_skill_ids = skill.prerequisite_skill_ids
    skill_model.misconceptions_schema_version = skill.misconceptions_schema_version
    skill_model.rubric_schema_version = skill.rubric_schema_version
    skill_model.skill_contents_schema_version = skill.skill_contents_schema_version
    skill_model.skill_contents = skill.skill_contents.to_dict()
    skill_model.misconceptions = [misconception.to_dict() for misconception in skill.misconceptions]
    skill_model.rubrics = [rubric.to_dict() for rubric in skill.rubrics]
    skill_model.next_misconception_id = skill.next_misconception_id
    return skill_model

def _save_skill(committer_id: str, skill: skill_domain.Skill, commit_message: str, change_list: List[skill_domain.SkillChange]) -> None:
    if False:
        print('Hello World!')
    'Validates a skill and commits it to persistent storage. If\n    successful, increments the version number of the incoming skill domain\n    object by 1.\n\n    Args:\n        committer_id: str. ID of the given committer.\n        skill: Skill. The skill domain object to be saved.\n        commit_message: str. The commit message.\n        change_list: list(SkillChange). List of changes applied to a skill.\n\n    Raises:\n        Exception. The skill model and the incoming skill domain object have\n            different version numbers.\n        Exception. Received invalid change list.\n    '
    if not change_list:
        raise Exception('Unexpected error: received an invalid change list when trying to save skill %s: %s' % (skill.id, change_list))
    skill.validate()
    skill_model = skill_models.SkillModel.get(skill.id, strict=True)
    if skill.version > skill_model.version:
        raise Exception('Unexpected error: trying to update version %s of skill from version %s. Please reload the page and try again.' % (skill_model.version, skill.version))
    if skill.version < skill_model.version:
        raise Exception('Trying to update version %s of skill from version %s, which is too old. Please reload the page and try again.' % (skill_model.version, skill.version))
    skill_model = populate_skill_model_fields(skill_model, skill)
    change_dicts = [change.to_dict() for change in change_list]
    skill_model.commit(committer_id, commit_message, change_dicts)
    caching_services.delete_multi(caching_services.CACHE_NAMESPACE_SKILL, None, [skill.id])
    skill.version += 1

def update_skill(committer_id: str, skill_id: str, change_list: List[skill_domain.SkillChange], commit_message: Optional[str]) -> None:
    if False:
        return 10
    'Updates a skill. Commits changes.\n\n    Args:\n        committer_id: str. The id of the user who is performing the update\n            action.\n        skill_id: str. The skill id.\n        change_list: list(SkillChange). These changes are applied in sequence to\n            produce the resulting skill.\n        commit_message: str or None. A description of changes made to the\n            skill. For published skills, this must be present; for\n            unpublished skills, it may be equal to None.\n\n    Raises:\n        ValueError. No commit message was provided.\n    '
    if not commit_message:
        raise ValueError('Expected a commit message, received none.')
    skill = apply_change_list(skill_id, change_list, committer_id)
    _save_skill(committer_id, skill, commit_message, change_list)
    create_skill_summary(skill.id)
    misconception_is_deleted = any((change.cmd == skill_domain.CMD_DELETE_SKILL_MISCONCEPTION for change in change_list))
    if misconception_is_deleted:
        deleted_skill_misconception_ids: List[str] = []
        for change in change_list:
            if change.cmd == skill_domain.CMD_DELETE_SKILL_MISCONCEPTION:
                delete_skill_misconception_cmd = cast(skill_domain.DeleteSkillMisconceptionCmd, change)
                deleted_skill_misconception_ids.append(skill.generate_skill_misconception_id(delete_skill_misconception_cmd.misconception_id))
        taskqueue_services.defer(taskqueue_services.FUNCTION_ID_UNTAG_DELETED_MISCONCEPTIONS, taskqueue_services.QUEUE_NAME_ONE_OFF_JOBS, committer_id, skill_id, skill.description, deleted_skill_misconception_ids)

def delete_skill(committer_id: str, skill_id: str, force_deletion: bool=False) -> None:
    if False:
        print('Hello World!')
    'Deletes the skill with the given skill_id.\n\n    Args:\n        committer_id: str. ID of the committer.\n        skill_id: str. ID of the skill to be deleted.\n        force_deletion: bool. If true, the skill and its history are fully\n            deleted and are unrecoverable. Otherwise, the skill and all\n            its history are marked as deleted, but the corresponding models are\n            still retained in the datastore. This last option is the preferred\n            one.\n    '
    skill_models.SkillModel.delete_multi([skill_id], committer_id, '', force_deletion=force_deletion)
    caching_services.delete_multi(caching_services.CACHE_NAMESPACE_SKILL, None, [skill_id])
    delete_skill_summary(skill_id)
    opportunity_services.delete_skill_opportunity(skill_id)
    suggestion_services.auto_reject_question_suggestions_for_skill_id(skill_id)

def delete_skill_summary(skill_id: str) -> None:
    if False:
        return 10
    'Delete a skill summary model.\n\n    Args:\n        skill_id: str. ID of the skill whose skill summary is to\n            be deleted.\n    '
    skill_summary_model = skill_models.SkillSummaryModel.get(skill_id, strict=False)
    if skill_summary_model is not None:
        skill_summary_model.delete()

def compute_summary_of_skill(skill: skill_domain.Skill) -> skill_domain.SkillSummary:
    if False:
        while True:
            i = 10
    'Create a SkillSummary domain object for a given Skill domain\n    object and return it.\n\n    Args:\n        skill: Skill. The skill object, for which the summary is to be computed.\n\n    Returns:\n        SkillSummary. The computed summary for the given skill.\n\n    Raises:\n        Exception. No data available for when the skill was last_updated.\n        Exception. No data available for when the skill was created.\n    '
    skill_model_misconception_count = len(skill.misconceptions)
    skill_model_worked_examples_count = len(skill.skill_contents.worked_examples)
    if skill.created_on is None:
        raise Exception('No data available for when the skill was created.')
    if skill.last_updated is None:
        raise Exception('No data available for when the skill was last_updated.')
    skill_summary = skill_domain.SkillSummary(skill.id, skill.description, skill.language_code, skill.version, skill_model_misconception_count, skill_model_worked_examples_count, skill.created_on, skill.last_updated)
    return skill_summary

def create_skill_summary(skill_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Creates and stores a summary of the given skill.\n\n    Args:\n        skill_id: str. ID of the skill.\n    '
    skill = skill_fetchers.get_skill_by_id(skill_id)
    skill_summary = compute_summary_of_skill(skill)
    save_skill_summary(skill_summary)

def populate_skill_summary_model_fields(skill_summary_model: skill_models.SkillSummaryModel, skill_summary: skill_domain.SkillSummary) -> skill_models.SkillSummaryModel:
    if False:
        return 10
    'Populate skill summary model with the data from skill summary object.\n\n    Args:\n        skill_summary_model: SkillSummaryModel. The model to populate.\n        skill_summary: SkillSummary. The skill summary domain object which\n            should be used to populate the model.\n\n    Returns:\n        SkillSummaryModel. Populated model.\n    '
    skill_summary_dict = {'description': skill_summary.description, 'language_code': skill_summary.language_code, 'version': skill_summary.version, 'misconception_count': skill_summary.misconception_count, 'worked_examples_count': skill_summary.worked_examples_count, 'skill_model_last_updated': skill_summary.skill_model_last_updated, 'skill_model_created_on': skill_summary.skill_model_created_on}
    if skill_summary_model is not None:
        skill_summary_model.populate(**skill_summary_dict)
    else:
        skill_summary_dict['id'] = skill_summary.id
        skill_summary_model = skill_models.SkillSummaryModel(**skill_summary_dict)
    return skill_summary_model

def save_skill_summary(skill_summary: skill_domain.SkillSummary) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Save a skill summary domain object as a SkillSummaryModel\n    entity in the datastore.\n\n    Args:\n        skill_summary: SkillSummaryModel. The skill summary object to be saved\n            in the datastore.\n    '
    existing_skill_summary_model = skill_models.SkillSummaryModel.get_by_id(skill_summary.id)
    skill_summary_model = populate_skill_summary_model_fields(existing_skill_summary_model, skill_summary)
    skill_summary_model.update_timestamps()
    skill_summary_model.put()

def create_user_skill_mastery(user_id: str, skill_id: str, degree_of_mastery: float) -> None:
    if False:
        return 10
    'Creates skill mastery of a user.\n\n    Args:\n        user_id: str. The user ID of the user for whom to create the model.\n        skill_id: str. The unique id of the skill.\n        degree_of_mastery: float. The degree of mastery of user in the skill.\n    '
    user_skill_mastery = skill_domain.UserSkillMastery(user_id, skill_id, degree_of_mastery)
    save_user_skill_mastery(user_skill_mastery)

def save_user_skill_mastery(user_skill_mastery: skill_domain.UserSkillMastery) -> None:
    if False:
        return 10
    'Stores skill mastery of a user.\n\n    Args:\n        user_skill_mastery: dict. The user skill mastery model of a user.\n    '
    user_skill_mastery_model = user_models.UserSkillMasteryModel(id=user_models.UserSkillMasteryModel.construct_model_id(user_skill_mastery.user_id, user_skill_mastery.skill_id), user_id=user_skill_mastery.user_id, skill_id=user_skill_mastery.skill_id, degree_of_mastery=user_skill_mastery.degree_of_mastery)
    user_skill_mastery_model.update_timestamps()
    user_skill_mastery_model.put()

def create_multi_user_skill_mastery(user_id: str, degrees_of_mastery: Dict[str, float]) -> None:
    if False:
        i = 10
        return i + 15
    'Creates the mastery of a user in multiple skills.\n\n    Args:\n        user_id: str. The user ID of the user.\n        degrees_of_mastery: dict(str, float). The keys are the requested\n            skill IDs. The values are the corresponding mastery degree of\n            the user.\n    '
    user_skill_mastery_models = []
    for (skill_id, degree_of_mastery) in degrees_of_mastery.items():
        user_skill_mastery_models.append(user_models.UserSkillMasteryModel(id=user_models.UserSkillMasteryModel.construct_model_id(user_id, skill_id), user_id=user_id, skill_id=skill_id, degree_of_mastery=degree_of_mastery))
    user_models.UserSkillMasteryModel.update_timestamps_multi(user_skill_mastery_models)
    user_models.UserSkillMasteryModel.put_multi(user_skill_mastery_models)

def get_user_skill_mastery(user_id: str, skill_id: str) -> Optional[float]:
    if False:
        for i in range(10):
            print('nop')
    'Fetches the mastery of user in a particular skill.\n\n    Args:\n        user_id: str. The user ID of the user.\n        skill_id: str. Unique id of the skill for which mastery degree is\n            requested.\n\n    Returns:\n        float or None. Mastery degree of the user for the requested skill, or\n        None if UserSkillMasteryModel does not exist for the skill.\n    '
    model_id = user_models.UserSkillMasteryModel.construct_model_id(user_id, skill_id)
    user_skill_mastery_model = user_models.UserSkillMasteryModel.get(model_id, strict=False)
    if not user_skill_mastery_model:
        return None
    degree_of_mastery: float = user_skill_mastery_model.degree_of_mastery
    return degree_of_mastery

def get_multi_user_skill_mastery(user_id: str, skill_ids: List[str]) -> Dict[str, Optional[float]]:
    if False:
        print('Hello World!')
    'Fetches the mastery of user in multiple skills.\n\n    Args:\n        user_id: str. The user ID of the user.\n        skill_ids: list(str). Skill IDs of the skill for which mastery degree is\n            requested.\n\n    Returns:\n        dict(str, float|None). The keys are the requested skill IDs. The values\n        are the corresponding mastery degree of the user or None if\n        UserSkillMasteryModel does not exist for the skill.\n    '
    degrees_of_mastery: Dict[str, Optional[float]] = {}
    model_ids = []
    for skill_id in skill_ids:
        model_ids.append(user_models.UserSkillMasteryModel.construct_model_id(user_id, skill_id))
    skill_mastery_models = user_models.UserSkillMasteryModel.get_multi(model_ids)
    for (skill_id, skill_mastery_model) in zip(skill_ids, skill_mastery_models):
        if skill_mastery_model is None:
            degrees_of_mastery[skill_id] = None
        else:
            degrees_of_mastery[skill_id] = skill_mastery_model.degree_of_mastery
    return degrees_of_mastery

def get_multi_users_skills_mastery(user_ids: List[str], skill_ids: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    if False:
        i = 10
        return i + 15
    'Fetches the mastery of user in multiple skills.\n\n    Args:\n        user_ids: list(str). The user IDs of the users.\n        skill_ids: list(str). Skill IDs of the skill for which mastery degree is\n            requested.\n\n    Returns:\n        dict(str, dict(str, float|None)). The keys are the user IDs and values\n        are dictionaries with keys as requested skill IDs and values\n        as the corresponding mastery degree of the user or None if\n        UserSkillMasteryModel does not exist for the skill.\n    '
    all_combinations = list(itertools.product(user_ids, skill_ids))
    model_ids = []
    for (user_id, skill_id) in all_combinations:
        model_ids.append(user_models.UserSkillMasteryModel.construct_model_id(user_id, skill_id))
    skill_mastery_models = user_models.UserSkillMasteryModel.get_multi(model_ids)
    degrees_of_masteries: Dict[str, Dict[str, Optional[float]]] = {user_id: {} for user_id in user_ids}
    for (i, (user_id, skill_id)) in enumerate(all_combinations):
        skill_mastery_model = skill_mastery_models[i]
        if skill_mastery_model is None:
            degrees_of_masteries[user_id][skill_id] = None
        else:
            degrees_of_masteries[user_id][skill_id] = skill_mastery_model.degree_of_mastery
    return degrees_of_masteries

def skill_has_associated_questions(skill_id: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Returns whether or not any question has this skill attached.\n\n    Args:\n        skill_id: str. The skill ID of the user.\n\n    Returns:\n        bool. Whether any question has this skill attached.\n    '
    question_ids = question_models.QuestionSkillLinkModel.get_all_question_ids_linked_to_skill_id(skill_id)
    return len(question_ids) > 0

def get_sorted_skill_ids(degrees_of_mastery: Dict[str, Optional[float]]) -> List[str]:
    if False:
        i = 10
        return i + 15
    "Sort the dict based on the mastery value.\n\n    Args:\n        degrees_of_mastery: dict(str, float|None). Dict mapping\n            skill ids to mastery level. The mastery level can be\n            float or None.\n\n    Returns:\n        list. List of the initial skill id's based on the mastery level.\n    "
    skill_dict_with_float_value = {skill_id: degree for (skill_id, degree) in degrees_of_mastery.items() if degree is not None}
    sort_fn: Callable[[str], float] = lambda skill_id: skill_dict_with_float_value[skill_id] if skill_dict_with_float_value.get(skill_id) else 0
    sorted_skill_ids_with_float_value = sorted(skill_dict_with_float_value, key=sort_fn)
    skill_ids_with_none_value = [skill_id for (skill_id, degree) in degrees_of_mastery.items() if degree is None]
    sorted_skill_ids = skill_ids_with_none_value + sorted_skill_ids_with_float_value
    return sorted_skill_ids[:feconf.MAX_NUMBER_OF_SKILL_IDS]

def filter_skills_by_mastery(user_id: str, skill_ids: List[str]) -> List[str]:
    if False:
        print('Hello World!')
    'Given a list of skill_ids, it returns a list of\n    feconf.MAX_NUMBER_OF_SKILL_IDS skill_ids in which the user has\n    the least mastery.(Please note that python 2.7 considers the None\n    type smaller than any value, so None types will be returned first)\n\n    Args:\n        user_id: str. The unique user ID of the user.\n        skill_ids: list(str). The skill_ids that are to be filtered.\n\n    Returns:\n        list(str). A list of the filtered skill_ids.\n    '
    degrees_of_mastery = get_multi_user_skill_mastery(user_id, skill_ids)
    filtered_skill_ids = get_sorted_skill_ids(degrees_of_mastery)
    arranged_filtered_skill_ids = []
    for skill_id in skill_ids:
        if skill_id in filtered_skill_ids:
            arranged_filtered_skill_ids.append(skill_id)
    return arranged_filtered_skill_ids

def get_untriaged_skill_summaries(skill_summaries: List[skill_domain.SkillSummary], skill_ids_assigned_to_some_topic: Set[str], merged_skill_ids: List[str]) -> List[skill_domain.SkillSummary]:
    if False:
        return 10
    'Returns a list of skill summaries for all skills that are untriaged.\n\n    Args:\n        skill_summaries: list(SkillSummary). The list of all skill summary\n            domain objects.\n        skill_ids_assigned_to_some_topic: set(str). The set of skill ids which\n            are assigned to some topic.\n        merged_skill_ids: list(str). List of skill IDs of merged skills.\n\n    Returns:\n        list(SkillSummary). A list of skill summaries for all skills that\n        are untriaged.\n    '
    untriaged_skill_summaries = []
    for skill_summary in skill_summaries:
        skill_id = skill_summary.id
        if skill_id not in skill_ids_assigned_to_some_topic and skill_id not in merged_skill_ids:
            untriaged_skill_summaries.append(skill_summary)
    return untriaged_skill_summaries

def get_categorized_skill_ids_and_descriptions() -> skill_domain.CategorizedSkills:
    if False:
        while True:
            i = 10
    'Returns a CategorizedSkills domain object for all the skills that are\n    categorized.\n\n    Returns:\n        CategorizedSkills. An instance of the CategorizedSkills domain object\n        for all the skills that are categorized.\n    '
    topics = topic_fetchers.get_all_topics()
    categorized_skills = skill_domain.CategorizedSkills()
    skill_ids = []
    for topic in topics:
        subtopics = topic.subtopics
        subtopic_titles = [subtopic.title for subtopic in subtopics]
        categorized_skills.add_topic(topic.name, subtopic_titles)
        for skill_id in topic.uncategorized_skill_ids:
            skill_ids.append(skill_id)
        for subtopic in subtopics:
            for skill_id in subtopic.skill_ids:
                skill_ids.append(skill_id)
    skill_descriptions = get_descriptions_of_skills(skill_ids)[0]
    for topic in topics:
        subtopics = topic.subtopics
        for skill_id in topic.uncategorized_skill_ids:
            description = skill_descriptions[skill_id]
            categorized_skills.add_uncategorized_skill(topic.name, skill_id, description)
        for subtopic in subtopics:
            for skill_id in subtopic.skill_ids:
                description = skill_descriptions[skill_id]
                categorized_skills.add_subtopic_skill(topic.name, subtopic.title, skill_id, description)
    return categorized_skills