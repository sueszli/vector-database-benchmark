"""Commands for operations on topics, and related models."""
from __future__ import annotations
import collections
import logging
from core import feconf
from core import utils
from core.constants import constants
from core.domain import caching_services
from core.domain import change_domain
from core.domain import feedback_services
from core.domain import fs_services
from core.domain import opportunity_services
from core.domain import rights_domain
from core.domain import role_services
from core.domain import state_domain
from core.domain import story_domain
from core.domain import story_fetchers
from core.domain import story_services
from core.domain import subtopic_page_domain
from core.domain import subtopic_page_services
from core.domain import suggestion_services
from core.domain import topic_domain
from core.domain import topic_fetchers
from core.domain import user_domain
from core.domain import user_services
from core.platform import models
from typing import Dict, List, Optional, Sequence, Tuple, cast
MYPY = False
if MYPY:
    from mypy_imports import topic_models
(topic_models,) = models.Registry.import_models([models.Names.TOPIC])

def _create_topic(committer_id: str, topic: topic_domain.Topic, commit_message: str, commit_cmds: List[topic_domain.TopicChange]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Creates a new topic, and ensures that rights for a new topic\n    are saved first.\n\n    Args:\n        committer_id: str. ID of the committer.\n        topic: Topic. Topic domain object.\n        commit_message: str. A description of changes made to the topic.\n        commit_cmds: list(TopicChange). A list of TopicChange objects that\n            represent change commands made to the given topic.\n    '
    topic.validate()
    if does_topic_with_name_exist(topic.name):
        raise utils.ValidationError("Topic with name '%s' already exists" % topic.name)
    if does_topic_with_url_fragment_exist(topic.url_fragment):
        raise utils.ValidationError("Topic with URL Fragment '%s' already exists" % topic.url_fragment)
    create_new_topic_rights(topic.id, committer_id)
    model = topic_models.TopicModel(id=topic.id, name=topic.name, abbreviated_name=topic.abbreviated_name, url_fragment=topic.url_fragment, thumbnail_bg_color=topic.thumbnail_bg_color, thumbnail_filename=topic.thumbnail_filename, thumbnail_size_in_bytes=topic.thumbnail_size_in_bytes, canonical_name=topic.canonical_name, description=topic.description, language_code=topic.language_code, canonical_story_references=[reference.to_dict() for reference in topic.canonical_story_references], additional_story_references=[reference.to_dict() for reference in topic.additional_story_references], uncategorized_skill_ids=topic.uncategorized_skill_ids, subtopic_schema_version=topic.subtopic_schema_version, story_reference_schema_version=topic.story_reference_schema_version, next_subtopic_id=topic.next_subtopic_id, subtopics=[subtopic.to_dict() for subtopic in topic.subtopics], meta_tag_content=topic.meta_tag_content, practice_tab_is_displayed=topic.practice_tab_is_displayed, page_title_fragment_for_web=topic.page_title_fragment_for_web, skill_ids_for_diagnostic_test=topic.skill_ids_for_diagnostic_test)
    commit_cmd_dicts = [commit_cmd.to_dict() for commit_cmd in commit_cmds]
    model.commit(committer_id, commit_message, commit_cmd_dicts)
    topic.version += 1
    generate_topic_summary(topic.id)

def does_topic_with_name_exist(topic_name: str) -> bool:
    if False:
        while True:
            i = 10
    'Checks if the topic with provided name exists.\n\n    Args:\n        topic_name: str. The topic name.\n\n    Returns:\n        bool. Whether the the topic name exists.\n\n    Raises:\n        Exception. Topic name is not a string.\n    '
    if not isinstance(topic_name, str):
        raise utils.ValidationError('Name should be a string.')
    existing_topic = topic_fetchers.get_topic_by_name(topic_name)
    return existing_topic is not None

def does_topic_with_url_fragment_exist(url_fragment: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Checks if topic with provided url fragment exists.\n\n    Args:\n        url_fragment: str. The url fragment for the topic.\n\n    Returns:\n        bool. Whether the the url fragment for the topic exists.\n\n    Raises:\n        Exception. Topic URL fragment is not a string.\n    '
    if not isinstance(url_fragment, str):
        raise utils.ValidationError('Topic URL fragment should be a string.')
    existing_topic = topic_fetchers.get_topic_by_url_fragment(url_fragment)
    return existing_topic is not None

def save_new_topic(committer_id: str, topic: topic_domain.Topic) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Saves a new topic.\n\n    Args:\n        committer_id: str. ID of the committer.\n        topic: Topic. Topic to be saved.\n    '
    commit_message = "New topic created with name '%s'." % topic.name
    _create_topic(committer_id, topic, commit_message, [topic_domain.TopicChange({'cmd': topic_domain.CMD_CREATE_NEW, 'name': topic.name})])

def apply_change_list(topic_id: str, change_list: Sequence[change_domain.BaseChange]) -> Tuple[topic_domain.Topic, Dict[str, subtopic_page_domain.SubtopicPage], List[int], List[int], Dict[str, List[subtopic_page_domain.SubtopicPageChange]]]:
    if False:
        for i in range(10):
            print('nop')
    'Applies a changelist to a topic and returns the result. The incoming\n    changelist should not have simultaneuous creations and deletion of\n    subtopics.\n\n    Args:\n        topic_id: str. ID of the given topic.\n        change_list: list(TopicChange). A change list to be applied to the given\n            topic.\n\n    Raises:\n        Exception. The incoming changelist had simultaneous creation and\n            deletion of subtopics.\n\n    Returns:\n        tuple(Topic, dict, list(int), list(int), list(SubtopicPageChange)). The\n        modified topic object, the modified subtopic pages dict keyed\n        by subtopic page id containing the updated domain objects of\n        each subtopic page, a list of ids of the deleted subtopics,\n        a list of ids of the newly created subtopics and a list of changes\n        applied to modified subtopic pages.\n    '
    topic = topic_fetchers.get_topic_by_id(topic_id)
    newly_created_subtopic_ids: List[int] = []
    existing_subtopic_page_ids_to_be_modified: List[int] = []
    deleted_subtopic_ids: List[int] = []
    modified_subtopic_pages_list: List[Optional[subtopic_page_domain.SubtopicPage]] = []
    modified_subtopic_pages: Dict[str, subtopic_page_domain.SubtopicPage] = {}
    modified_subtopic_change_cmds: Dict[str, List[subtopic_page_domain.SubtopicPageChange]] = collections.defaultdict(list)
    for change in change_list:
        if change.cmd == subtopic_page_domain.CMD_UPDATE_SUBTOPIC_PAGE_PROPERTY:
            update_subtopic_page_property_cmd = cast(subtopic_page_domain.UpdateSubtopicPagePropertyCmd, change)
            if update_subtopic_page_property_cmd.subtopic_id < topic.next_subtopic_id:
                existing_subtopic_page_ids_to_be_modified.append(update_subtopic_page_property_cmd.subtopic_id)
                subtopic_page_id = subtopic_page_domain.SubtopicPage.get_subtopic_page_id(topic_id, update_subtopic_page_property_cmd.subtopic_id)
                modified_subtopic_change_cmds[subtopic_page_id].append(update_subtopic_page_property_cmd)
    modified_subtopic_pages_list = subtopic_page_services.get_subtopic_pages_with_ids(topic_id, existing_subtopic_page_ids_to_be_modified)
    for subtopic_page in modified_subtopic_pages_list:
        assert subtopic_page is not None
        modified_subtopic_pages[subtopic_page.id] = subtopic_page
    try:
        for change in change_list:
            if change.cmd == topic_domain.CMD_ADD_SUBTOPIC:
                add_subtopic_cmd = cast(topic_domain.AddSubtopicCmd, change)
                topic.add_subtopic(add_subtopic_cmd.subtopic_id, add_subtopic_cmd.title, add_subtopic_cmd.url_fragment)
                subtopic_page_id = subtopic_page_domain.SubtopicPage.get_subtopic_page_id(topic_id, add_subtopic_cmd.subtopic_id)
                modified_subtopic_pages[subtopic_page_id] = subtopic_page_domain.SubtopicPage.create_default_subtopic_page(add_subtopic_cmd.subtopic_id, topic_id)
                modified_subtopic_change_cmds[subtopic_page_id].append(subtopic_page_domain.SubtopicPageChange({'cmd': 'create_new', 'topic_id': topic_id, 'subtopic_id': add_subtopic_cmd.subtopic_id}))
                newly_created_subtopic_ids.append(add_subtopic_cmd.subtopic_id)
            elif change.cmd == topic_domain.CMD_DELETE_SUBTOPIC:
                delete_subtopic_cmd = cast(topic_domain.DeleteSubtopicCmd, change)
                topic.delete_subtopic(delete_subtopic_cmd.subtopic_id)
                if delete_subtopic_cmd.subtopic_id in newly_created_subtopic_ids:
                    raise Exception('The incoming changelist had simultaneous creation and deletion of subtopics.')
                deleted_subtopic_ids.append(delete_subtopic_cmd.subtopic_id)
            elif change.cmd == topic_domain.CMD_ADD_CANONICAL_STORY:
                add_canonical_story_cmd = cast(topic_domain.AddCanonicalStoryCmd, change)
                topic.add_canonical_story(add_canonical_story_cmd.story_id)
            elif change.cmd == topic_domain.CMD_DELETE_CANONICAL_STORY:
                delete_canonical_story_cmd = cast(topic_domain.DeleteCanonicalStoryCmd, change)
                topic.delete_canonical_story(delete_canonical_story_cmd.story_id)
            elif change.cmd == topic_domain.CMD_REARRANGE_CANONICAL_STORY:
                rearrange_canonical_story_cmd = cast(topic_domain.RearrangeCanonicalStoryCmd, change)
                topic.rearrange_canonical_story(rearrange_canonical_story_cmd.from_index, rearrange_canonical_story_cmd.to_index)
            elif change.cmd == topic_domain.CMD_ADD_ADDITIONAL_STORY:
                add_additional_story_cmd = cast(topic_domain.AddAdditionalStoryCmd, change)
                topic.add_additional_story(add_additional_story_cmd.story_id)
            elif change.cmd == topic_domain.CMD_DELETE_ADDITIONAL_STORY:
                delete_additional_story_cmd = cast(topic_domain.DeleteAdditionalStoryCmd, change)
                topic.delete_additional_story(delete_additional_story_cmd.story_id)
            elif change.cmd == topic_domain.CMD_ADD_UNCATEGORIZED_SKILL_ID:
                add_uncategorized_skill_id_cmd = cast(topic_domain.AddUncategorizedSkillIdCmd, change)
                topic.add_uncategorized_skill_id(add_uncategorized_skill_id_cmd.new_uncategorized_skill_id)
            elif change.cmd == topic_domain.CMD_REMOVE_UNCATEGORIZED_SKILL_ID:
                remove_uncategorized_skill_id_cmd = cast(topic_domain.RemoveUncategorizedSkillIdCmd, change)
                topic.remove_uncategorized_skill_id(remove_uncategorized_skill_id_cmd.uncategorized_skill_id)
            elif change.cmd == topic_domain.CMD_MOVE_SKILL_ID_TO_SUBTOPIC:
                move_skill_id_to_subtopic_cmd = cast(topic_domain.MoveSkillIdToSubtopicCmd, change)
                topic.move_skill_id_to_subtopic(move_skill_id_to_subtopic_cmd.old_subtopic_id, move_skill_id_to_subtopic_cmd.new_subtopic_id, move_skill_id_to_subtopic_cmd.skill_id)
            elif change.cmd == topic_domain.CMD_REARRANGE_SKILL_IN_SUBTOPIC:
                rearrange_skill_in_subtopic_cmd = cast(topic_domain.RearrangeSkillInSubtopicCmd, change)
                topic.rearrange_skill_in_subtopic(rearrange_skill_in_subtopic_cmd.subtopic_id, rearrange_skill_in_subtopic_cmd.from_index, rearrange_skill_in_subtopic_cmd.to_index)
            elif change.cmd == topic_domain.CMD_REARRANGE_SUBTOPIC:
                rearrange_subtopic_cmd = cast(topic_domain.RearrangeSubtopicCmd, change)
                topic.rearrange_subtopic(rearrange_subtopic_cmd.from_index, rearrange_subtopic_cmd.to_index)
            elif change.cmd == topic_domain.CMD_REMOVE_SKILL_ID_FROM_SUBTOPIC:
                remove_skill_id_from_subtopic_cmd = cast(topic_domain.RemoveSkillIdFromSubtopicCmd, change)
                topic.remove_skill_id_from_subtopic(remove_skill_id_from_subtopic_cmd.subtopic_id, remove_skill_id_from_subtopic_cmd.skill_id)
            elif change.cmd == topic_domain.CMD_UPDATE_TOPIC_PROPERTY:
                if change.property_name == topic_domain.TOPIC_PROPERTY_NAME:
                    update_topic_name_cmd = cast(topic_domain.UpdateTopicPropertyNameCmd, change)
                    topic.update_name(update_topic_name_cmd.new_value)
                elif change.property_name == topic_domain.TOPIC_PROPERTY_ABBREVIATED_NAME:
                    update_abbreviated_name_cmd = cast(topic_domain.UpdateTopicPropertyAbbreviatedNameCmd, change)
                    topic.update_abbreviated_name(update_abbreviated_name_cmd.new_value)
                elif change.property_name == topic_domain.TOPIC_PROPERTY_URL_FRAGMENT:
                    update_url_fragment_cmd = cast(topic_domain.UpdateTopicPropertyUrlFragmentCmd, change)
                    topic.update_url_fragment(update_url_fragment_cmd.new_value)
                elif change.property_name == topic_domain.TOPIC_PROPERTY_DESCRIPTION:
                    update_topic_description_cmd = cast(topic_domain.UpdateTopicPropertyDescriptionCmd, change)
                    topic.update_description(update_topic_description_cmd.new_value)
                elif change.property_name == topic_domain.TOPIC_PROPERTY_LANGUAGE_CODE:
                    update_topic_language_code_cmd = cast(topic_domain.UpdateTopicPropertyLanguageCodeCmd, change)
                    topic.update_language_code(update_topic_language_code_cmd.new_value)
                elif change.property_name == topic_domain.TOPIC_PROPERTY_THUMBNAIL_FILENAME:
                    update_topic_thumbnail_filename_cmd = cast(topic_domain.UpdateTopicPropertyThumbnailFilenameCmd, change)
                    update_thumbnail_filename(topic, update_topic_thumbnail_filename_cmd.new_value)
                elif change.property_name == topic_domain.TOPIC_PROPERTY_THUMBNAIL_BG_COLOR:
                    update_topic_thumbnail_bg_color_cmd = cast(topic_domain.UpdateTopicPropertyThumbnailBGColorCmd, change)
                    topic.update_thumbnail_bg_color(update_topic_thumbnail_bg_color_cmd.new_value)
                elif change.property_name == topic_domain.TOPIC_PROPERTY_META_TAG_CONTENT:
                    update_topic_meta_tag_content_cmd = cast(topic_domain.UpdateTopicPropertyMetaTagContentCmd, change)
                    topic.update_meta_tag_content(update_topic_meta_tag_content_cmd.new_value)
                elif change.property_name == topic_domain.TOPIC_PROPERTY_PRACTICE_TAB_IS_DISPLAYED:
                    update_practice_tab_is_displayed_cmd = cast(topic_domain.UpdateTopicPropertyPracticeTabIsDisplayedCmd, change)
                    topic.update_practice_tab_is_displayed(update_practice_tab_is_displayed_cmd.new_value)
                elif change.property_name == topic_domain.TOPIC_PROPERTY_PAGE_TITLE_FRAGMENT_FOR_WEB:
                    update_title_fragment_for_web_cmd = cast(topic_domain.UpdateTopicPropertyTitleFragmentForWebCmd, change)
                    topic.update_page_title_fragment_for_web(update_title_fragment_for_web_cmd.new_value)
                elif change.property_name == topic_domain.TOPIC_PROPERTY_SKILL_IDS_FOR_DIAGNOSTIC_TEST:
                    update_skill_ids_for_diagnostic_test_cmd = cast(topic_domain.UpdateTopicPropertySkillIdsForDiagnosticTestCmd, change)
                    topic.update_skill_ids_for_diagnostic_test(update_skill_ids_for_diagnostic_test_cmd.new_value)
            elif change.cmd == subtopic_page_domain.CMD_UPDATE_SUBTOPIC_PAGE_PROPERTY:
                assert isinstance(change.subtopic_id, int)
                subtopic_page_id = subtopic_page_domain.SubtopicPage.get_subtopic_page_id(topic_id, change.subtopic_id)
                if modified_subtopic_pages[subtopic_page_id] is None or change.subtopic_id in deleted_subtopic_ids:
                    raise Exception("The subtopic with id %s doesn't exist" % change.subtopic_id)
                if change.property_name == subtopic_page_domain.SUBTOPIC_PAGE_PROPERTY_PAGE_CONTENTS_HTML:
                    update_subtopic_page_contents_html_cmd = cast(subtopic_page_domain.UpdateSubtopicPagePropertyPageContentsHtmlCmd, change)
                    page_contents = state_domain.SubtitledHtml.from_dict(update_subtopic_page_contents_html_cmd.new_value)
                    page_contents.validate()
                    modified_subtopic_pages[subtopic_page_id].update_page_contents_html(page_contents)
                elif change.property_name == subtopic_page_domain.SUBTOPIC_PAGE_PROPERTY_PAGE_CONTENTS_AUDIO:
                    update_subtopic_page_contents_audio_cmd = cast(subtopic_page_domain.UpdateSubtopicPagePropertyPageContentsAudioCmd, change)
                    modified_subtopic_pages[subtopic_page_id].update_page_contents_audio(state_domain.RecordedVoiceovers.from_dict(update_subtopic_page_contents_audio_cmd.new_value))
            elif change.cmd == topic_domain.CMD_UPDATE_SUBTOPIC_PROPERTY:
                update_subtopic_property_cmd = cast(topic_domain.UpdateSubtopicPropertyCmd, change)
                if update_subtopic_property_cmd.property_name == topic_domain.SUBTOPIC_PROPERTY_TITLE:
                    topic.update_subtopic_title(update_subtopic_property_cmd.subtopic_id, update_subtopic_property_cmd.new_value)
                if update_subtopic_property_cmd.property_name == topic_domain.SUBTOPIC_PROPERTY_THUMBNAIL_FILENAME:
                    update_subtopic_thumbnail_filename(topic, update_subtopic_property_cmd.subtopic_id, update_subtopic_property_cmd.new_value)
                if update_subtopic_property_cmd.property_name == topic_domain.SUBTOPIC_PROPERTY_THUMBNAIL_BG_COLOR:
                    topic.update_subtopic_thumbnail_bg_color(update_subtopic_property_cmd.subtopic_id, update_subtopic_property_cmd.new_value)
                if update_subtopic_property_cmd.property_name == topic_domain.SUBTOPIC_PROPERTY_URL_FRAGMENT:
                    topic.update_subtopic_url_fragment(update_subtopic_property_cmd.subtopic_id, update_subtopic_property_cmd.new_value)
            elif change.cmd == topic_domain.CMD_MIGRATE_SUBTOPIC_SCHEMA_TO_LATEST_VERSION:
                continue
        return (topic, modified_subtopic_pages, deleted_subtopic_ids, newly_created_subtopic_ids, modified_subtopic_change_cmds)
    except Exception as e:
        logging.error('%s %s %s %s' % (e.__class__.__name__, e, topic_id, change_list))
        raise e

def _save_topic(committer_id: str, topic: topic_domain.Topic, commit_message: Optional[str], change_list: Sequence[change_domain.BaseChange]) -> None:
    if False:
        while True:
            i = 10
    'Validates a topic and commits it to persistent storage. If\n    successful, increments the version number of the incoming topic domain\n    object by 1.\n\n    Args:\n        committer_id: str. ID of the given committer.\n        topic: Topic. The topic domain object to be saved.\n        commit_message: str|None. The commit description message, for\n            unpublished topics, it may be equal to None.\n        change_list: list(TopicChange). List of changes applied to a topic.\n\n    Raises:\n        Exception. Received invalid change list.\n        Exception. The topic model and the incoming topic domain\n            object have different version numbers.\n    '
    if not change_list:
        raise Exception('Unexpected error: received an invalid change list when trying to save topic %s: %s' % (topic.id, change_list))
    topic_rights = topic_fetchers.get_topic_rights(topic.id, strict=True)
    topic.validate(strict=topic_rights.topic_is_published)
    topic_model = topic_models.TopicModel.get(topic.id, strict=True)
    if topic.version > topic_model.version:
        raise Exception('Unexpected error: trying to update version %s of topic from version %s. Please reload the page and try again.' % (topic_model.version, topic.version))
    if topic.version < topic_model.version:
        raise Exception('Trying to update version %s of topic from version %s, which is too old. Please reload the page and try again.' % (topic_model.version, topic.version))
    topic_model_to_commit = populate_topic_model_fields(topic_model, topic)
    change_dicts = [change.to_dict() for change in change_list]
    topic_model_to_commit.commit(committer_id, commit_message, change_dicts)
    caching_services.delete_multi(caching_services.CACHE_NAMESPACE_TOPIC, None, [topic.id])
    topic.version += 1

def update_topic_and_subtopic_pages(committer_id: str, topic_id: str, change_list: Sequence[change_domain.BaseChange], commit_message: Optional[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Updates a topic and its subtopic pages. Commits changes.\n\n    Args:\n        committer_id: str. The id of the user who is performing the update\n            action.\n        topic_id: str. The topic id.\n        change_list: list(TopicChange and SubtopicPageChange). These changes are\n            applied in sequence to produce the resulting topic.\n        commit_message: str or None. A description of changes made to the\n            topic.\n\n    Raises:\n        ValueError. Current user does not have enough rights to edit a topic.\n    '
    topic_rights = topic_fetchers.get_topic_rights(topic_id, strict=True)
    if topic_rights.topic_is_published and (not commit_message):
        raise ValueError('Expected a commit message, received none.')
    old_topic = topic_fetchers.get_topic_by_id(topic_id)
    (updated_topic, updated_subtopic_pages_dict, deleted_subtopic_ids, newly_created_subtopic_ids, updated_subtopic_pages_change_cmds_dict) = apply_change_list(topic_id, change_list)
    if old_topic.url_fragment != updated_topic.url_fragment and does_topic_with_url_fragment_exist(updated_topic.url_fragment):
        raise utils.ValidationError("Topic with URL Fragment '%s' already exists" % updated_topic.url_fragment)
    if old_topic.name != updated_topic.name and does_topic_with_name_exist(updated_topic.name):
        raise utils.ValidationError("Topic with name '%s' already exists" % updated_topic.name)
    _save_topic(committer_id, updated_topic, commit_message, change_list)
    for subtopic_id in deleted_subtopic_ids:
        if subtopic_id not in newly_created_subtopic_ids:
            subtopic_page_services.delete_subtopic_page(committer_id, topic_id, subtopic_id)
    for (subtopic_page_id, subtopic_page) in updated_subtopic_pages_dict.items():
        subtopic_page_change_list = updated_subtopic_pages_change_cmds_dict[subtopic_page_id]
        subtopic_id = subtopic_page.get_subtopic_id_from_subtopic_page_id()
        if subtopic_id not in deleted_subtopic_ids:
            subtopic_page_services.save_subtopic_page(committer_id, subtopic_page, commit_message, subtopic_page_change_list)
    generate_topic_summary(topic_id)
    if old_topic.name != updated_topic.name:
        opportunity_services.update_opportunities_with_new_topic_name(updated_topic.id, updated_topic.name)

def delete_uncategorized_skill(user_id: str, topic_id: str, uncategorized_skill_id: str) -> None:
    if False:
        while True:
            i = 10
    'Removes skill with given id from the topic.\n\n    Args:\n        user_id: str. The id of the user who is performing the action.\n        topic_id: str. The id of the topic from which to remove the skill.\n        uncategorized_skill_id: str. The uncategorized skill to remove from the\n            topic.\n    '
    change_list = [topic_domain.TopicChange({'cmd': 'remove_uncategorized_skill_id', 'uncategorized_skill_id': uncategorized_skill_id})]
    update_topic_and_subtopic_pages(user_id, topic_id, change_list, 'Removed %s from uncategorized skill ids' % uncategorized_skill_id)

def add_uncategorized_skill(user_id: str, topic_id: str, uncategorized_skill_id: str) -> None:
    if False:
        print('Hello World!')
    'Adds a skill with given id to the topic.\n\n    Args:\n        user_id: str. The id of the user who is performing the action.\n        topic_id: str. The id of the topic to which the skill is to be added.\n        uncategorized_skill_id: str. The id of the uncategorized skill to add\n            to the topic.\n    '
    change_list = [topic_domain.TopicChange({'cmd': 'add_uncategorized_skill_id', 'new_uncategorized_skill_id': uncategorized_skill_id})]
    update_topic_and_subtopic_pages(user_id, topic_id, change_list, 'Added %s to uncategorized skill ids' % uncategorized_skill_id)

def publish_story(topic_id: str, story_id: str, committer_id: str) -> None:
    if False:
        return 10
    'Marks the given story as published.\n\n    Args:\n        topic_id: str. The id of the topic.\n        story_id: str. The id of the given story.\n        committer_id: str. ID of the committer.\n\n    Raises:\n        Exception. The given story does not exist.\n        Exception. The story is already published.\n        Exception. The user does not have enough rights to publish the story.\n    '

    def _are_nodes_valid_for_publishing(story_nodes: List[story_domain.StoryNode]) -> None:
        if False:
            i = 10
            return i + 15
        "Validates the story nodes before publishing.\n\n        Args:\n            story_nodes: list(dict(str, *)). The list of story nodes dicts.\n\n        Raises:\n            Exception. The story node doesn't contain any exploration id or the\n                exploration id is invalid or isn't published yet.\n        "
        exploration_id_list = []
        for node in story_nodes:
            if not node.exploration_id:
                raise Exception('Story node with id %s does not contain an exploration id.' % node.id)
            exploration_id_list.append(node.exploration_id)
        story_services.validate_explorations_for_story(exploration_id_list, True)
    topic = topic_fetchers.get_topic_by_id(topic_id, strict=True)
    user = user_services.get_user_actions_info(committer_id)
    if role_services.ACTION_CHANGE_STORY_STATUS not in user.actions:
        raise Exception('The user does not have enough rights to publish the story.')
    story = story_fetchers.get_story_by_id(story_id, strict=False)
    if story is None:
        raise Exception("A story with the given ID doesn't exist")
    for node in story.story_contents.nodes:
        if node.id == story.story_contents.initial_node_id:
            _are_nodes_valid_for_publishing([node])
    topic.publish_story(story_id)
    change_list = [topic_domain.TopicChange({'cmd': topic_domain.CMD_PUBLISH_STORY, 'story_id': story_id})]
    _save_topic(committer_id, topic, 'Published story with id %s' % story_id, change_list)
    generate_topic_summary(topic.id)
    linked_exp_ids = story.story_contents.get_all_linked_exp_ids()
    opportunity_services.add_new_exploration_opportunities(story_id, linked_exp_ids)

def unpublish_story(topic_id: str, story_id: str, committer_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Marks the given story as unpublished.\n\n    Args:\n        topic_id: str. The id of the topic.\n        story_id: str. The id of the given story.\n        committer_id: str. ID of the committer.\n\n    Raises:\n        Exception. The given story does not exist.\n        Exception. The story is already unpublished.\n        Exception. The user does not have enough rights to unpublish the story.\n    '
    user = user_services.get_user_actions_info(committer_id)
    if role_services.ACTION_CHANGE_STORY_STATUS not in user.actions:
        raise Exception('The user does not have enough rights to unpublish the story.')
    topic = topic_fetchers.get_topic_by_id(topic_id, strict=False)
    if topic is None:
        raise Exception("A topic with the given ID doesn't exist")
    story = story_fetchers.get_story_by_id(story_id, strict=False)
    if story is None:
        raise Exception("A story with the given ID doesn't exist")
    topic.unpublish_story(story_id)
    change_list = [topic_domain.TopicChange({'cmd': topic_domain.CMD_UNPUBLISH_STORY, 'story_id': story_id})]
    _save_topic(committer_id, topic, 'Unpublished story with id %s' % story_id, change_list)
    generate_topic_summary(topic.id)
    exp_ids = story.story_contents.get_all_linked_exp_ids()
    opportunity_services.delete_exploration_opportunities(exp_ids)
    suggestion_services.auto_reject_translation_suggestions_for_exp_ids(exp_ids)

def delete_canonical_story(user_id: str, topic_id: str, story_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Removes story with given id from the topic.\n\n    NOTE TO DEVELOPERS: Presently, this function only removes story_reference\n    from canonical_story_references list.\n\n    Args:\n        user_id: str. The id of the user who is performing the action.\n        topic_id: str. The id of the topic from which to remove the story.\n        story_id: str. The story to remove from the topic.\n    '
    change_list = [topic_domain.TopicChange({'cmd': topic_domain.CMD_DELETE_CANONICAL_STORY, 'story_id': story_id})]
    update_topic_and_subtopic_pages(user_id, topic_id, change_list, 'Removed %s from canonical story ids' % story_id)

def add_canonical_story(user_id: str, topic_id: str, story_id: str) -> None:
    if False:
        print('Hello World!')
    'Adds a story to the canonical story reference list of a topic.\n\n    Args:\n        user_id: str. The id of the user who is performing the action.\n        topic_id: str. The id of the topic to which the story is to be added.\n        story_id: str. The story to add to the topic.\n    '
    change_list = [topic_domain.TopicChange({'cmd': topic_domain.CMD_ADD_CANONICAL_STORY, 'story_id': story_id})]
    update_topic_and_subtopic_pages(user_id, topic_id, change_list, 'Added %s to canonical story ids' % story_id)

def delete_additional_story(user_id: str, topic_id: str, story_id: str) -> None:
    if False:
        return 10
    'Removes story with given id from the topic.\n\n    NOTE TO DEVELOPERS: Presently, this function only removes story_reference\n    from additional_story_references list.\n\n    Args:\n        user_id: str. The id of the user who is performing the action.\n        topic_id: str. The id of the topic from which to remove the story.\n        story_id: str. The story to remove from the topic.\n    '
    change_list = [topic_domain.TopicChange({'cmd': topic_domain.CMD_DELETE_ADDITIONAL_STORY, 'story_id': story_id})]
    update_topic_and_subtopic_pages(user_id, topic_id, change_list, 'Removed %s from additional story ids' % story_id)

def add_additional_story(user_id: str, topic_id: str, story_id: str) -> None:
    if False:
        while True:
            i = 10
    'Adds a story to the additional story reference list of a topic.\n\n    Args:\n        user_id: str. The id of the user who is performing the action.\n        topic_id: str. The id of the topic to which the story is to be added.\n        story_id: str. The story to add to the topic.\n    '
    change_list = [topic_domain.TopicChange({'cmd': topic_domain.CMD_ADD_ADDITIONAL_STORY, 'story_id': story_id})]
    update_topic_and_subtopic_pages(user_id, topic_id, change_list, 'Added %s to additional story ids' % story_id)

def delete_topic(committer_id: str, topic_id: str, force_deletion: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    'Deletes the topic with the given topic_id.\n\n    Args:\n        committer_id: str. ID of the committer.\n        topic_id: str. ID of the topic to be deleted.\n        force_deletion: bool. If true, the topic and its history are fully\n            deleted and are unrecoverable. Otherwise, the topic and all\n            its history are marked as deleted, but the corresponding models are\n            still retained in the datastore. This last option is the preferred\n            one.\n\n    Raises:\n        ValueError. User does not have enough rights to delete a topic.\n    '
    topic_rights_model = topic_models.TopicRightsModel.get(topic_id)
    topic_rights_model.delete(committer_id, feconf.COMMIT_MESSAGE_TOPIC_DELETED, force_deletion=force_deletion)
    delete_topic_summary(topic_id)
    topic_model = topic_models.TopicModel.get(topic_id)
    for subtopic in topic_model.subtopics:
        subtopic_page_services.delete_subtopic_page(committer_id, topic_id, subtopic['id'])
    all_story_references = topic_model.canonical_story_references + topic_model.additional_story_references
    for story_reference in all_story_references:
        story_services.delete_story(committer_id, story_reference['story_id'], force_deletion=force_deletion)
    topic_model.delete(committer_id, feconf.COMMIT_MESSAGE_TOPIC_DELETED, force_deletion=force_deletion)
    feedback_services.delete_threads_for_multiple_entities(feconf.ENTITY_TYPE_TOPIC, [topic_id])
    caching_services.delete_multi(caching_services.CACHE_NAMESPACE_TOPIC, None, [topic_id])
    opportunity_services.delete_exploration_opportunities_corresponding_to_topic(topic_id)

def delete_topic_summary(topic_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Delete a topic summary model.\n\n    Args:\n        topic_id: str. ID of the topic whose topic summary is to\n            be deleted.\n    '
    topic_models.TopicSummaryModel.get(topic_id).delete()

def update_story_and_topic_summary(committer_id: str, story_id: str, change_list: List[story_domain.StoryChange], commit_message: str, topic_id: str) -> None:
    if False:
        while True:
            i = 10
    'Updates a story. Commits changes. Then generates a new\n    topic summary.\n\n    Args:\n        committer_id: str. The id of the user who is performing the update\n            action.\n        story_id: str. The story id.\n        change_list: list(StoryChange). These changes are applied in sequence to\n            produce the resulting story.\n        commit_message: str. A description of changes made to the\n            story.\n        topic_id: str. The id of the topic to which the story is belongs.\n    '
    story_services.update_story(committer_id, story_id, change_list, commit_message)
    generate_topic_summary(topic_id)

def generate_topic_summary(topic_id: str) -> None:
    if False:
        print('Hello World!')
    'Creates and stores a summary of the given topic.\n\n    Args:\n        topic_id: str. ID of the topic.\n    '
    topic = topic_fetchers.get_topic_by_id(topic_id)
    topic_summary = compute_summary_of_topic(topic)
    save_topic_summary(topic_summary)

def compute_summary_of_topic(topic: topic_domain.Topic) -> topic_domain.TopicSummary:
    if False:
        for i in range(10):
            print('nop')
    'Create a TopicSummary domain object for a given Topic domain\n    object and return it.\n\n    Args:\n        topic: Topic. The topic object for which the summary is to be computed.\n\n    Returns:\n        TopicSummary. The computed summary for the given topic.\n\n    Raises:\n        Exception. No data available for when the topic was last updated.\n    '
    canonical_story_count = 0
    additional_story_count = 0
    published_node_count = 0
    for reference in topic.canonical_story_references:
        if reference.story_is_published:
            canonical_story_count += 1
            story_summary = story_fetchers.get_story_summary_by_id(reference.story_id)
            published_node_count += len(story_summary.node_titles)
    for reference in topic.additional_story_references:
        if reference.story_is_published:
            additional_story_count += 1
    topic_model_canonical_story_count = canonical_story_count
    topic_model_additional_story_count = additional_story_count
    total_model_published_node_count = published_node_count
    topic_model_uncategorized_skill_count = len(topic.uncategorized_skill_ids)
    topic_model_subtopic_count = len(topic.subtopics)
    total_skill_count = topic_model_uncategorized_skill_count
    for subtopic in topic.subtopics:
        total_skill_count = total_skill_count + len(subtopic.skill_ids)
    if topic.created_on is None or topic.last_updated is None:
        raise Exception('No data available for when the topic was last updated.')
    topic_summary = topic_domain.TopicSummary(topic.id, topic.name, topic.canonical_name, topic.language_code, topic.description, topic.version, topic_model_canonical_story_count, topic_model_additional_story_count, topic_model_uncategorized_skill_count, topic_model_subtopic_count, total_skill_count, total_model_published_node_count, topic.thumbnail_filename, topic.thumbnail_bg_color, topic.url_fragment, topic.created_on, topic.last_updated)
    return topic_summary

def save_topic_summary(topic_summary: topic_domain.TopicSummary) -> None:
    if False:
        while True:
            i = 10
    'Save a topic summary domain object as a TopicSummaryModel\n    entity in the datastore.\n\n    Args:\n        topic_summary: TopicSummary. The topic summary object to be saved\n            in the datastore.\n    '
    existing_topic_summary_model = topic_models.TopicSummaryModel.get_by_id(topic_summary.id)
    topic_summary_model = populate_topic_summary_model_fields(existing_topic_summary_model, topic_summary)
    topic_summary_model.update_timestamps()
    topic_summary_model.put()

def publish_topic(topic_id: str, committer_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Marks the given topic as published.\n\n    Args:\n        topic_id: str. The id of the given topic.\n        committer_id: str. ID of the committer.\n\n    Raises:\n        Exception. The given topic does not exist.\n        Exception. The topic is already published.\n        Exception. The user does not have enough rights to publish the topic.\n    '
    topic_rights = topic_fetchers.get_topic_rights(topic_id, strict=False)
    if topic_rights is None:
        raise Exception('The given topic does not exist')
    topic = topic_fetchers.get_topic_by_id(topic_id)
    topic.validate(strict=True)
    user = user_services.get_user_actions_info(committer_id)
    if role_services.ACTION_CHANGE_TOPIC_STATUS not in user.actions:
        raise Exception('The user does not have enough rights to publish the topic.')
    if topic_rights.topic_is_published:
        raise Exception('The topic is already published.')
    topic_rights.topic_is_published = True
    commit_cmds = [topic_domain.TopicRightsChange({'cmd': topic_domain.CMD_PUBLISH_TOPIC})]
    save_topic_rights(topic_rights, committer_id, 'Published the topic', commit_cmds)

def unpublish_topic(topic_id: str, committer_id: str) -> None:
    if False:
        return 10
    'Marks the given topic as unpublished.\n\n    Args:\n        topic_id: str. The id of the given topic.\n        committer_id: str. ID of the committer.\n\n    Raises:\n        Exception. The given topic does not exist.\n        Exception. The topic is already unpublished.\n        Exception. The user does not have enough rights to unpublish the topic.\n    '
    topic_rights = topic_fetchers.get_topic_rights(topic_id, strict=False)
    if topic_rights is None:
        raise Exception('The given topic does not exist')
    user = user_services.get_user_actions_info(committer_id)
    if role_services.ACTION_CHANGE_TOPIC_STATUS not in user.actions:
        raise Exception('The user does not have enough rights to unpublish the topic.')
    if not topic_rights.topic_is_published:
        raise Exception('The topic is already unpublished.')
    topic_rights.topic_is_published = False
    commit_cmds = [topic_domain.TopicRightsChange({'cmd': topic_domain.CMD_UNPUBLISH_TOPIC})]
    save_topic_rights(topic_rights, committer_id, 'Unpublished the topic', commit_cmds)

def save_topic_rights(topic_rights: topic_domain.TopicRights, committer_id: str, commit_message: str, commit_cmds: List[topic_domain.TopicRightsChange]) -> None:
    if False:
        i = 10
        return i + 15
    'Saves a TopicRights domain object to the datastore.\n\n    Args:\n        topic_rights: TopicRights. The rights object for the given\n            topic.\n        committer_id: str. ID of the committer.\n        commit_message: str. Descriptive message for the commit.\n        commit_cmds: list(TopicRightsChange). A list of commands describing\n            what kind of commit was done.\n    '
    model = topic_models.TopicRightsModel.get(topic_rights.id, strict=True)
    model.manager_ids = topic_rights.manager_ids
    model.topic_is_published = topic_rights.topic_is_published
    commit_cmd_dicts = [commit_cmd.to_dict() for commit_cmd in commit_cmds]
    model.commit(committer_id, commit_message, commit_cmd_dicts)

def create_new_topic_rights(topic_id: str, committer_id: str) -> None:
    if False:
        while True:
            i = 10
    'Creates a new topic rights object and saves it to the datastore.\n\n    Args:\n        topic_id: str. ID of the topic.\n        committer_id: str. ID of the committer.\n    '
    topic_rights = topic_domain.TopicRights(topic_id, [], False)
    commit_cmds = [{'cmd': topic_domain.CMD_CREATE_NEW}]
    topic_models.TopicRightsModel(id=topic_rights.id, manager_ids=topic_rights.manager_ids, topic_is_published=topic_rights.topic_is_published).commit(committer_id, 'Created new topic rights', commit_cmds)

def filter_published_topic_ids(topic_ids: List[str]) -> List[str]:
    if False:
        while True:
            i = 10
    'Given list of topic IDs, returns the IDs of all topics that are published\n    in that list.\n\n    Args:\n        topic_ids: list(str). The list of topic ids.\n\n    Returns:\n        list(str). The topic IDs in the passed in list corresponding to\n        published topics.\n    '
    topic_rights_models = topic_models.TopicRightsModel.get_multi(topic_ids)
    published_topic_ids = []
    for (ind, model) in enumerate(topic_rights_models):
        if model is None:
            continue
        rights = topic_fetchers.get_topic_rights_from_model(model)
        if rights.topic_is_published:
            published_topic_ids.append(topic_ids[ind])
    return published_topic_ids

def check_can_edit_topic(user: user_domain.UserActionsInfo, topic_rights: Optional[topic_domain.TopicRights]) -> bool:
    if False:
        print('Hello World!')
    'Checks whether the user can edit the given topic.\n\n    Args:\n        user: UserActionsInfo. Object having user_id, role and actions for\n            given user.\n        topic_rights: TopicRights or None. Rights object for the given topic.\n\n    Returns:\n        bool. Whether the given user can edit the given topic.\n    '
    if topic_rights is None:
        return False
    if role_services.ACTION_EDIT_ANY_TOPIC in user.actions:
        return True
    if role_services.ACTION_EDIT_OWNED_TOPIC not in user.actions:
        return False
    if user.user_id and topic_rights.is_manager(user.user_id):
        return True
    return False

def deassign_user_from_all_topics(committer: user_domain.UserActionsInfo, user_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Deassigns given user from all topics assigned to them.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the user\n            who is performing the action.\n        user_id: str. The ID of the user.\n\n    Raises:\n        Exception. The committer does not have rights to modify a role.\n        Exception. Guest users are not allowed to deassing users from\n            all topics.\n    '
    topic_rights_list = topic_fetchers.get_topic_rights_with_user(user_id)
    if committer.user_id is None:
        raise Exception('Guest users are not allowed to deassing users from all topics.')
    for topic_rights in topic_rights_list:
        topic_rights.manager_ids.remove(user_id)
        commit_cmds = [topic_domain.TopicRightsChange({'cmd': topic_domain.CMD_REMOVE_MANAGER_ROLE, 'removed_user_id': user_id})]
        save_topic_rights(topic_rights, committer.user_id, 'Removed all assigned topics from %s' % user_services.get_username(user_id), commit_cmds)

def deassign_manager_role_from_topic(committer: user_domain.UserActionsInfo, user_id: str, topic_id: str) -> None:
    if False:
        return 10
    'Deassigns given user from all topics assigned to them.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the user\n            who is performing the action.\n        user_id: str. The ID of the user.\n        topic_id: str. The ID of the topic.\n\n    Raises:\n        Exception. The committer does not have rights to modify a role.\n        Exception. Guest users are not allowed to deassing manager role\n            from topic.\n    '
    if committer.user_id is None:
        raise Exception('Guest users are not allowed to deassing manager role from topic.')
    topic_rights = topic_fetchers.get_topic_rights(topic_id)
    if user_id not in topic_rights.manager_ids:
        raise Exception('User does not have manager rights in topic.')
    topic_rights.manager_ids.remove(user_id)
    commit_cmds = [topic_domain.TopicRightsChange({'cmd': topic_domain.CMD_REMOVE_MANAGER_ROLE, 'removed_user_id': user_id})]
    save_topic_rights(topic_rights, committer.user_id, 'Removed all assigned topics from %s' % user_services.get_username(user_id), commit_cmds)

def assign_role(committer: user_domain.UserActionsInfo, assignee: user_domain.UserActionsInfo, new_role: str, topic_id: str) -> None:
    if False:
        i = 10
        return i + 15
    "Assigns a new role to the user.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the user\n            who is performing the action.\n        assignee: UserActionsInfo. UserActionsInfo object for the user\n            whose role is being changed.\n        new_role: str. The name of the new role. Possible values are:\n            ROLE_MANAGER.\n        topic_id: str. ID of the topic.\n\n    Raises:\n        Exception. The committer does not have rights to modify a role.\n        Exception. The assignee is already a manager for the topic.\n        Exception. The assignee doesn't have enough rights to become a manager.\n        Exception. The role is invalid.\n        Exception. Guest user is not allowed to assign roles to a user.\n        Exception. The role of the Guest user cannot be changed.\n    "
    committer_id = committer.user_id
    if committer_id is None:
        raise Exception('Guest user is not allowed to assign roles to a user.')
    topic_rights = topic_fetchers.get_topic_rights(topic_id)
    if role_services.ACTION_MODIFY_CORE_ROLES_FOR_ANY_ACTIVITY not in committer.actions:
        logging.error('User %s tried to allow user %s to be a %s of topic %s but was refused permission.' % (committer_id, assignee.user_id, new_role, topic_id))
        raise Exception('UnauthorizedUserException: Could not assign new role.')
    if assignee.user_id is None:
        raise Exception('Cannot change the role of the Guest user.')
    assignee_username = user_services.get_username(assignee.user_id)
    if role_services.ACTION_EDIT_OWNED_TOPIC not in assignee.actions:
        raise Exception("The assignee doesn't have enough rights to become a manager.")
    old_role = topic_domain.ROLE_NONE
    if topic_rights.is_manager(assignee.user_id):
        old_role = topic_domain.ROLE_MANAGER
    if new_role == topic_domain.ROLE_MANAGER:
        if topic_rights.is_manager(assignee.user_id):
            raise Exception('This user already is a manager for this topic')
        topic_rights.manager_ids.append(assignee.user_id)
    elif new_role == topic_domain.ROLE_NONE:
        if topic_rights.is_manager(assignee.user_id):
            topic_rights.manager_ids.remove(assignee.user_id)
        else:
            old_role = topic_domain.ROLE_NONE
    else:
        raise Exception('Invalid role: %s' % new_role)
    commit_message = rights_domain.ASSIGN_ROLE_COMMIT_MESSAGE_TEMPLATE % (assignee_username, old_role, new_role)
    commit_cmds = [topic_domain.TopicRightsChange({'cmd': topic_domain.CMD_CHANGE_ROLE, 'assignee_id': assignee.user_id, 'old_role': old_role, 'new_role': new_role})]
    save_topic_rights(topic_rights, committer_id, commit_message, commit_cmds)

def get_story_titles_in_topic(topic: topic_domain.Topic) -> List[str]:
    if False:
        print('Hello World!')
    'Returns titles of the stories present in the topic.\n\n    Args:\n        topic: Topic. The topic domain objects.\n\n    Returns:\n        list(str). The list of story titles in the topic.\n    '
    canonical_story_references = topic.canonical_story_references
    story_ids = [story.story_id for story in canonical_story_references]
    stories = story_fetchers.get_stories_by_ids(story_ids)
    story_titles = [story.title for story in stories if story is not None]
    return story_titles

def update_thumbnail_filename(topic: topic_domain.Topic, new_thumbnail_filename: str) -> None:
    if False:
        return 10
    'Updates the thumbnail filename and file size in a topic object.\n\n    Args:\n        topic: topic_domain.Topic. The topic domain object whose thumbnail\n            is to be updated.\n        new_thumbnail_filename: str. The updated thumbnail filename\n            for the topic.\n\n    Raises:\n        Exception. The thumbnail does not exist for expected topic in\n            the filesystem.\n    '
    fs = fs_services.GcsFileSystem(feconf.ENTITY_TYPE_TOPIC, topic.id)
    filepath = '%s/%s' % (constants.ASSET_TYPE_THUMBNAIL, new_thumbnail_filename)
    if fs.isfile(filepath):
        thumbnail_size_in_bytes = len(fs.get(filepath))
        topic.update_thumbnail_filename_and_size(new_thumbnail_filename, thumbnail_size_in_bytes)
    else:
        raise Exception('The thumbnail %s for topic with id %s does not exist in the filesystem.' % (new_thumbnail_filename, topic.id))

def update_subtopic_thumbnail_filename(topic: topic_domain.Topic, subtopic_id: int, new_thumbnail_filename: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Updates the thumbnail filename and file size in a subtopic.\n\n    Args:\n        topic: topic_domain.Topic. The topic domain object containing\n            the subtopic whose thumbnail is to be updated.\n        subtopic_id: int. The id of the subtopic to edit.\n        new_thumbnail_filename: str. The new thumbnail filename for the\n            subtopic.\n\n    Raises:\n        Exception. The thumbnail does not exist for expected topic in\n            the filesystem.\n    '
    fs = fs_services.GcsFileSystem(feconf.ENTITY_TYPE_TOPIC, topic.id)
    filepath = '%s/%s' % (constants.ASSET_TYPE_THUMBNAIL, new_thumbnail_filename)
    if fs.isfile(filepath):
        thumbnail_size_in_bytes = len(fs.get(filepath))
        topic.update_subtopic_thumbnail_filename_and_size(subtopic_id, new_thumbnail_filename, thumbnail_size_in_bytes)
    else:
        raise Exception('The thumbnail %s for subtopic with topic_id %s does not exist in the filesystem.' % (new_thumbnail_filename, topic.id))

def get_topic_id_to_diagnostic_test_skill_ids(topic_ids: List[str]) -> Dict[str, List[str]]:
    if False:
        while True:
            i = 10
    'Returns a dict with topic ID as key and a list of diagnostic test\n    skill IDs as value.\n\n    Args:\n        topic_ids: List(str). A list of topic IDs.\n\n    Raises:\n        Exception. The topic models for some of the given topic IDs do not\n            exist.\n\n    Returns:\n        dict(str, list(str)). A dict with topic ID as key and a list of\n        diagnostic test skill IDs as value.\n    '
    topic_id_to_diagnostic_test_skill_ids = {}
    topics = topic_fetchers.get_topics_by_ids(topic_ids)
    for topic in topics:
        if topic is None:
            continue
        topic_id_to_diagnostic_test_skill_ids[topic.id] = topic.skill_ids_for_diagnostic_test
    correct_topic_ids = list(topic_id_to_diagnostic_test_skill_ids.keys())
    incorrect_topic_ids = [topic_id for topic_id in topic_ids if topic_id not in correct_topic_ids]
    if incorrect_topic_ids:
        error_msg = 'No corresponding topic models exist for these topic IDs: %s.' % ', '.join(incorrect_topic_ids)
        raise Exception(error_msg)
    return topic_id_to_diagnostic_test_skill_ids

def populate_topic_model_fields(topic_model: topic_models.TopicModel, topic: topic_domain.Topic) -> topic_models.TopicModel:
    if False:
        for i in range(10):
            print('nop')
    'Populate topic model with the data from topic object.\n\n    Args:\n        topic_model: TopicModel. The model to populate.\n        topic: Topic. The topic domain object which should be used to\n            populate the model.\n\n    Returns:\n        TopicModel. Populated model.\n    '
    topic_model.description = topic.description
    topic_model.name = topic.name
    topic_model.canonical_name = topic.canonical_name
    topic_model.abbreviated_name = topic.abbreviated_name
    topic_model.url_fragment = topic.url_fragment
    topic_model.thumbnail_bg_color = topic.thumbnail_bg_color
    topic_model.thumbnail_filename = topic.thumbnail_filename
    topic_model.thumbnail_size_in_bytes = topic.thumbnail_size_in_bytes
    topic_model.canonical_story_references = [reference.to_dict() for reference in topic.canonical_story_references]
    topic_model.additional_story_references = [reference.to_dict() for reference in topic.additional_story_references]
    topic_model.uncategorized_skill_ids = topic.uncategorized_skill_ids
    topic_model.subtopics = [subtopic.to_dict() for subtopic in topic.subtopics]
    topic_model.subtopic_schema_version = topic.subtopic_schema_version
    topic_model.story_reference_schema_version = topic.story_reference_schema_version
    topic_model.next_subtopic_id = topic.next_subtopic_id
    topic_model.language_code = topic.language_code
    topic_model.meta_tag_content = topic.meta_tag_content
    topic_model.practice_tab_is_displayed = topic.practice_tab_is_displayed
    topic_model.page_title_fragment_for_web = topic.page_title_fragment_for_web
    topic_model.skill_ids_for_diagnostic_test = topic.skill_ids_for_diagnostic_test
    return topic_model

def populate_topic_summary_model_fields(topic_summary_model: topic_models.TopicSummaryModel, topic_summary: topic_domain.TopicSummary) -> topic_models.TopicSummaryModel:
    if False:
        i = 10
        return i + 15
    'Populate topic summary model with the data from topic summary object.\n\n    Args:\n        topic_summary_model: TopicSummaryModel. The model to populate.\n        topic_summary: TopicSummary. The topic summary domain object which\n            should be used to populate the model.\n\n    Returns:\n        TopicSummaryModel. Populated model.\n    '
    topic_summary_dict = {'name': topic_summary.name, 'description': topic_summary.description, 'canonical_name': topic_summary.canonical_name, 'language_code': topic_summary.language_code, 'version': topic_summary.version, 'additional_story_count': topic_summary.additional_story_count, 'canonical_story_count': topic_summary.canonical_story_count, 'uncategorized_skill_count': topic_summary.uncategorized_skill_count, 'subtopic_count': topic_summary.subtopic_count, 'total_skill_count': topic_summary.total_skill_count, 'total_published_node_count': topic_summary.total_published_node_count, 'thumbnail_filename': topic_summary.thumbnail_filename, 'thumbnail_bg_color': topic_summary.thumbnail_bg_color, 'topic_model_last_updated': topic_summary.topic_model_last_updated, 'topic_model_created_on': topic_summary.topic_model_created_on, 'url_fragment': topic_summary.url_fragment}
    if topic_summary_model is not None:
        topic_summary_model.populate(**topic_summary_dict)
    else:
        topic_summary_dict['id'] = topic_summary.id
        topic_summary_model = topic_models.TopicSummaryModel(**topic_summary_dict)
    return topic_summary_model

def get_topic_id_to_topic_name_dict(topic_ids: List[str]) -> Dict[str, str]:
    if False:
        i = 10
        return i + 15
    'Returns a dict with topic ID as key and topic name as value, for all\n    given topic IDs.\n\n    Args:\n        topic_ids: List(str). A list of topic IDs.\n\n    Raises:\n        Exception. The topic models for some of the given topic IDs do not\n            exist.\n\n    Returns:\n        dict(str, str). A dict with topic ID as key and topic name as value.\n    '
    topic_id_to_topic_name = {}
    topics = topic_fetchers.get_topics_by_ids(topic_ids)
    for topic in topics:
        if topic is None:
            continue
        topic_id_to_topic_name[topic.id] = topic.name
    correct_topic_ids = list(topic_id_to_topic_name.keys())
    incorrect_topic_ids = [topic_id for topic_id in topic_ids if topic_id not in correct_topic_ids]
    if incorrect_topic_ids:
        error_msg = 'No corresponding topic models exist for these topic IDs: %s.' % ', '.join(incorrect_topic_ids)
        raise Exception(error_msg)
    return topic_id_to_topic_name

def get_chapter_counts_in_topic_summaries(topic_summary_dicts: List[topic_domain.FrontendTopicSummaryDict]) -> Dict[str, topic_domain.TopicChapterCounts]:
    if False:
        while True:
            i = 10
    'Returns topic chapter counts for each topic summary dict.\n\n    Args:\n        topic_summary_dicts: List[FrontendTopicSummaryDict]. A list of\n            topic summary dicts.\n\n    Returns:\n        Dict[str, TopicChapterCounts]. Dict of topic id and topic chapter\n        counts domain object.\n    '
    topic_summary_id_mapping: Dict[str, topic_domain.FrontendTopicSummaryDict] = {}
    for topic_summary in topic_summary_dicts:
        topic_summary_id_mapping.update({topic_summary['id']: topic_summary})
    topic_ids = [summary['id'] for summary in topic_summary_dicts]
    all_topics = topic_fetchers.get_topics_by_ids(topic_ids)
    all_valid_topics = [topic for topic in all_topics if topic is not None]
    all_story_ids: List[str] = []
    topic_chapter_counts_dict: Dict[str, topic_domain.TopicChapterCounts] = {}
    for topic in all_valid_topics:
        story_ids = [story_reference.story_id for story_reference in topic.canonical_story_references]
        all_story_ids = all_story_ids + story_ids
    all_stories = story_fetchers.get_stories_by_ids(all_story_ids)
    all_valid_stories = [story for story in all_stories if story is not None]
    story_id_mapping: Dict[str, story_domain.Story] = {}
    for story in all_valid_stories:
        story_id_mapping.update({story.id: story})
    for topic in all_valid_topics:
        topic_summary_dict = topic_summary_id_mapping[topic.id]
        upcoming_chapters_count = 0
        overdue_chapters_count = 0
        total_chapter_counts = []
        published_chapter_counts = []
        stories = [story_id_mapping[story_reference.story_id] for story_reference in topic.canonical_story_references]
        for story in stories:
            nodes = story.story_contents.nodes
            total_chapters_count = len(nodes)
            published_chapters_count = 0
            for node in nodes:
                if node.status == constants.STORY_NODE_STATUS_PUBLISHED:
                    published_chapters_count += 1
                elif node.is_node_upcoming():
                    upcoming_chapters_count += 1
                elif node.is_node_behind_schedule():
                    overdue_chapters_count += 1
            total_chapter_counts.append(total_chapters_count)
            published_chapter_counts.append(published_chapters_count)
        topic_chapter_counts = topic_domain.TopicChapterCounts(upcoming_chapters_count, overdue_chapters_count, total_chapter_counts, published_chapter_counts)
        topic_chapter_counts_dict.update({topic_summary_dict['id']: topic_chapter_counts})
    return topic_chapter_counts_dict