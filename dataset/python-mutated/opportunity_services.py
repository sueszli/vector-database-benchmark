"""Commands that can be used to operate on opportunity models."""
from __future__ import annotations
import collections
import logging
from core import feconf
from core.constants import constants
from core.domain import exp_domain
from core.domain import exp_fetchers
from core.domain import opportunity_domain
from core.domain import question_fetchers
from core.domain import story_domain
from core.domain import story_fetchers
from core.domain import suggestion_services
from core.domain import topic_domain
from core.domain import topic_fetchers
from core.domain import translation_services
from core.platform import models
from typing import Dict, List, Optional, Sequence, Tuple
MYPY = False
if MYPY:
    from mypy_imports import opportunity_models
    from mypy_imports import user_models
(opportunity_models, user_models) = models.Registry.import_models([models.Names.OPPORTUNITY, models.Names.USER])

def is_exploration_available_for_contribution(exp_id: str) -> bool:
    if False:
        return 10
    "Checks whether a given exploration id belongs to a curated list of\n    exploration i.e, whether it's used as the chapter of any story.\n\n    Args:\n        exp_id: str. The id of the exploration which is needed to be checked.\n\n    Returns:\n        bool. Whether the given exp_id belongs to the curated explorations.\n    "
    model = opportunity_models.ExplorationOpportunitySummaryModel.get(exp_id, strict=False)
    return model is not None

def get_exploration_opportunity_summary_from_model(model: opportunity_models.ExplorationOpportunitySummaryModel) -> opportunity_domain.ExplorationOpportunitySummary:
    if False:
        i = 10
        return i + 15
    'Returns the ExplorationOpportunitySummary object out of the model.\n\n    Args:\n        model: ExplorationOpportunitySummaryModel. The exploration opportunity\n            summary model.\n\n    Returns:\n        ExplorationOpportunitySummary. The corresponding\n        ExplorationOpportunitySummary object.\n    '
    set_of_all_languages = set(model.incomplete_translation_language_codes + model.language_codes_needing_voice_artists + model.language_codes_with_assigned_voice_artists)
    supported_language_codes = set((language['id'] for language in constants.SUPPORTED_AUDIO_LANGUAGES))
    missing_language_codes = list(supported_language_codes - set_of_all_languages)
    if missing_language_codes:
        logging.info('Missing language codes %s in exploration opportunity model with id %s' % (missing_language_codes, model.id))
    new_incomplete_translation_language_codes = model.incomplete_translation_language_codes + missing_language_codes
    return opportunity_domain.ExplorationOpportunitySummary(model.id, model.topic_id, model.topic_name, model.story_id, model.story_title, model.chapter_title, model.content_count, new_incomplete_translation_language_codes, model.translation_counts, model.language_codes_needing_voice_artists, model.language_codes_with_assigned_voice_artists, {}, False)

def _construct_new_opportunity_summary_models(exploration_opportunity_summary_list: List[opportunity_domain.ExplorationOpportunitySummary]) -> List[opportunity_models.ExplorationOpportunitySummaryModel]:
    if False:
        i = 10
        return i + 15
    'Create ExplorationOpportunitySummaryModels from domain objects.\n\n    Args:\n        exploration_opportunity_summary_list: list(\n            ExplorationOpportunitySummary). A list of exploration opportunity\n            summary object.\n\n    Returns:\n        list(ExplorationOpportunitySummaryModel). A list of\n        ExplorationOpportunitySummaryModel to be stored in the datastore.\n    '
    exploration_opportunity_summary_model_list = []
    for opportunity_summary in exploration_opportunity_summary_list:
        model = opportunity_models.ExplorationOpportunitySummaryModel(id=opportunity_summary.id, topic_id=opportunity_summary.topic_id, topic_name=opportunity_summary.topic_name, story_id=opportunity_summary.story_id, story_title=opportunity_summary.story_title, chapter_title=opportunity_summary.chapter_title, content_count=opportunity_summary.content_count, incomplete_translation_language_codes=opportunity_summary.incomplete_translation_language_codes, translation_counts=opportunity_summary.translation_counts, language_codes_needing_voice_artists=opportunity_summary.language_codes_needing_voice_artists, language_codes_with_assigned_voice_artists=opportunity_summary.language_codes_with_assigned_voice_artists)
        exploration_opportunity_summary_model_list.append(model)
    return exploration_opportunity_summary_model_list

def _save_multi_exploration_opportunity_summary(exploration_opportunity_summary_list: List[opportunity_domain.ExplorationOpportunitySummary]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Stores multiple ExplorationOpportunitySummary into datastore as a\n    ExplorationOpportunitySummaryModel.\n\n    Args:\n        exploration_opportunity_summary_list: list(\n            ExplorationOpportunitySummary). A list of exploration opportunity\n            summary object.\n    '
    exploration_opportunity_summary_model_list = _construct_new_opportunity_summary_models(exploration_opportunity_summary_list)
    opportunity_models.ExplorationOpportunitySummaryModel.update_timestamps_multi(exploration_opportunity_summary_model_list)
    opportunity_models.ExplorationOpportunitySummaryModel.put_multi(exploration_opportunity_summary_model_list)

def create_exp_opportunity_summary(topic: topic_domain.Topic, story: story_domain.Story, exploration: exp_domain.Exploration) -> opportunity_domain.ExplorationOpportunitySummary:
    if False:
        print('Hello World!')
    'Create an ExplorationOpportunitySummary object with the given topic,\n    story and exploration object.\n\n    Args:\n        topic: Topic. The topic object to which the opportunity belongs.\n        story: Story. The story object to which the opportunity belongs.\n        exploration: Exploration. The exploration object to which the\n            opportunity belongs.\n\n    Returns:\n        ExplorationOpportunitySummary. The exploration opportunity summary\n        object.\n    '
    complete_translation_language_list = translation_services.get_languages_with_complete_translation(exploration)
    language_codes_needing_voice_artists = set(complete_translation_language_list)
    incomplete_translation_language_codes = _compute_exploration_incomplete_translation_languages(complete_translation_language_list)
    if exploration.language_code in incomplete_translation_language_codes:
        incomplete_translation_language_codes.remove(exploration.language_code)
        language_codes_needing_voice_artists.add(exploration.language_code)
    content_count = exploration.get_content_count()
    translation_counts = translation_services.get_translation_counts(feconf.TranslatableEntityType.EXPLORATION, exploration)
    story_node = story.story_contents.get_node_with_corresponding_exp_id(exploration.id)
    exploration_opportunity_summary = opportunity_domain.ExplorationOpportunitySummary(exploration.id, topic.id, topic.name, story.id, story.title, story_node.title, content_count, incomplete_translation_language_codes, translation_counts, list(language_codes_needing_voice_artists), [], {})
    return exploration_opportunity_summary

def _compute_exploration_incomplete_translation_languages(complete_translation_languages: List[str]) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Computes all languages that are not 100% translated in an exploration.\n\n    Args:\n        complete_translation_languages: list(str). List of complete translation\n            language codes in the exploration.\n\n    Returns:\n        list(str). List of incomplete translation language codes sorted\n        alphabetically.\n    '
    audio_language_codes = set((language['id'] for language in constants.SUPPORTED_AUDIO_LANGUAGES))
    incomplete_translation_language_codes = audio_language_codes - set(complete_translation_languages)
    return sorted(list(incomplete_translation_language_codes))

def add_new_exploration_opportunities(story_id: str, exp_ids: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Adds new exploration opportunity into the model.\n\n    Args:\n        story_id: str. ID of the story.\n        exp_ids: list(str). A list of exploration ids for which new\n            opportunities are to be created. All exp_ids must be part of the\n            given story.\n    '
    story = story_fetchers.get_story_by_id(story_id)
    topic = topic_fetchers.get_topic_by_id(story.corresponding_topic_id)
    _create_exploration_opportunities(story, topic, exp_ids)

def _create_exploration_opportunities(story: story_domain.Story, topic: topic_domain.Topic, exp_ids: List[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Creates new exploration opportunities corresponding to the supplied\n    story, topic, and exploration IDs.\n\n    Args:\n        story: Story. The story domain object corresponding to the exploration\n            opportunities.\n        topic: Topic. The topic domain object corresponding to the exploration\n            opportunities.\n        exp_ids: list(str). A list of exploration ids for which new\n            opportunities are to be created. All exp_ids must be part of the\n            given story.\n    '
    explorations = exp_fetchers.get_multiple_explorations_by_id(exp_ids)
    exploration_opportunity_summary_list = []
    for exploration in explorations.values():
        exploration_opportunity_summary_list.append(create_exp_opportunity_summary(topic, story, exploration))
    _save_multi_exploration_opportunity_summary(exploration_opportunity_summary_list)

def compute_opportunity_models_with_updated_exploration(exp_id: str, content_count: int, translation_counts: Dict[str, int]) -> List[opportunity_models.ExplorationOpportunitySummaryModel]:
    if False:
        return 10
    'Updates the opportunities models with the changes made in the\n    exploration.\n\n    Args:\n        exp_id: str. The exploration id which is also the id of the opportunity\n            model.\n        content_count: int. The number of contents available in the exploration.\n        translation_counts: dict(str, int). The number of translations available\n            for the exploration in different languages.\n\n    Returns:\n        list(ExplorationOpportunitySummaryModel). A list of opportunity models\n        which are updated.\n    '
    updated_exploration = exp_fetchers.get_exploration_by_id(exp_id)
    complete_translation_language_list = []
    for (language_code, translation_count) in translation_counts.items():
        if translation_count == content_count:
            complete_translation_language_list.append(language_code)
    model = opportunity_models.ExplorationOpportunitySummaryModel.get(exp_id)
    exploration_opportunity_summary = get_exploration_opportunity_summary_from_model(model)
    exploration_opportunity_summary.content_count = content_count
    exploration_opportunity_summary.translation_counts = translation_counts
    incomplete_translation_language_codes = _compute_exploration_incomplete_translation_languages(complete_translation_language_list)
    if updated_exploration.language_code in incomplete_translation_language_codes:
        incomplete_translation_language_codes.remove(updated_exploration.language_code)
    exploration_opportunity_summary.incomplete_translation_language_codes = incomplete_translation_language_codes
    new_languages_for_voiceover = set(complete_translation_language_list) - set(exploration_opportunity_summary.language_codes_with_assigned_voice_artists)
    language_codes_needing_voice_artists_set = set(exploration_opportunity_summary.language_codes_needing_voice_artists)
    language_codes_needing_voice_artists_set |= set(new_languages_for_voiceover)
    exploration_opportunity_summary.language_codes_needing_voice_artists = list(language_codes_needing_voice_artists_set)
    exploration_opportunity_summary.validate()
    return _construct_new_opportunity_summary_models([exploration_opportunity_summary])

def update_translation_opportunity_with_accepted_suggestion(exploration_id: str, language_code: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Updates the translation opportunity for the accepted suggestion in the\n    ExplorationOpportunitySummaryModel.\n\n    Args:\n        exploration_id: str. The ID of the exploration.\n        language_code: str. The langauge code of the accepted translation\n            suggestion.\n    '
    model = opportunity_models.ExplorationOpportunitySummaryModel.get(exploration_id)
    exp_opportunity_summary = get_exploration_opportunity_summary_from_model(model)
    if language_code in exp_opportunity_summary.translation_counts:
        exp_opportunity_summary.translation_counts[language_code] += 1
    else:
        exp_opportunity_summary.translation_counts[language_code] = 1
    if exp_opportunity_summary.content_count == exp_opportunity_summary.translation_counts[language_code]:
        exp_opportunity_summary.incomplete_translation_language_codes.remove(language_code)
        exp_opportunity_summary.language_codes_needing_voice_artists.append(language_code)
    exp_opportunity_summary.validate()
    _save_multi_exploration_opportunity_summary([exp_opportunity_summary])

def update_exploration_opportunities_with_story_changes(story: story_domain.Story, exp_ids: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Updates the opportunities models with the story changes.\n\n    Args:\n        story: Story. The new story object.\n        exp_ids: list(str). A list of exploration IDs whose exploration\n            opportunity summary models need to be updated.\n    '
    exp_opportunity_models_with_none = opportunity_models.ExplorationOpportunitySummaryModel.get_multi(exp_ids)
    exploration_opportunity_summary_list = []
    for exp_opportunity_model in exp_opportunity_models_with_none:
        assert exp_opportunity_model is not None
        exploration_opportunity_summary = get_exploration_opportunity_summary_from_model(exp_opportunity_model)
        exploration_opportunity_summary.story_title = story.title
        node = story.story_contents.get_node_with_corresponding_exp_id(exploration_opportunity_summary.id)
        exploration_opportunity_summary.chapter_title = node.title
        exploration_opportunity_summary.validate()
        exploration_opportunity_summary_list.append(exploration_opportunity_summary)
    _save_multi_exploration_opportunity_summary(exploration_opportunity_summary_list)

def delete_exploration_opportunities(exp_ids: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Deletes the ExplorationOpportunitySummaryModel models corresponding to\n    the given exp_ids.\n\n    Args:\n        exp_ids: list(str). A list of exploration IDs whose opportunity summary\n            models are to be deleted.\n    '
    exp_opportunity_models = opportunity_models.ExplorationOpportunitySummaryModel.get_multi(exp_ids)
    exp_opportunity_models_to_be_deleted = [model for model in exp_opportunity_models if model is not None]
    opportunity_models.ExplorationOpportunitySummaryModel.delete_multi(exp_opportunity_models_to_be_deleted)

def delete_exploration_opportunities_corresponding_to_topic(topic_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Deletes the ExplorationOpportunitySummaryModel models which corresponds\n    to the given topic_id.\n\n    Args:\n        topic_id: str. The ID of the topic.\n    '
    exp_opportunity_models = opportunity_models.ExplorationOpportunitySummaryModel.get_by_topic(topic_id)
    opportunity_models.ExplorationOpportunitySummaryModel.delete_multi(list(exp_opportunity_models))

def update_exploration_opportunities(old_story: story_domain.Story, new_story: story_domain.Story) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Updates the opportunities models according to the changes made in the\n    story.\n\n    Args:\n        old_story: Story. The old story object which is now updated.\n        new_story: Story. The new story object.\n    '
    model_ids_need_update = set([])
    exp_ids_in_old_story = old_story.story_contents.get_all_linked_exp_ids()
    exp_ids_in_new_story = new_story.story_contents.get_all_linked_exp_ids()
    new_added_exp_ids = set(exp_ids_in_new_story) - set(exp_ids_in_old_story)
    deleted_exp_ids = set(exp_ids_in_old_story) - set(exp_ids_in_new_story)
    unchanged_exp_ids = set(exp_ids_in_new_story) - new_added_exp_ids
    if old_story.title != new_story.title:
        model_ids_need_update |= set(unchanged_exp_ids)
    else:
        for exp_id in unchanged_exp_ids:
            new_node = new_story.story_contents.get_node_with_corresponding_exp_id(exp_id)
            old_node = old_story.story_contents.get_node_with_corresponding_exp_id(exp_id)
            if old_node.title != new_node.title:
                model_ids_need_update.add(exp_id)
    update_exploration_opportunities_with_story_changes(new_story, list(model_ids_need_update))
    add_new_exploration_opportunities(new_story.id, list(new_added_exp_ids))
    delete_exploration_opportunities(list(deleted_exp_ids))

def delete_exp_opportunities_corresponding_to_story(story_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Deletes the ExplorationOpportunitySummaryModel models which corresponds\n    to the given story_id.\n\n    Args:\n        story_id: str. The ID of the story.\n    '
    exp_opprtunity_model_class = opportunity_models.ExplorationOpportunitySummaryModel
    exp_opportunity_models: Sequence[opportunity_models.ExplorationOpportunitySummaryModel] = exp_opprtunity_model_class.get_all().filter(exp_opprtunity_model_class.story_id == story_id).fetch()
    exp_opprtunity_model_class.delete_multi(list(exp_opportunity_models))

def get_translation_opportunities(language_code: str, topic_name: Optional[str], cursor: Optional[str]) -> Tuple[List[opportunity_domain.ExplorationOpportunitySummary], Optional[str], bool]:
    if False:
        print('Hello World!')
    'Returns a list of opportunities available for translation in a specific\n    language.\n\n    Args:\n        cursor: str or None. If provided, the list of returned entities\n            starts from this datastore cursor. Otherwise, the returned\n            entities start from the beginning of the full list of entities.\n        language_code: str. The language for which translation opportunities\n            should be fetched.\n        topic_name: str or None. The topic for which translation opportunities\n            should be fetched. If topic_name is None or empty, fetch\n            translation opportunities from all topics.\n\n    Returns:\n        3-tuple(opportunities, cursor, more). where:\n            opportunities: list(ExplorationOpportunitySummary). A list of\n                ExplorationOpportunitySummary domain objects.\n            cursor: str or None. A query cursor pointing to the next batch of\n                results. If there are no more results, this might be None.\n            more: bool. If True, there are (probably) more results after this\n                batch. If False, there are no further results after this batch.\n    '
    page_size = constants.OPPORTUNITIES_PAGE_SIZE
    (exp_opportunity_summary_models, cursor, more) = opportunity_models.ExplorationOpportunitySummaryModel.get_all_translation_opportunities(page_size, cursor, language_code, topic_name)
    opportunity_summaries = []
    opportunity_summary_exp_ids = [opportunity.id for opportunity in exp_opportunity_summary_models]
    exp_id_to_in_review_count = {}
    if len(opportunity_summary_exp_ids) > 0:
        exp_id_to_in_review_count = _build_exp_id_to_translation_suggestion_in_review_count(opportunity_summary_exp_ids, language_code)
    for exp_opportunity_summary_model in exp_opportunity_summary_models:
        opportunity_summary = get_exploration_opportunity_summary_from_model(exp_opportunity_summary_model)
        if opportunity_summary.id in exp_id_to_in_review_count:
            opportunity_summary.translation_in_review_counts = {language_code: exp_id_to_in_review_count[opportunity_summary.id]}
        opportunity_summaries.append(opportunity_summary)
    return (opportunity_summaries, cursor, more)

def _build_exp_id_to_translation_suggestion_in_review_count(exp_ids: List[str], language_code: str) -> Dict[str, int]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a dict mapping exploration ID to the count of corresponding\n    translation suggestions that are currently in review.\n\n    Args:\n        exp_ids: list(str). List of exploration IDs for which to count\n            corresponding translations suggestions.\n        language_code: str. The language for which translation suggestions\n            should be fetched.\n\n    Returns:\n        dict(str, int). Dict of exploration IDs to counts of corresponding\n        translation suggestions currently in review.\n    '
    exp_id_to_in_review_count: Dict[str, int] = collections.defaultdict(int)
    suggestions_in_review = suggestion_services.get_translation_suggestions_in_review_by_exp_ids(exp_ids, language_code)
    for suggestion in suggestions_in_review:
        if suggestion is not None:
            exp_id_to_in_review_count[suggestion.target_id] += 1
    return exp_id_to_in_review_count

def get_exploration_opportunity_summaries_by_ids(ids: List[str]) -> Dict[str, Optional[opportunity_domain.ExplorationOpportunitySummary]]:
    if False:
        while True:
            i = 10
    'Returns a dict with key as id and value representing\n    ExplorationOpportunitySummary objects corresponding to the opportunity id.\n\n    Args:\n        ids: list(str). A list of opportunity ids.\n\n    Returns:\n        dict(str, ExplorationOpportunitySummary|None). A dict with key as the\n        opportunity id and values representing the ExplorationOpportunitySummary\n        domain objects corresponding to the opportunity id if exist else None.\n    '
    opportunities: Dict[str, Optional[opportunity_domain.ExplorationOpportunitySummary]] = {opportunity_id: None for opportunity_id in ids}
    exp_opportunity_summary_models = opportunity_models.ExplorationOpportunitySummaryModel.get_multi(ids)
    for exp_opportunity_summary_model in exp_opportunity_summary_models:
        if exp_opportunity_summary_model is not None:
            opportunities[exp_opportunity_summary_model.id] = get_exploration_opportunity_summary_from_model(exp_opportunity_summary_model)
    return opportunities

def get_exploration_opportunity_summary_by_id(opportunity_id: str) -> Optional[opportunity_domain.ExplorationOpportunitySummary]:
    if False:
        while True:
            i = 10
    'Returns an ExplorationOpportunitySummary object corresponding to the\n    opportunity id.\n\n    Args:\n        opportunity_id: str. An opportunity id.\n\n    Returns:\n        ExplorationOpportunitySummary|None. An ExplorationOpportunitySummary\n        domain object corresponding to the opportunity id if it exists, else\n        None.\n    '
    exp_opportunity_summary_model = opportunity_models.ExplorationOpportunitySummaryModel.get(opportunity_id, strict=False)
    if exp_opportunity_summary_model is None:
        return None
    return get_exploration_opportunity_summary_from_model(exp_opportunity_summary_model)

def get_exploration_opportunity_summaries_by_topic_id(topic_id: str) -> List[opportunity_domain.ExplorationOpportunitySummary]:
    if False:
        print('Hello World!')
    'Returns a list of all exploration opportunity summaries\n    with the given topic ID.\n\n    Args:\n        topic_id: str. The topic for which opportunity summaries\n            are fetched.\n\n    Returns:\n        list(ExplorationOpportunitySummary). A list of all\n        exploration opportunity summaries with the given topic ID.\n    '
    opportunity_summaries = []
    exp_opportunity_summary_models = opportunity_models.ExplorationOpportunitySummaryModel.get_by_topic(topic_id)
    for exp_opportunity_summary_model in exp_opportunity_summary_models:
        opportunity_summary = get_exploration_opportunity_summary_from_model(exp_opportunity_summary_model)
        opportunity_summaries.append(opportunity_summary)
    return opportunity_summaries

def update_opportunities_with_new_topic_name(topic_id: str, topic_name: str) -> None:
    if False:
        return 10
    'Updates the exploration opportunity summary models with new topic name.\n\n    Args:\n        topic_id: str. The corresponding topic id of the opportunity.\n        topic_name: str. The new topic name.\n    '
    exp_opportunity_models = opportunity_models.ExplorationOpportunitySummaryModel.get_by_topic(topic_id)
    exploration_opportunity_summary_list = []
    for exp_opportunity_model in exp_opportunity_models:
        exploration_opportunity_summary = get_exploration_opportunity_summary_from_model(exp_opportunity_model)
        exploration_opportunity_summary.topic_name = topic_name
        exploration_opportunity_summary.validate()
        exploration_opportunity_summary_list.append(exploration_opportunity_summary)
    _save_multi_exploration_opportunity_summary(exploration_opportunity_summary_list)

def get_skill_opportunity_from_model(model: opportunity_models.SkillOpportunityModel) -> opportunity_domain.SkillOpportunity:
    if False:
        return 10
    'Returns a SkillOpportunity domain object from a SkillOpportunityModel.\n\n    Args:\n        model: SkillOpportunityModel. The skill opportunity model.\n\n    Returns:\n        SkillOpportunity. The corresponding SkillOpportunity object.\n    '
    return opportunity_domain.SkillOpportunity(model.id, model.skill_description, model.question_count)

def get_skill_opportunities(cursor: Optional[str]) -> Tuple[List[opportunity_domain.SkillOpportunity], Optional[str], bool]:
    if False:
        i = 10
        return i + 15
    'Returns a list of skill opportunities available for questions.\n\n    Args:\n        cursor: str or None. If provided, the list of returned entities\n            starts from this datastore cursor. Otherwise, the returned\n            entities start from the beginning of the full list of entities.\n\n    Returns:\n        3-tuple(opportunities, cursor, more). where:\n            opportunities: list(SkillOpportunity). A list of SkillOpportunity\n                domain objects.\n            cursor: str or None. A query cursor pointing to the next\n                batch of results. If there are no more results, this might\n                be None.\n            more: bool. If True, there are (probably) more results after\n                this batch. If False, there are no further results after\n                this batch.\n    '
    (skill_opportunity_models, cursor, more) = opportunity_models.SkillOpportunityModel.get_skill_opportunities(constants.OPPORTUNITIES_PAGE_SIZE, cursor)
    opportunities = []
    for skill_opportunity_model in skill_opportunity_models:
        skill_opportunity = get_skill_opportunity_from_model(skill_opportunity_model)
        opportunities.append(skill_opportunity)
    return (opportunities, cursor, more)

def get_skill_opportunities_by_ids(ids: List[str]) -> Dict[str, Optional[opportunity_domain.SkillOpportunity]]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of SkillOpportunity domain objects corresponding to the\n    given list of ids.\n\n    Args:\n        ids: list(str). A list of the opportunity ids.\n\n    Returns:\n        dict(str, SkillOpportunity|None). A dict with key as the\n        opportunity id and values representing the SkillOpportunity\n        domain objects corresponding to the opportunity id if exist else None.\n    '
    opportunities: Dict[str, Optional[opportunity_domain.SkillOpportunity]] = {opportunity_id: None for opportunity_id in ids}
    skill_opportunity_models = opportunity_models.SkillOpportunityModel.get_multi(ids)
    for skill_opportunity_model in skill_opportunity_models:
        if skill_opportunity_model is not None:
            opportunities[skill_opportunity_model.id] = get_skill_opportunity_from_model(skill_opportunity_model)
    return opportunities

def create_skill_opportunity(skill_id: str, skill_description: str) -> None:
    if False:
        while True:
            i = 10
    'Creates a SkillOpportunityModel entity in the datastore.\n\n    Args:\n        skill_id: str. The skill_id of the opportunity.\n        skill_description: str. The skill_description of the opportunity.\n\n    Raises:\n        Exception. If a SkillOpportunityModel corresponding to the supplied\n            skill_id already exists.\n    '
    skill_opportunity_model = opportunity_models.SkillOpportunityModel.get_by_id(skill_id)
    if skill_opportunity_model is not None:
        raise Exception('SkillOpportunity corresponding to skill ID %s already exists.' % skill_id)
    (questions, _) = question_fetchers.get_questions_and_skill_descriptions_by_skill_ids(constants.MAX_QUESTIONS_PER_SKILL, [skill_id], 0)
    skill_opportunity = opportunity_domain.SkillOpportunity(skill_id=skill_id, skill_description=skill_description, question_count=len(questions))
    _save_skill_opportunities([skill_opportunity])

def _save_skill_opportunities(skill_opportunities: List[opportunity_domain.SkillOpportunity]) -> None:
    if False:
        print('Hello World!')
    'Saves SkillOpportunity domain objects into datastore as\n    SkillOpportunityModel objects.\n\n    Args:\n        skill_opportunities: list(SkillOpportunity). A list of SkillOpportunity\n            domain objects.\n    '
    skill_opportunity_models = []
    for skill_opportunity in skill_opportunities:
        skill_opportunity.validate()
        model = opportunity_models.SkillOpportunityModel(id=skill_opportunity.id, skill_description=skill_opportunity.skill_description, question_count=skill_opportunity.question_count)
        skill_opportunity_models.append(model)
    opportunity_models.SkillOpportunityModel.update_timestamps_multi(skill_opportunity_models)
    opportunity_models.SkillOpportunityModel.put_multi(skill_opportunity_models)

def update_skill_opportunity_skill_description(skill_id: str, new_description: str) -> None:
    if False:
        while True:
            i = 10
    'Updates the skill_description of the SkillOpportunityModel with\n    new_description.\n\n    Args:\n        skill_id: str. The corresponding skill_id of the opportunity.\n        new_description: str. The new skill_description.\n    '
    skill_opportunity = _get_skill_opportunity(skill_id)
    if skill_opportunity is not None:
        skill_opportunity.skill_description = new_description
        _save_skill_opportunities([skill_opportunity])

def _get_skill_opportunity(skill_id: str) -> Optional[opportunity_domain.SkillOpportunity]:
    if False:
        while True:
            i = 10
    'Returns the SkillOpportunity domain object representing a\n    SkillOpportunityModel with the supplied skill_id in the datastore.\n\n    Args:\n        skill_id: str. The corresponding skill_id of the opportunity.\n\n    Returns:\n        SkillOpportunity or None. The domain object representing a\n        SkillOpportunity with the supplied skill_id, or None if it does not\n        exist.\n    '
    skill_opportunity_model = opportunity_models.SkillOpportunityModel.get_by_id(skill_id)
    if skill_opportunity_model is not None:
        return get_skill_opportunity_from_model(skill_opportunity_model)
    return None

def delete_skill_opportunity(skill_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Deletes the SkillOpportunityModel corresponding to the supplied skill_id.\n\n    Args:\n        skill_id: str. The skill_id corresponding to the to-be-deleted\n            SkillOpportunityModel.\n    '
    skill_opportunity_model = opportunity_models.SkillOpportunityModel.get_by_id(skill_id)
    if skill_opportunity_model is not None:
        opportunity_models.SkillOpportunityModel.delete(skill_opportunity_model)

def increment_question_counts(skill_ids: List[str], delta: int) -> None:
    if False:
        return 10
    'Increments question_count(s) of SkillOpportunityModel(s) with\n    corresponding skill_ids.\n\n    Args:\n        skill_ids: list(str). A list of skill_ids corresponding to\n            SkillOpportunityModel(s).\n        delta: int. The delta for which to increment each question_count.\n    '
    updated_skill_opportunities = _get_skill_opportunities_with_updated_question_counts(skill_ids, delta)
    _save_skill_opportunities(updated_skill_opportunities)

def update_skill_opportunities_on_question_linked_skills_change(old_skill_ids: List[str], new_skill_ids: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Updates question_count(s) of SkillOpportunityModel(s) corresponding to\n    the change in linked skill IDs for a question from old_skill_ids to\n    new_skill_ids, e.g. if skill_id1 is in old_skill_ids, but not in\n    new_skill_ids, the question_count of the SkillOpportunityModel for skill_id1\n    would be decremented.\n\n    NOTE: Since this method is updating the question_counts based on the change\n    of skill_ids from old_skill_ids to new_skill_ids, the input skill_id lists\n    must be related.\n\n    Args:\n        old_skill_ids: list(str). A list of old skill_id(s).\n        new_skill_ids: list(str). A list of new skill_id(s).\n    '
    old_skill_ids_set = set(old_skill_ids)
    new_skill_ids_set = set(new_skill_ids)
    new_skill_ids_added_to_question = new_skill_ids_set - old_skill_ids_set
    skill_ids_removed_from_question = old_skill_ids_set - new_skill_ids_set
    updated_skill_opportunities = []
    updated_skill_opportunities.extend(_get_skill_opportunities_with_updated_question_counts(list(new_skill_ids_added_to_question), 1))
    updated_skill_opportunities.extend(_get_skill_opportunities_with_updated_question_counts(list(skill_ids_removed_from_question), -1))
    _save_skill_opportunities(updated_skill_opportunities)

def _get_skill_opportunities_with_updated_question_counts(skill_ids: List[str], delta: int) -> List[opportunity_domain.SkillOpportunity]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of SkillOpportunities with corresponding skill_ids\n    with question_count(s) updated by delta.\n\n    Args:\n        skill_ids: List(str). The IDs of the matching SkillOpportunityModels\n            in the datastore.\n        delta: int. The delta by which to update each question_count (can be\n            negative).\n\n    Returns:\n        list(SkillOpportunity). The updated SkillOpportunities.\n    '
    updated_skill_opportunities = []
    skill_opportunity_models = opportunity_models.SkillOpportunityModel.get_multi(skill_ids)
    for skill_opportunity_model in skill_opportunity_models:
        if skill_opportunity_model is not None:
            skill_opportunity = get_skill_opportunity_from_model(skill_opportunity_model)
            skill_opportunity.question_count = max(skill_opportunity.question_count + delta, 0)
            updated_skill_opportunities.append(skill_opportunity)
    return updated_skill_opportunities

def regenerate_opportunities_related_to_topic(topic_id: str, delete_existing_opportunities: bool=False) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Regenerates opportunity models which belongs to a given topic.\n\n    Args:\n        topic_id: str. The ID of the topic.\n        delete_existing_opportunities: bool. Whether to delete all the existing\n            opportunities related to the given topic.\n\n    Returns:\n        int. The number of opportunity models created.\n\n    Raises:\n        Exception. Failure to regenerate opportunities for given topic.\n    '
    if delete_existing_opportunities:
        exp_opportunity_models = opportunity_models.ExplorationOpportunitySummaryModel.get_by_topic(topic_id)
        opportunity_models.ExplorationOpportunitySummaryModel.delete_multi(list(exp_opportunity_models))
    topic = topic_fetchers.get_topic_by_id(topic_id)
    story_ids = topic.get_canonical_story_ids()
    stories = story_fetchers.get_stories_by_ids(story_ids)
    exp_ids = []
    non_existing_story_ids = []
    for (index, story) in enumerate(stories):
        if story is None:
            non_existing_story_ids.append(story_ids[index])
        else:
            exp_ids += story.story_contents.get_all_linked_exp_ids()
    exp_ids_to_exp = exp_fetchers.get_multiple_explorations_by_id(exp_ids, strict=False)
    non_existing_exp_ids = set(exp_ids) - set(exp_ids_to_exp.keys())
    if len(non_existing_exp_ids) > 0 or len(non_existing_story_ids) > 0:
        raise Exception('Failed to regenerate opportunities for topic id: %s, missing_exp_with_ids: %s, missing_story_with_ids: %s' % (topic_id, list(non_existing_exp_ids), non_existing_story_ids))
    exploration_opportunity_summary_list = []
    for story in stories:
        assert story is not None
        for exp_id in story.story_contents.get_all_linked_exp_ids():
            exploration_opportunity_summary_list.append(create_exp_opportunity_summary(topic, story, exp_ids_to_exp[exp_id]))
    _save_multi_exploration_opportunity_summary(exploration_opportunity_summary_list)
    return len(exploration_opportunity_summary_list)

def update_pinned_opportunity_model(user_id: str, language_code: str, topic_id: str, lesson_id: Optional[str]) -> None:
    if False:
        print('Hello World!')
    'Pins/Unpins Reviewable opportunities in Contributor Dashboard.\n\n    Args:\n        user_id: str. The ID of the user.\n        language_code: str. The language code for which opportunity\n            has to be pinned.\n        topic_id: str. The topic id of the opportunity to be\n            pinned.\n        lesson_id: str or None. The opportunity_id/exp_id of opportunity\n            to be pinned. None if user wants to unpin the opportunity.\n    '
    pinned_opportunity = user_models.PinnedOpportunityModel.get_model(user_id, language_code, topic_id)
    if not pinned_opportunity and (not lesson_id):
        return
    if not pinned_opportunity and lesson_id:
        user_models.PinnedOpportunityModel.create(user_id=user_id, language_code=language_code, topic_id=topic_id, opportunity_id=lesson_id)
    elif pinned_opportunity:
        pinned_opportunity.opportunity_id = lesson_id
        pinned_opportunity.update_timestamps()
        pinned_opportunity.put()

def get_pinned_lesson(user_id: str, language_code: str, topic_id: str) -> Optional[opportunity_domain.ExplorationOpportunitySummary]:
    if False:
        print('Hello World!')
    "Retrieves the pinned lesson for a user in a specific language and topic.\n\n    NOTE: If the pinned lesson exists, it will have the 'is_pinned'\n    attribute set to True.\n\n    Args:\n        user_id: str. The ID of the user for whom to retrieve the pinned\n            lesson.\n        language_code: str. The ISO 639-1 language code for the\n            desired language.\n        topic_id: str. The ID of the topic for which to retrieve\n            the pinned lesson.\n\n    Returns:\n        ExplorationOpportunitySummary or None. The pinned lesson as an\n        ExplorationOpportunitySummary object, or None if no\n        pinned lesson exists.\n    "
    pinned_opportunity = user_models.PinnedOpportunityModel.get_model(user_id, language_code, topic_id)
    if pinned_opportunity and pinned_opportunity.opportunity_id is not None:
        model = opportunity_models.ExplorationOpportunitySummaryModel.get(pinned_opportunity.opportunity_id)
        exploration_opportunity_summary = get_exploration_opportunity_summary_from_model(model)
        exploration_opportunity_summary.is_pinned = True
        return exploration_opportunity_summary
    return None