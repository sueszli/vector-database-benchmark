"""Services for questions data model."""
from __future__ import annotations
import copy
import logging
from core import feconf
from core import utils
from core.constants import constants
from core.domain import opportunity_services
from core.domain import question_domain
from core.domain import question_fetchers
from core.domain import skill_domain
from core.domain import skill_fetchers
from core.domain import state_domain
from core.platform import models
from typing import Dict, List, Literal, Optional, Tuple, Union, cast, overload
MYPY = False
if MYPY:
    from mypy_imports import question_models
    from mypy_imports import skill_models
    from mypy_imports import transaction_services
(question_models, skill_models) = models.Registry.import_models([models.Names.QUESTION, models.Names.SKILL])
transaction_services = models.Registry.import_transaction_services()

def create_new_question(committer_id: str, question: question_domain.Question, commit_message: str) -> None:
    if False:
        while True:
            i = 10
    'Creates a new question.\n\n    Args:\n        committer_id: str. ID of the committer.\n        question: Question. Question domain object.\n        commit_message: str. A description of changes made to the question.\n    '
    question.validate()
    model = question_models.QuestionModel(id=question.id, question_state_data=question.question_state_data.to_dict(), language_code=question.language_code, version=question.version, linked_skill_ids=question.linked_skill_ids, question_state_data_schema_version=question.question_state_data_schema_version, inapplicable_skill_misconception_ids=question.inapplicable_skill_misconception_ids, next_content_id_index=question.next_content_id_index)
    model.commit(committer_id, commit_message, [{'cmd': question_domain.CMD_CREATE_NEW}])
    question.version += 1
    create_question_summary(question.id)
    opportunity_services.increment_question_counts(question.linked_skill_ids, 1)

def link_multiple_skills_for_question(user_id: str, question_id: str, skill_ids: List[str], skill_difficulties: List[float]) -> None:
    if False:
        i = 10
        return i + 15
    'Links multiple skill IDs to a question. To do that, it creates multiple\n    new QuestionSkillLink models. It also adds the skill ids to the\n    linked_skill_ids of the Question.\n\n    Args:\n        user_id: str. ID of the creator.\n        question_id: str. ID of the question linked to the skills.\n        skill_ids: list(str). ID of the skills to which the question is linked.\n        skill_difficulties: list(float). The difficulty of the question with\n            respect to a skill, represented by a float between\n            0 and 1 (inclusive).\n\n    Raises:\n        Exception. Number of elements in skill ids and skill difficulties\n            are different.\n    '
    if len(skill_ids) != len(skill_difficulties):
        raise Exception('Skill difficulties and skill ids should match. The lengths of the two lists are different.')
    question = get_question_by_id(question_id)
    new_question_skill_link_models = []
    for (index, skill_id) in enumerate(skill_ids):
        new_question_skill_link_models.append(question_models.QuestionSkillLinkModel.create(question_id, skill_id, skill_difficulties[index]))
    new_linked_skill_ids = copy.deepcopy(question.linked_skill_ids)
    new_linked_skill_ids.extend(skill_ids)
    _update_linked_skill_ids_of_question(user_id, question_id, list(set(new_linked_skill_ids)), question.linked_skill_ids)
    question_models.QuestionSkillLinkModel.put_multi_question_skill_links(new_question_skill_link_models)

def create_new_question_skill_link(user_id: str, question_id: str, skill_id: str, skill_difficulty: float) -> None:
    if False:
        i = 10
        return i + 15
    'Creates a new QuestionSkillLink model and adds the skill id\n    to the linked skill ids for the Question model.\n\n    Args:\n        user_id: str. ID of the creator.\n        question_id: str. ID of the question linked to the skill.\n        skill_id: str. ID of the skill to which the question is linked.\n        skill_difficulty: float. The difficulty between [0, 1] of the skill.\n    '
    question = get_question_by_id(question_id)
    question_skill_link_model = question_models.QuestionSkillLinkModel.create(question_id, skill_id, skill_difficulty)
    question_skill_link_model.update_timestamps()
    question_skill_link_model.put()
    if skill_id not in question.linked_skill_ids:
        new_linked_skill_ids = copy.deepcopy(question.linked_skill_ids)
        new_linked_skill_ids.append(skill_id)
        _update_linked_skill_ids_of_question(user_id, question_id, new_linked_skill_ids, question.linked_skill_ids)

def update_question_skill_link_difficulty(question_id: str, skill_id: str, new_difficulty: float) -> None:
    if False:
        i = 10
        return i + 15
    'Updates the difficulty value of question skill link.\n\n    Args:\n        question_id: str. ID of the question.\n        skill_id: str. ID of the skill.\n        new_difficulty: float. New difficulty value.\n\n    Raises:\n        Exception. Given question and skill are not linked.\n    '
    question_skill_link_id = question_models.QuestionSkillLinkModel.get_model_id(question_id, skill_id)
    question_skill_link_model = question_models.QuestionSkillLinkModel.get(question_skill_link_id, strict=False)
    if question_skill_link_model is None:
        raise Exception('The given question and skill are not linked.')
    question_skill_link_model.skill_difficulty = new_difficulty
    question_skill_link_model.update_timestamps()
    question_skill_link_model.put()

def _update_linked_skill_ids_of_question(user_id: str, question_id: str, new_linked_skill_ids: List[str], old_linked_skill_ids: List[str]) -> None:
    if False:
        print('Hello World!')
    'Updates the question linked_skill ids in the Question model.\n\n    Args:\n        user_id: str. ID of the creator.\n        question_id: str. ID of the question linked to the skill.\n        new_linked_skill_ids: list(str). New linked skill IDs of the question.\n        old_linked_skill_ids: list(str). Current linked skill IDs of the\n            question.\n    '
    change_dict: Dict[str, Union[str, List[str]]] = {'cmd': 'update_question_property', 'property_name': 'linked_skill_ids', 'new_value': new_linked_skill_ids, 'old_value': old_linked_skill_ids}
    change_list = [question_domain.QuestionChange(change_dict)]
    update_question(user_id, question_id, change_list, 'Updated linked skill ids')
    opportunity_services.update_skill_opportunities_on_question_linked_skills_change(old_linked_skill_ids, new_linked_skill_ids)

def delete_question_skill_link(user_id: str, question_id: str, skill_id: str) -> None:
    if False:
        return 10
    'Deleted a QuestionSkillLink model and removes the linked skill id\n    from the Question model of question_id.\n\n    Args:\n        user_id: str. ID of the creator.\n        question_id: str. ID of the question linked to the skill.\n        skill_id: str. ID of the skill to which the question is linked.\n    '
    question = get_question_by_id(question_id)
    new_linked_skill_ids = copy.deepcopy(question.linked_skill_ids)
    new_linked_skill_ids.remove(skill_id)
    question_skill_link_id = question_models.QuestionSkillLinkModel.get_model_id(question_id, skill_id)
    question_skill_link_model = question_models.QuestionSkillLinkModel.get(question_skill_link_id)
    if new_linked_skill_ids:
        _update_linked_skill_ids_of_question(user_id, question_id, new_linked_skill_ids, question.linked_skill_ids)
    else:
        delete_question(user_id, question_id)
    question_skill_link_model.delete()

def get_total_question_count_for_skill_ids(skill_ids: List[str]) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Returns the number of questions assigned to the given skill_ids.\n\n    Args:\n        skill_ids: list(str). Skill IDs for which the question count is\n            requested.\n\n    Returns:\n        int. The total number of questions assigned to the given skill_ids.\n    '
    question_skill_link_model = question_models.QuestionSkillLinkModel
    question_count = question_skill_link_model.get_total_question_count_for_skill_ids(skill_ids)
    return question_count

def get_questions_by_skill_ids(total_question_count: int, skill_ids: List[str], require_medium_difficulty: bool) -> List[question_domain.Question]:
    if False:
        print('Hello World!')
    'Returns constant number of questions linked to each given skill id.\n\n    Args:\n        total_question_count: int. The total number of questions to return.\n        skill_ids: list(str). The IDs of the skills to which the questions\n            should be linked.\n        require_medium_difficulty: bool. Indicates whether the returned\n            questions should be of medium difficulty.\n\n    Returns:\n        list(Question). The list containing an expected number of\n        total_question_count questions linked to each given skill id.\n        question count per skill will be total_question_count divided by\n        length of skill_ids, and it will be rounded up if not evenly\n        divisible. If not enough questions for one skill, simply return\n        all questions linked to it. The order of questions will follow the\n        order of given skill ids, and the order of questions for the same\n        skill is random when require_medium_difficulty is false, otherwise\n        the order is sorted by absolute value of the difference between\n        skill difficulty and the medium difficulty.\n\n    Raises:\n        Exception. Question count is higher than the maximum limit.\n    '
    if total_question_count > feconf.MAX_QUESTIONS_FETCHABLE_AT_ONE_TIME:
        raise Exception('Question count is too high, please limit the question count to %d.' % feconf.MAX_QUESTIONS_FETCHABLE_AT_ONE_TIME)
    if require_medium_difficulty:
        question_skill_link_models = question_models.QuestionSkillLinkModel.get_question_skill_links_based_on_difficulty_equidistributed_by_skill(total_question_count, skill_ids, constants.SKILL_DIFFICULTY_LABEL_TO_FLOAT[constants.SKILL_DIFFICULTY_MEDIUM])
    else:
        question_skill_link_models = question_models.QuestionSkillLinkModel.get_question_skill_links_equidistributed_by_skill(total_question_count, skill_ids)
    question_ids = [model.question_id for model in question_skill_link_models]
    questions_with_none = question_fetchers.get_questions_by_ids(question_ids)
    questions: List[question_domain.Question] = []
    for question in questions_with_none:
        assert question is not None
        questions.append(question)
    return questions

def get_new_question_id() -> str:
    if False:
        print('Hello World!')
    'Returns a new question id.\n\n    Returns:\n        str. A new question id.\n    '
    return question_models.QuestionModel.get_new_id('')

def add_question(committer_id: str, question: question_domain.Question) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Saves a new question.\n\n    Args:\n        committer_id: str. ID of the committer.\n        question: Question. Question to be saved.\n    '
    commit_message = 'New question created'
    create_new_question(committer_id, question, commit_message)

def delete_question(committer_id: str, question_id: str, force_deletion: bool=False) -> None:
    if False:
        return 10
    'Deletes the question with the given question_id.\n\n    Args:\n        committer_id: str. ID of the committer.\n        question_id: str. ID of the question.\n        force_deletion: bool. If true, the question and its history are fully\n            deleted and are unrecoverable. Otherwise, the question and all\n            its history are marked as deleted, but the corresponding models are\n            still retained in the datastore. This last option is the preferred\n            one.\n    '

    @transaction_services.run_in_transaction_wrapper
    def delete_question_model_transactional(question_id: str, committer_id: str, force_deletion: bool) -> None:
        if False:
            print('Hello World!')
        'Inner function that is to be done in a transaction.'
        question_model = question_models.QuestionModel.get_by_id(question_id)
        if question_model is not None:
            opportunity_services.increment_question_counts(question_model.linked_skill_ids, -1)
        question_models.QuestionModel.delete_multi([question_id], committer_id, feconf.COMMIT_MESSAGE_QUESTION_DELETED, force_deletion=force_deletion)
    delete_question_model_transactional(question_id, committer_id, force_deletion)
    question_summary_model = question_models.QuestionSummaryModel.get(question_id, strict=False)
    if question_summary_model is not None:
        question_summary_model.delete()

def get_question_skill_link_from_model(question_skill_link_model: question_models.QuestionSkillLinkModel, skill_description: str) -> question_domain.QuestionSkillLink:
    if False:
        i = 10
        return i + 15
    'Returns domain object representing the given question skill link model.\n\n    Args:\n        question_skill_link_model: QuestionSkillLinkModel. The question skill\n            link model loaded from the datastore.\n        skill_description: str. The description of skill linked to question.\n\n    Returns:\n        QuestionSkillLink. The domain object representing the question skill\n        link model.\n    '
    return question_domain.QuestionSkillLink(question_skill_link_model.question_id, question_skill_link_model.skill_id, skill_description, question_skill_link_model.skill_difficulty)

@overload
def get_question_by_id(question_id: str) -> question_domain.Question:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_question_by_id(question_id: str, *, strict: Literal[True]) -> question_domain.Question:
    if False:
        return 10
    ...

@overload
def get_question_by_id(question_id: str, *, strict: Literal[False]) -> Optional[question_domain.Question]:
    if False:
        return 10
    ...

def get_question_by_id(question_id: str, strict: bool=True) -> Optional[question_domain.Question]:
    if False:
        return 10
    'Returns a domain object representing a question.\n\n    Args:\n        question_id: str. ID of the question.\n        strict: bool. Whether to fail noisily if no question with the given\n            id exists in the datastore.\n\n    Returns:\n        Question or None. The domain object representing a question with the\n        given id, or None if it does not exist.\n    '
    question_model = question_models.QuestionModel.get(question_id, strict=strict)
    if question_model:
        question = question_fetchers.get_question_from_model(question_model)
        return question
    else:
        return None

def get_question_skill_links_of_skill(skill_id: str, skill_description: str) -> List[question_domain.QuestionSkillLink]:
    if False:
        print('Hello World!')
    'Returns a list of QuestionSkillLinks of\n    a particular skill ID.\n\n    Args:\n        skill_id: str. ID of the skill.\n        skill_description: str. Description of the skill.\n\n    Returns:\n        list(QuestionSkillLink). The list of question skill link\n        domain objects that are linked to the skill ID or an empty list\n        if the skill does not exist.\n    '
    question_skill_links = [get_question_skill_link_from_model(model, skill_description) for model in question_models.QuestionSkillLinkModel.get_models_by_skill_id(skill_id)]
    return question_skill_links

def get_skills_linked_to_question(question_id: str) -> List[skill_domain.Skill]:
    if False:
        i = 10
        return i + 15
    'Returns a list of skills linked to a particular question.\n\n    Args:\n        question_id: str. ID of the question.\n\n    Returns:\n        list(Skill). The list of skills that are linked to the question.\n    '
    question = get_question_by_id(question_id)
    skills = skill_fetchers.get_multi_skills(question.linked_skill_ids)
    return skills

def replace_skill_id_for_all_questions(curr_skill_id: str, curr_skill_description: str, new_skill_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Updates the skill ID of QuestionSkillLinkModels to the superseding\n    skill ID.\n\n    Args:\n        curr_skill_id: str. ID of the current skill.\n        curr_skill_description: str. Description of the current skill.\n        new_skill_id: str. ID of the superseding skill.\n    '
    old_question_skill_link_models = question_models.QuestionSkillLinkModel.get_models_by_skill_id(curr_skill_id)
    old_question_skill_links = get_question_skill_links_of_skill(curr_skill_id, curr_skill_description)
    new_question_skill_link_models = []
    question_ids = set()
    for question_skill_link in old_question_skill_links:
        question_ids.add(question_skill_link.question_id)
        new_question_skill_link_models.append(question_models.QuestionSkillLinkModel.create(question_skill_link.question_id, new_skill_id, question_skill_link.skill_difficulty))
    question_models.QuestionSkillLinkModel.delete_multi_question_skill_links(list(old_question_skill_link_models))
    question_models.QuestionSkillLinkModel.put_multi_question_skill_links(new_question_skill_link_models)
    old_questions = question_models.QuestionModel.get_multi(list(question_ids))
    new_questions = []
    for question in old_questions:
        assert question is not None
        new_question = copy.deepcopy(question)
        new_question.linked_skill_ids.remove(curr_skill_id)
        new_question.linked_skill_ids.append(new_skill_id)
        new_questions.append(new_question)
    question_models.QuestionModel.put_multi_questions(new_questions)

def get_displayable_question_skill_link_details(question_count: int, skill_ids: List[str], offset: int) -> Tuple[List[Optional[question_domain.QuestionSummary]], List[question_domain.MergedQuestionSkillLink]]:
    if False:
        while True:
            i = 10
    'Returns the list of question summaries and corresponding skill\n    descriptions linked to all the skills given by skill_ids.\n\n    Args:\n        question_count: int. The number of questions to fetch.\n        skill_ids: list(str). The ids of skills for which the linked questions\n            are to be retrieved.\n        offset: int. Number of query results to skip.\n\n    Raises:\n        Exception. Querying linked question summaries for more than 3 skills at\n            a time is not supported currently.\n\n    Returns:\n        list(QuestionSummary), list(MergedQuestionSkillLink).\n        The list of questions linked to the given skill ids, the list of\n        MergedQuestionSkillLink objects, keyed by question ID.\n    '
    if len(skill_ids) == 0:
        return ([], [])
    if len(skill_ids) > 3:
        raise Exception('Querying linked question summaries for more than 3 skills at a time is not supported currently.')
    question_skill_link_models = question_models.QuestionSkillLinkModel.get_question_skill_links_by_skill_ids(question_count, skill_ids, offset)
    question_ids = []
    grouped_skill_ids = []
    grouped_difficulties = []
    for question_skill_link in question_skill_link_models:
        if question_skill_link.question_id not in question_ids:
            question_ids.append(question_skill_link.question_id)
            grouped_skill_ids.append([question_skill_link.skill_id])
            grouped_difficulties.append([question_skill_link.skill_difficulty])
        else:
            grouped_skill_ids[-1].append(question_skill_link.skill_id)
            grouped_difficulties[-1].append(question_skill_link.skill_difficulty)
    merged_question_skill_links = []
    for (ind, skill_ids_list) in enumerate(grouped_skill_ids):
        skills = skill_models.SkillModel.get_multi(skill_ids_list)
        merged_question_skill_links.append(question_domain.MergedQuestionSkillLink(question_ids[ind], skill_ids_list, [skill.description if skill else None for skill in skills], grouped_difficulties[ind]))
    question_summaries = get_question_summaries_by_ids(question_ids)
    return (question_summaries, merged_question_skill_links)

def get_question_summaries_by_ids(question_ids: List[str]) -> List[Optional[question_domain.QuestionSummary]]:
    if False:
        while True:
            i = 10
    'Returns a list of domain objects representing question summaries.\n\n    Args:\n        question_ids: list(str). List of question ids.\n\n    Returns:\n        list(QuestionSummary|None). A list of domain objects representing\n        question summaries with the given ids or None when the id is not valid.\n    '
    question_summary_model_list = question_models.QuestionSummaryModel.get_multi(question_ids)
    question_summaries: List[Optional[question_domain.QuestionSummary]] = []
    for question_summary_model in question_summary_model_list:
        if question_summary_model is not None:
            question_summaries.append(get_question_summary_from_model(question_summary_model))
        else:
            question_summaries.append(None)
    return question_summaries

def apply_change_list(question_id: str, change_list: List[question_domain.QuestionChange]) -> question_domain.Question:
    if False:
        i = 10
        return i + 15
    'Applies a changelist to a pristine question and returns the result.\n\n    Args:\n        question_id: str. ID of the given question.\n        change_list: list(QuestionChange). A change list to be applied to the\n            given question. Each entry in change_list is a QuestionChange\n            object.\n\n    Returns:\n        Question. The resulting question domain object.\n\n    Raises:\n        Exception. The change list is not applicable to the question.\n    '
    question = get_question_by_id(question_id)
    question_property_inapplicable_skill_misconception_ids = question_domain.QUESTION_PROPERTY_INAPPLICABLE_SKILL_MISCONCEPTION_IDS
    try:
        for change in change_list:
            if change.cmd == question_domain.CMD_UPDATE_QUESTION_PROPERTY:
                if change.property_name == question_domain.QUESTION_PROPERTY_LANGUAGE_CODE:
                    update_language_code_cmd = cast(question_domain.UpdateQuestionPropertyLanguageCodeCmd, change)
                    question.update_language_code(update_language_code_cmd.new_value)
                elif change.property_name == question_domain.QUESTION_PROPERTY_QUESTION_STATE_DATA:
                    update_question_state_data_cmd = cast(question_domain.UpdateQuestionPropertyQuestionStateDataCmd, change)
                    state_domain_object = state_domain.State.from_dict(update_question_state_data_cmd.new_value)
                    question.update_question_state_data(state_domain_object)
                elif change.property_name == question_domain.QUESTION_PROPERTY_LINKED_SKILL_IDS:
                    update_linked_skill_ids_cmd = cast(question_domain.UpdateQuestionPropertyLinkedSkillIdsCmd, change)
                    question.update_linked_skill_ids(update_linked_skill_ids_cmd.new_value)
                elif change.property_name == question_property_inapplicable_skill_misconception_ids:
                    update_skill_misconception_ids_cmd = cast(question_domain.UpdateQuestionPropertySkillMisconceptionIdsCmd, change)
                    question.update_inapplicable_skill_misconception_ids(update_skill_misconception_ids_cmd.new_value)
                elif change.property_name == question_domain.QUESTION_PROPERTY_NEXT_CONTENT_ID_INDEX:
                    cmd = cast(question_domain.UpdateQuestionPropertyNextContentIdIndexCmd, change)
                    question.update_next_content_id_index(cmd.new_value)
        return question
    except Exception as e:
        logging.error('%s %s %s %s' % (e.__class__.__name__, e, question_id, change_list))
        raise e

def _save_question(committer_id: str, question: question_domain.Question, change_list: List[question_domain.QuestionChange], commit_message: str) -> None:
    if False:
        while True:
            i = 10
    'Validates a question and commits it to persistent storage. If\n    successful, increments the version number of the incoming question domain\n    object by 1.\n\n    Args:\n        committer_id: str. The id of the user who is performing the update\n            action.\n        question: Question. The domain object representing a question.\n        change_list: list(QuestionChange). A list of QuestionChange objects.\n            These changes are applied in sequence to produce the resulting\n            question.\n        commit_message: str. A description of changes made to the\n            question.\n\n    Raises:\n        Exception. Received an invalid change list.\n    '
    if not change_list:
        raise Exception('Unexpected error: received an invalid change list when trying to save question %s: %s' % (question.id, change_list))
    question.validate()
    question_model = question_models.QuestionModel.get(question.id)
    question_model.question_state_data = question.question_state_data.to_dict()
    question_model.language_code = question.language_code
    question_model.question_state_data_schema_version = question.question_state_data_schema_version
    question_model.linked_skill_ids = question.linked_skill_ids
    question_model.inapplicable_skill_misconception_ids = question.inapplicable_skill_misconception_ids
    question_model.next_content_id_index = question.next_content_id_index
    change_dicts = [change.to_dict() for change in change_list]
    question_model.commit(committer_id, commit_message, change_dicts)
    question.version += 1

def update_question(committer_id: str, question_id: str, change_list: List[question_domain.QuestionChange], commit_message: str, version: Optional[int]=None) -> None:
    if False:
        print('Hello World!')
    'Updates a question. Commits changes.\n\n    Args:\n        committer_id: str. The ID of the user who is performing the update\n            action.\n        question_id: str. The question ID.\n        change_list: list(QuestionChange). A list of QuestionChange objects.\n            These changes are applied in sequence to produce the resulting\n            question.\n        commit_message: str. A description of changes made to the\n            question.\n        version: int. Version of the question coming from payload while\n            updating question.\n\n    Raises:\n        ValueError. No commit message was provided.\n        Exception. Version of the incoming question and current question\n            differs.\n    '
    if not commit_message:
        raise ValueError('Expected a commit message, received none.')
    updated_question = apply_change_list(question_id, change_list)
    if version is not None and version < updated_question.version:
        raise Exception('Unexpected error: trying to update version %s of question from version %s. Please reload the page and try again.' % (version, updated_question.version))
    if version is not None and version > updated_question.version:
        raise Exception('Trying to update version %s of question from version %s, which is too old. Please reload the page and try again.' % (version, updated_question.version))
    _save_question(committer_id, updated_question, change_list, commit_message)
    create_question_summary(question_id)

def create_question_summary(question_id: str) -> None:
    if False:
        while True:
            i = 10
    'Creates and stores a summary of the given question.\n\n    Args:\n        question_id: str. ID of the question.\n    '
    question = get_question_by_id(question_id)
    question_summary = compute_summary_of_question(question)
    save_question_summary(question_summary)

def compute_summary_of_question(question: question_domain.Question) -> question_domain.QuestionSummary:
    if False:
        while True:
            i = 10
    'Create a QuestionSummary domain object for a given Question domain\n    object and return it.\n\n    Args:\n        question: Question. The question object for which the summary\n            is to be computed.\n\n    Returns:\n        QuestionSummary. The computed summary for the given question.\n\n    Raises:\n        Exception. No interaction_id found for the given question.\n        Exception. No data available for when the question was last_updated on.\n    '
    question_content = question.question_state_data.content.html
    answer_groups = question.question_state_data.interaction.answer_groups
    misconception_ids = []
    for answer_group in answer_groups:
        misconception_id = answer_group.to_dict()['tagged_skill_misconception_id']
        if misconception_id is not None:
            misconception_ids.append(misconception_id)
    misconception_ids.extend(question.inapplicable_skill_misconception_ids)
    interaction_id = question.question_state_data.interaction.id
    if interaction_id is None:
        raise Exception('No interaction_id found for the given question.')
    if question.created_on is None or question.last_updated is None:
        raise Exception('No data available for when the question was last_updated on.')
    question_summary = question_domain.QuestionSummary(question.id, question_content, misconception_ids, interaction_id, question.created_on, question.last_updated, question.version)
    return question_summary

def save_question_summary(question_summary: question_domain.QuestionSummary) -> None:
    if False:
        return 10
    'Save a question summary domain object as a QuestionSummaryModel\n    entity in the datastore.\n\n    Args:\n        question_summary: QuestionSummaryModel. The question summary object to\n            be saved in the datastore.\n    '
    question_summary_model = question_models.QuestionSummaryModel(id=question_summary.id, question_model_last_updated=question_summary.last_updated, question_model_created_on=question_summary.created_on, question_content=question_summary.question_content, misconception_ids=question_summary.misconception_ids, interaction_id=question_summary.interaction_id, version=question_summary.version)
    question_summary_model.update_timestamps()
    question_summary_model.put()

def get_question_summary_from_model(question_summary_model: question_models.QuestionSummaryModel) -> question_domain.QuestionSummary:
    if False:
        for i in range(10):
            print('nop')
    'Returns a domain object for an Oppia question summary given a\n    question summary model.\n\n    Args:\n        question_summary_model: QuestionSummaryModel. The QuestionSummary model\n            object to fetch corresponding QuestionSummary domain object.\n\n    Returns:\n        QuestionSummary. The domain object corresponding to the given question\n        summary model.\n    '
    return question_domain.QuestionSummary(question_summary_model.id, question_summary_model.question_content, question_summary_model.misconception_ids, question_summary_model.interaction_id, question_summary_model.question_model_created_on, question_summary_model.question_model_last_updated, question_summary_model.version)

def get_interaction_id_for_question(question_id: str) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    'Returns the interaction id for the given question.\n\n    Args:\n        question_id: str. ID of the question.\n\n    Returns:\n        str. The ID of the interaction of the question.\n\n    Raises:\n        Exception. The question does not exists of the ID question_id.\n    '
    question = get_question_by_id(question_id, strict=False)
    if question is None:
        raise Exception('No questions exists with the given question id.')
    return question.question_state_data.interaction.id

def untag_deleted_misconceptions(committer_id: str, skill_id: str, skill_description: str, deleted_skill_misconception_ids: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Untags deleted misconceptions from questions belonging\n    to a skill with the provided skill_id.\n\n    Args:\n        committer_id: str. The id of the user who triggered the update.\n        skill_id: str. The skill id.\n        skill_description: str. The description of the skill.\n        deleted_skill_misconception_ids: list(str). The skill misconception\n            ids of deleted misconceptions. The list items take the form\n            <skill_id>-<misconception_id>.\n    '
    question_skill_links = get_question_skill_links_of_skill(skill_id, skill_description)
    question_ids = [model.question_id for model in question_skill_links]
    questions_with_none = question_fetchers.get_questions_by_ids(question_ids)
    for question in questions_with_none:
        assert question is not None
        change_list = []
        inapplicable_skill_misconception_ids = question.inapplicable_skill_misconception_ids
        deleted_inapplicable_skill_misconception_ids = list(set(deleted_skill_misconception_ids) & set(inapplicable_skill_misconception_ids))
        if deleted_inapplicable_skill_misconception_ids:
            new_inapplicable_skill_misconception_ids = utils.compute_list_difference(question.inapplicable_skill_misconception_ids, deleted_inapplicable_skill_misconception_ids)
            change_list.append(question_domain.QuestionChange({'cmd': 'update_question_property', 'property_name': 'inapplicable_skill_misconception_ids', 'new_value': new_inapplicable_skill_misconception_ids, 'old_value': question.inapplicable_skill_misconception_ids}))
        old_question_state_data_dict = question.question_state_data.to_dict()
        answer_groups = list(question.question_state_data.interaction.answer_groups)
        for answer_group in answer_groups:
            tagged_skill_misconception_id = answer_group.to_dict()['tagged_skill_misconception_id']
            if tagged_skill_misconception_id in deleted_skill_misconception_ids:
                answer_group.tagged_skill_misconception_id = None
        question.question_state_data.interaction.answer_groups = answer_groups
        change_list.append(question_domain.QuestionChange({'cmd': 'update_question_property', 'property_name': 'question_state_data', 'new_value': question.question_state_data.to_dict(), 'old_value': old_question_state_data_dict}))
        update_question(committer_id, question.id, change_list, 'Untagged deleted skill misconception ids.')

def populate_question_model_fields(question_model: question_models.QuestionModel, question: question_domain.Question) -> question_models.QuestionModel:
    if False:
        print('Hello World!')
    'Populate question model with the data from question object.\n\n    Args:\n        question_model: QuestionModel. The model to populate.\n        question: Question. The question domain object which should be used to\n            populate the model.\n\n    Returns:\n        QuestionModel. Populated model.\n    '
    question_model.question_state_data = question.question_state_data.to_dict()
    question_model.question_state_data_schema_version = question.question_state_data_schema_version
    question_model.next_content_id_index = question.next_content_id_index
    question_model.language_code = question.language_code
    question_model.linked_skill_ids = question.linked_skill_ids
    question_model.inapplicable_skill_misconception_ids = question.inapplicable_skill_misconception_ids
    return question_model

def populate_question_summary_model_fields(question_summary_model: question_models.QuestionSummaryModel, question_summary: question_domain.QuestionSummary) -> question_models.QuestionSummaryModel:
    if False:
        while True:
            i = 10
    'Populate question summary model with the data from question summary\n    object.\n\n    Args:\n        question_summary_model: QuestionSummaryModel. The model to populate.\n        question_summary: QuestionSummary. The question summary domain object\n            which should be used to populate the model.\n\n    Returns:\n        QuestionSummaryModel. Populated model.\n    '
    question_summary_dict = {'question_model_last_updated': question_summary.last_updated, 'question_model_created_on': question_summary.created_on, 'question_content': question_summary.question_content, 'interaction_id': question_summary.interaction_id, 'version': question_summary.version, 'misconception_ids': question_summary.misconception_ids}
    if question_summary_model is not None:
        question_summary_model.populate(**question_summary_dict)
    else:
        question_summary_dict['id'] = question_summary.id
        question_summary_model = question_models.QuestionSummaryModel(**question_summary_dict)
    return question_summary_model