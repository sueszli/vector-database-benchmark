"""Controllers for the diagnostic test player page."""
from __future__ import annotations
import collections
from core import feconf
from core import platform_feature_list
from core.constants import constants
from core.controllers import acl_decorators
from core.controllers import base
from core.domain import platform_feature_services
from core.domain import question_domain
from core.domain import question_services
from core.domain import topic_fetchers
from typing import Dict, List, TypedDict, cast

class DiagnosticTestPlayerPage(base.BaseHandler[Dict[str, str], Dict[str, str]]):
    """Renders the diagnostic test player page."""
    URL_PATH_ARGS_SCHEMAS: Dict[str, str] = {}
    HANDLER_ARGS_SCHEMAS: Dict[str, Dict[str, str]] = {'GET': {}}

    @acl_decorators.open_access
    def get(self) -> None:
        if False:
            while True:
                i = 10
        'Handles GET requests.'
        if platform_feature_services.is_feature_enabled(platform_feature_list.ParamNames.DIAGNOSTIC_TEST.value):
            self.render_template('diagnostic-test-player-page.mainpage.html')
        else:
            raise self.PageNotFoundException

def normalize_comma_separated_ids(comma_separated_ids: str) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Normalizes a string of comma-separated question IDs into a list of\n    question IDs.\n\n    Args:\n        comma_separated_ids: str. Comma separated question IDs.\n\n    Returns:\n        list(str). A list of question IDs.\n    '
    if not comma_separated_ids:
        return list([])
    return list(comma_separated_ids.split(','))

class DiagnosticTestQuestionsHandlerNormalizedRequestDict(TypedDict):
    """Dict representation of DiagnosticTestQuestionsHandler's
    normalized_request dictionary.
    """
    excluded_question_ids: List[str]

class DiagnosticTestQuestionsHandler(base.BaseHandler[Dict[str, str], DiagnosticTestQuestionsHandlerNormalizedRequestDict]):
    """Handler class to fetch the questions from the diagnostic test skills of
    the given topic ID.
    """
    GET_HANDLER_ERROR_RETURN_TYPE = feconf.HANDLER_TYPE_JSON
    URL_PATH_ARGS_SCHEMAS = {'topic_id': {'schema': {'type': 'basestring', 'validators': [{'id': 'is_regex_matched', 'regex_pattern': constants.ENTITY_ID_REGEX}]}}}
    HANDLER_ARGS_SCHEMAS = {'GET': {'excluded_question_ids': {'schema': {'type': 'object_dict', 'validation_method': normalize_comma_separated_ids}}}}

    @acl_decorators.open_access
    def get(self, topic_id: str) -> None:
        if False:
            print('Hello World!')
        'Retrieves diagnostic test questions for a specific topic.\n\n        Args:\n            topic_id: str. The ID of the topic.\n        '
        request_data = cast(DiagnosticTestQuestionsHandlerNormalizedRequestDict, self.normalized_request)
        excluded_question_ids: List[str] = request_data['excluded_question_ids']
        topic = topic_fetchers.get_topic_by_id(topic_id, strict=False)
        if topic is None:
            raise self.PageNotFoundException('No corresponding topic exists for the given topic ID.')
        diagnostic_test_skill_ids = topic.skill_ids_for_diagnostic_test
        skill_id_to_questions_map: Dict[str, List[question_domain.Question]] = collections.defaultdict(list)
        for skill_id in diagnostic_test_skill_ids:
            questions = question_services.get_questions_by_skill_ids(feconf.MAX_QUESTIONS_FETCHABLE_AT_ONE_TIME, [skill_id], require_medium_difficulty=True)
            for question in questions:
                if question.id in excluded_question_ids:
                    continue
                if len(skill_id_to_questions_map[skill_id]) < 2:
                    skill_id_to_questions_map[skill_id].append(question)
                    excluded_question_ids.append(question.id)
                else:
                    break
        skill_id_to_questions_dict: Dict[str, Dict[str, question_domain.QuestionDict]] = collections.defaultdict(dict)
        for (skill_id, linked_questions) in skill_id_to_questions_map.items():
            if len(linked_questions) < 2:
                continue
            skill_id_to_questions_dict[skill_id][feconf.DIAGNOSTIC_TEST_QUESTION_TYPE_MAIN] = linked_questions[0].to_dict()
            skill_id_to_questions_dict[skill_id][feconf.DIAGNOSTIC_TEST_QUESTION_TYPE_BACKUP] = linked_questions[1].to_dict()
        self.render_json({'skill_id_to_questions_dict': skill_id_to_questions_dict})