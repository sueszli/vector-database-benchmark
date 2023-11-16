"""Controllers for the practice sessions page."""
from __future__ import annotations
from core import feconf
from core.constants import constants
from core.controllers import acl_decorators
from core.controllers import base
from core.domain import skill_fetchers
from core.domain import topic_fetchers
from typing import Dict, List, TypedDict

class PracticeSessionsPage(base.BaseHandler[Dict[str, str], Dict[str, str]]):
    """Renders the practice sessions page."""
    URL_PATH_ARGS_SCHEMAS = {'classroom_url_fragment': constants.SCHEMA_FOR_CLASSROOM_URL_FRAGMENTS, 'topic_url_fragment': constants.SCHEMA_FOR_TOPIC_URL_FRAGMENTS}
    HANDLER_ARGS_SCHEMAS = {'GET': {'selected_subtopic_ids': {'schema': {'type': 'custom', 'obj_type': 'JsonEncodedInString'}}}}

    @acl_decorators.can_access_topic_viewer_page
    def get(self, _: str) -> None:
        if False:
            print('Hello World!')
        'Renders the practice session page.'
        self.render_template('practice-session-page.mainpage.html')

    def handle_exception(self, exception: BaseException, unused_debug_mode: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Handles exceptions raised by this handler.\n\n        Args:\n            exception: Exception. The exception raised by the handler.\n            unused_debug_mode: bool. Whether the app is running in debug mode.\n        '
        if isinstance(exception, self.InvalidInputException):
            (_, _, classroom_url_fragment, topic_url_fragment, _, _) = self.request.path.split('/')
            self.redirect('/learn/%s/%s/practice' % (classroom_url_fragment, topic_url_fragment))
            return
        super().handle_exception(exception, unused_debug_mode)

class PracticeSessionsPageDataHandlerNormalizedRequestDict(TypedDict):
    """Dict representation of PracticeSessionsPageDataHandler's
    normalized_request dictionary.
    """
    selected_subtopic_ids: List[int]

class PracticeSessionsPageDataHandler(base.BaseHandler[Dict[str, str], PracticeSessionsPageDataHandlerNormalizedRequestDict]):
    """Fetches relevant data for the practice sessions page."""
    GET_HANDLER_ERROR_RETURN_TYPE = feconf.HANDLER_TYPE_JSON
    URL_PATH_ARGS_SCHEMAS = {'classroom_url_fragment': constants.SCHEMA_FOR_CLASSROOM_URL_FRAGMENTS, 'topic_url_fragment': constants.SCHEMA_FOR_TOPIC_URL_FRAGMENTS}
    HANDLER_ARGS_SCHEMAS = {'GET': {'selected_subtopic_ids': {'schema': {'type': 'custom', 'obj_type': 'JsonEncodedInString'}}}}

    @acl_decorators.can_access_topic_viewer_page
    def get(self, topic_name: str) -> None:
        if False:
            return 10
        'Retrieves information about a topic.\n\n        Args:\n            topic_name: str. The topic name.\n\n        Raises:\n            PageNotFoundException. The page cannot be found.\n        '
        assert self.normalized_request is not None
        topic = topic_fetchers.get_topic_by_name(topic_name)
        selected_subtopic_ids = self.normalized_request['selected_subtopic_ids']
        selected_skill_ids = []
        for subtopic in topic.subtopics:
            if subtopic.id in selected_subtopic_ids:
                selected_skill_ids.extend(subtopic.skill_ids)
        try:
            skills = skill_fetchers.get_multi_skills(selected_skill_ids)
        except Exception as e:
            raise self.PageNotFoundException(e)
        skill_ids_to_descriptions_map = {}
        for skill in skills:
            skill_ids_to_descriptions_map[skill.id] = skill.description
        self.values.update({'topic_name': topic.name, 'skill_ids_to_descriptions_map': skill_ids_to_descriptions_map})
        self.render_json(self.values)