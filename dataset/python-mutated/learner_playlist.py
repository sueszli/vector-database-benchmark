"""Controllers for the learner playlist."""
from __future__ import annotations
from core.constants import constants
from core.controllers import acl_decorators
from core.controllers import base
from core.domain import learner_playlist_services
from core.domain import learner_progress_services
from typing import Dict, Optional, TypedDict

class LearnerPlaylistHandlerNormalizedPayloadDict(TypedDict):
    """Dict representation of LearnerPlaylistHandler's
    normalized_request dictionary.
    """
    index: Optional[int]

class LearnerPlaylistHandler(base.BaseHandler[LearnerPlaylistHandlerNormalizedPayloadDict, Dict[str, str]]):
    """Handles operations related to the learner playlist."""
    URL_PATH_ARGS_SCHEMAS = {'activity_type': {'schema': {'type': 'basestring', 'choices': [constants.ACTIVITY_TYPE_EXPLORATION, constants.ACTIVITY_TYPE_COLLECTION]}}, 'activity_id': {'schema': {'type': 'basestring'}}}
    HANDLER_ARGS_SCHEMAS = {'POST': {'index': {'schema': {'type': 'int'}, 'default_value': None}}, 'DELETE': {}}

    @acl_decorators.can_access_learner_dashboard
    def post(self, activity_type: str, activity_id: str) -> None:
        if False:
            i = 10
            return i + 15
        assert self.user_id is not None
        assert self.normalized_payload is not None
        position_to_be_inserted_in = self.normalized_payload.get('index')
        belongs_to_completed_or_incomplete_list = False
        playlist_limit_exceeded = False
        belongs_to_subscribed_activities = False
        if activity_type == constants.ACTIVITY_TYPE_EXPLORATION:
            (belongs_to_completed_or_incomplete_list, playlist_limit_exceeded, belongs_to_subscribed_activities) = learner_progress_services.add_exp_to_learner_playlist(self.user_id, activity_id, position_to_be_inserted=position_to_be_inserted_in)
        elif activity_type == constants.ACTIVITY_TYPE_COLLECTION:
            (belongs_to_completed_or_incomplete_list, playlist_limit_exceeded, belongs_to_subscribed_activities) = learner_progress_services.add_collection_to_learner_playlist(self.user_id, activity_id, position_to_be_inserted=position_to_be_inserted_in)
        self.values.update({'belongs_to_completed_or_incomplete_list': belongs_to_completed_or_incomplete_list, 'playlist_limit_exceeded': playlist_limit_exceeded, 'belongs_to_subscribed_activities': belongs_to_subscribed_activities})
        self.render_json(self.values)

    @acl_decorators.can_access_learner_dashboard
    def delete(self, activity_type: str, activity_id: str) -> None:
        if False:
            i = 10
            return i + 15
        assert self.user_id is not None
        if activity_type == constants.ACTIVITY_TYPE_EXPLORATION:
            learner_playlist_services.remove_exploration_from_learner_playlist(self.user_id, activity_id)
        elif activity_type == constants.ACTIVITY_TYPE_COLLECTION:
            learner_playlist_services.remove_collection_from_learner_playlist(self.user_id, activity_id)
        self.render_json(self.values)