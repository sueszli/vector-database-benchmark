"""Controllers for the translation changes."""
from __future__ import annotations
import io
from core import feconf
from core.constants import constants
from core.controllers import acl_decorators
from core.controllers import base
from core.domain import fs_services
from core.domain import rights_domain
from core.domain import rights_manager
from core.domain import user_services
from mutagen import mp3
from typing import Dict, TypedDict

class AudioUploadHandlerNormalizedRequestDict(TypedDict):
    """Dict representation of AudioUploadHandler's
    normalized_request dictionary.
    """
    raw_audio_file: bytes

class AudioUploadHandler(base.BaseHandler[Dict[str, str], AudioUploadHandlerNormalizedRequestDict]):
    """Handles audio file uploads (to Google Cloud Storage in production, and
    to the local datastore in dev).
    """
    GET_HANDLER_ERROR_RETURN_TYPE = feconf.HANDLER_TYPE_JSON
    URL_PATH_ARGS_SCHEMAS = {'exploration_id': {'schema': {'type': 'basestring', 'validators': [{'id': 'is_regex_matched', 'regex_pattern': constants.ENTITY_ID_REGEX}]}}}
    HANDLER_ARGS_SCHEMAS = {'POST': {'raw_audio_file': {'schema': {'type': 'basestring', 'validators': [{'id': 'is_valid_audio_file'}]}}, 'filename': {'schema': {'type': 'basestring', 'validators': [{'id': 'is_regex_matched', 'regex_pattern': '[^\\s]+(\\.(?i)(mp3))$'}]}}}}
    _FILENAME_PREFIX = 'audio'

    @acl_decorators.can_voiceover_exploration
    def post(self, exploration_id: str) -> None:
        if False:
            while True:
                i = 10
        'Saves an audio file uploaded by a content creator.\n\n        Args:\n            exploration_id: str. The exploration ID.\n        '
        assert self.normalized_payload is not None
        assert self.normalized_request is not None
        raw_audio_file = self.normalized_request['raw_audio_file']
        filename = self.normalized_payload['filename']
        tempbuffer = io.BytesIO()
        tempbuffer.write(raw_audio_file)
        tempbuffer.seek(0)
        audio = mp3.MP3(tempbuffer)
        tempbuffer.close()
        mimetype = audio.mime[0]
        duration_secs = audio.info.length
        del audio
        fs = fs_services.GcsFileSystem(feconf.ENTITY_TYPE_EXPLORATION, exploration_id)
        fs.commit('%s/%s' % (self._FILENAME_PREFIX, filename), raw_audio_file, mimetype=mimetype)
        self.render_json({'filename': filename, 'duration_secs': duration_secs})

class StartedTranslationTutorialEventHandler(base.BaseHandler[Dict[str, str], Dict[str, str]]):
    """Records that this user has started the state translation tutorial."""
    GET_HANDLER_ERROR_RETURN_TYPE = feconf.HANDLER_TYPE_JSON
    URL_PATH_ARGS_SCHEMAS = {'exploration_id': {'schema': {'type': 'basestring', 'validators': [{'id': 'is_regex_matched', 'regex_pattern': constants.ENTITY_ID_REGEX}]}}}
    HANDLER_ARGS_SCHEMAS: Dict[str, Dict[str, str]] = {'POST': {}}

    @acl_decorators.can_play_exploration_as_logged_in_user
    def post(self, unused_exploration_id: str) -> None:
        if False:
            while True:
                i = 10
        'Records that the user has started the state translation tutorial.\n\n        unused_exploration_id: str. The unused exploration ID.\n        '
        assert self.user_id is not None
        user_services.record_user_started_state_translation_tutorial(self.user_id)
        self.render_json({})

class VoiceArtistManagementHandlerNormalizedPayloadDict(TypedDict):
    """Dict representation of VoiceArtistManagementHandler's
    normalized_payload dictionary.
    """
    username: str

class VoiceArtistManagementHandlerNormalizedRequestDict(TypedDict):
    """Dict representation of VoiceArtistManagementHandler's
    normalized_request dictionary.
    """
    voice_artist: str

class VoiceArtistManagementHandler(base.BaseHandler[VoiceArtistManagementHandlerNormalizedPayloadDict, VoiceArtistManagementHandlerNormalizedRequestDict]):
    """Handles assignment of voice artists."""
    GET_HANDLER_ERROR_RETURN_TYPE = feconf.HANDLER_TYPE_JSON
    URL_PATH_ARGS_SCHEMAS = {'entity_type': {'schema': {'type': 'basestring', 'choices': [feconf.ENTITY_TYPE_EXPLORATION]}}, 'entity_id': {'schema': {'type': 'basestring', 'validators': [{'id': 'is_regex_matched', 'regex_pattern': constants.ENTITY_ID_REGEX}]}}}
    HANDLER_ARGS_SCHEMAS = {'POST': {'username': {'schema': {'type': 'basestring', 'validators': [{'id': 'is_valid_username_string'}]}}}, 'DELETE': {'voice_artist': {'schema': {'type': 'basestring', 'validators': [{'id': 'is_valid_username_string'}]}}}}

    @acl_decorators.can_add_voice_artist
    def post(self, unused_entity_type: str, entity_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Assigns a voice artist role.\n\n        Args:\n            unused_entity_type: str. The unused entity type.\n            entity_id: str. The entity ID.\n        '
        assert self.normalized_payload is not None
        voice_artist = self.normalized_payload['username']
        voice_artist_id = user_services.get_user_id_from_username(voice_artist)
        if voice_artist_id is None:
            raise self.InvalidInputException('Sorry, we could not find the specified user.')
        rights_manager.assign_role_for_exploration(self.user, entity_id, voice_artist_id, rights_domain.ROLE_VOICE_ARTIST)
        self.render_json({})

    @acl_decorators.can_remove_voice_artist
    def delete(self, unused_entity_type: str, entity_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Removes the voice artist role from a user.\n\n        Args:\n            unused_entity_type: str. The unused entity type.\n            entity_id: str. The entity ID.\n        '
        assert self.normalized_request is not None
        voice_artist = self.normalized_request['voice_artist']
        voice_artist_id = user_services.get_user_id_from_username(voice_artist)
        assert voice_artist_id is not None
        rights_manager.deassign_role_for_exploration(self.user, entity_id, voice_artist_id)
        self.render_json({})