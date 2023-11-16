from typing import Dict, Optional, Sequence, Union
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.http import HttpRequest, HttpResponse
from django.utils.translation import gettext as _
from typing_extensions import TypeAlias
from zerver.context_processors import get_valid_realm_from_request
from zerver.lib.compatibility import is_pronouns_field_type_supported
from zerver.lib.events import do_events_register
from zerver.lib.exceptions import JsonableError, MissingAuthenticationError
from zerver.lib.narrow_helpers import narrow_dataclasses_from_tuples
from zerver.lib.request import REQ, RequestNotes, has_request_variables
from zerver.lib.response import json_success
from zerver.lib.validator import check_bool, check_dict, check_int, check_list, check_string
from zerver.models import Stream, UserProfile

def _default_all_public_streams(user_profile: UserProfile, all_public_streams: Optional[bool]) -> bool:
    if False:
        i = 10
        return i + 15
    if all_public_streams is not None:
        return all_public_streams
    else:
        return user_profile.default_all_public_streams

def _default_narrow(user_profile: UserProfile, narrow: Sequence[Sequence[str]]) -> Sequence[Sequence[str]]:
    if False:
        while True:
            i = 10
    default_stream: Optional[Stream] = user_profile.default_events_register_stream
    if not narrow and default_stream is not None:
        narrow = [['stream', default_stream.name]]
    return narrow
NarrowT: TypeAlias = Sequence[Sequence[str]]

@has_request_variables
def events_register_backend(request: HttpRequest, maybe_user_profile: Union[UserProfile, AnonymousUser], apply_markdown: bool=REQ(default=False, json_validator=check_bool), client_gravatar_raw: Optional[bool]=REQ('client_gravatar', default=None, json_validator=check_bool), slim_presence: bool=REQ(default=False, json_validator=check_bool), all_public_streams: Optional[bool]=REQ(default=None, json_validator=check_bool), include_subscribers: bool=REQ(default=False, json_validator=check_bool), client_capabilities: Optional[Dict[str, bool]]=REQ(json_validator=check_dict([('notification_settings_null', check_bool)], [('bulk_message_deletion', check_bool), ('user_avatar_url_field_optional', check_bool), ('stream_typing_notifications', check_bool), ('user_settings_object', check_bool), ('linkifier_url_template', check_bool)], value_validator=check_bool), default=None), event_types: Optional[Sequence[str]]=REQ(json_validator=check_list(check_string), default=None), fetch_event_types: Optional[Sequence[str]]=REQ(json_validator=check_list(check_string), default=None), narrow: NarrowT=REQ(json_validator=check_list(check_list(check_string, length=2)), default=[]), queue_lifespan_secs: int=REQ(json_validator=check_int, default=0, documentation_pending=True)) -> HttpResponse:
    if False:
        return 10
    if client_gravatar_raw is None:
        client_gravatar = maybe_user_profile.is_authenticated
    else:
        client_gravatar = client_gravatar_raw
    if maybe_user_profile.is_authenticated:
        user_profile = maybe_user_profile
        spectator_requested_language = None
        assert isinstance(user_profile, UserProfile)
        realm = user_profile.realm
        include_streams = True
        if all_public_streams and (not user_profile.can_access_public_streams()):
            raise JsonableError(_('User not authorized for this query'))
        all_public_streams = _default_all_public_streams(user_profile, all_public_streams)
        narrow = _default_narrow(user_profile, narrow)
    else:
        user_profile = None
        realm = get_valid_realm_from_request(request)
        if not realm.allow_web_public_streams_access():
            raise MissingAuthenticationError
        if client_gravatar:
            raise JsonableError(_("Invalid '{key}' parameter for anonymous request").format(key='client_gravatar'))
        if include_subscribers:
            raise JsonableError(_("Invalid '{key}' parameter for anonymous request").format(key='include_subscribers'))
        spectator_requested_language = request.COOKIES.get(settings.LANGUAGE_COOKIE_NAME, realm.default_language)
        all_public_streams = False
        include_streams = False
    if client_capabilities is None:
        client_capabilities = {}
    client = RequestNotes.get_notes(request).client
    assert client is not None
    pronouns_field_type_supported = is_pronouns_field_type_supported(request.headers.get('User-Agent'))
    modern_narrow = narrow_dataclasses_from_tuples(narrow)
    ret = do_events_register(user_profile, realm, client, apply_markdown, client_gravatar, slim_presence, event_types, queue_lifespan_secs, all_public_streams, narrow=modern_narrow, include_subscribers=include_subscribers, include_streams=include_streams, client_capabilities=client_capabilities, fetch_event_types=fetch_event_types, spectator_requested_language=spectator_requested_language, pronouns_field_type_supported=pronouns_field_type_supported)
    return json_success(request, data=ret)