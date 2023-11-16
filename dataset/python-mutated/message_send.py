from email.headerregistry import Address
from typing import Dict, Iterable, Optional, Sequence, Union, cast
from django.core import validators
from django.core.exceptions import ValidationError
from django.http import HttpRequest, HttpResponse
from django.utils.translation import gettext as _
from zerver.actions.message_send import check_send_message, compute_irc_user_fullname, compute_jabber_user_fullname, create_mirror_user_if_needed, extract_private_recipients, extract_stream_indicator
from zerver.lib.exceptions import JsonableError
from zerver.lib.message import render_markdown
from zerver.lib.request import REQ, RequestNotes, has_request_variables
from zerver.lib.response import json_success
from zerver.lib.topic import REQ_topic
from zerver.lib.validator import check_string_in, to_float
from zerver.lib.zcommand import process_zcommands
from zerver.lib.zephyr import compute_mit_user_fullname
from zerver.models import Client, Message, RealmDomain, UserProfile, get_user_including_cross_realm

class InvalidMirrorInputError(Exception):
    pass

def create_mirrored_message_users(client: Client, user_profile: UserProfile, recipients: Iterable[str], sender: str, recipient_type_name: str) -> UserProfile:
    if False:
        print('Hello World!')
    sender_email = sender.strip().lower()
    referenced_users = {sender_email}
    if recipient_type_name == 'private':
        for email in recipients:
            referenced_users.add(email.lower())
    if client.name == 'zephyr_mirror':
        user_check = same_realm_zephyr_user
        fullname_function = compute_mit_user_fullname
    elif client.name == 'irc_mirror':
        user_check = same_realm_irc_user
        fullname_function = compute_irc_user_fullname
    elif client.name in ('jabber_mirror', 'JabberMirror'):
        user_check = same_realm_jabber_user
        fullname_function = compute_jabber_user_fullname
    else:
        raise InvalidMirrorInputError('Unrecognized mirroring client')
    for email in referenced_users:
        if not user_check(user_profile, email):
            raise InvalidMirrorInputError('At least one user cannot be mirrored')
    for email in referenced_users:
        create_mirror_user_if_needed(user_profile.realm, email, fullname_function)
    sender_user_profile = get_user_including_cross_realm(sender_email, user_profile.realm)
    return sender_user_profile

def same_realm_zephyr_user(user_profile: UserProfile, email: str) -> bool:
    if False:
        i = 10
        return i + 15
    try:
        validators.validate_email(email)
    except ValidationError:
        return False
    domain = Address(addr_spec=email).domain.lower()
    return user_profile.realm.is_zephyr_mirror_realm and RealmDomain.objects.filter(realm=user_profile.realm, domain=domain).exists()

def same_realm_irc_user(user_profile: UserProfile, email: str) -> bool:
    if False:
        while True:
            i = 10
    try:
        validators.validate_email(email)
    except ValidationError:
        return False
    domain = Address(addr_spec=email).domain.lower()
    if domain.startswith('irc.'):
        domain = domain[len('irc.'):]
    return RealmDomain.objects.filter(realm=user_profile.realm, domain=domain).exists()

def same_realm_jabber_user(user_profile: UserProfile, email: str) -> bool:
    if False:
        i = 10
        return i + 15
    try:
        validators.validate_email(email)
    except ValidationError:
        return False
    domain = Address(addr_spec=email).domain.lower()
    return RealmDomain.objects.filter(realm=user_profile.realm, domain=domain).exists()

@has_request_variables
def send_message_backend(request: HttpRequest, user_profile: UserProfile, req_type: str=REQ('type', str_validator=check_string_in(Message.API_RECIPIENT_TYPES)), req_to: Optional[str]=REQ('to', default=None), req_sender: Optional[str]=REQ('sender', default=None, documentation_pending=True), forged_str: Optional[str]=REQ('forged', default=None, documentation_pending=True), topic_name: Optional[str]=REQ_topic(), message_content: str=REQ('content'), widget_content: Optional[str]=REQ(default=None, documentation_pending=True), local_id: Optional[str]=REQ(default=None), queue_id: Optional[str]=REQ(default=None), time: Optional[float]=REQ(default=None, converter=to_float, documentation_pending=True)) -> HttpResponse:
    if False:
        return 10
    recipient_type_name = req_type
    if recipient_type_name == 'direct':
        recipient_type_name = 'private'
    message_to: Union[Sequence[int], Sequence[str]] = []
    if req_to is not None:
        if recipient_type_name == 'stream':
            stream_indicator = extract_stream_indicator(req_to)
            if isinstance(stream_indicator, int):
                message_to = [stream_indicator]
            else:
                message_to = [stream_indicator]
        else:
            message_to = extract_private_recipients(req_to)
    forged = forged_str is not None and forged_str in ['yes', 'true']
    client = RequestNotes.get_notes(request).client
    assert client is not None
    can_forge_sender = user_profile.can_forge_sender
    if forged and (not can_forge_sender):
        raise JsonableError(_('User not authorized for this query'))
    realm = user_profile.realm
    if client.name in ['zephyr_mirror', 'irc_mirror', 'jabber_mirror', 'JabberMirror']:
        if req_sender is None:
            raise JsonableError(_('Missing sender'))
        if recipient_type_name != 'private' and (not can_forge_sender):
            raise JsonableError(_('User not authorized for this query'))
        if not all((isinstance(to_item, str) for to_item in message_to)):
            raise JsonableError(_('Mirroring not allowed with recipient user IDs'))
        message_to = cast(Sequence[str], message_to)
        try:
            mirror_sender = create_mirrored_message_users(client, user_profile, message_to, req_sender, recipient_type_name)
        except InvalidMirrorInputError:
            raise JsonableError(_('Invalid mirrored message'))
        if client.name == 'zephyr_mirror' and (not user_profile.realm.is_zephyr_mirror_realm):
            raise JsonableError(_('Zephyr mirroring is not allowed in this organization'))
        sender = mirror_sender
    else:
        if req_sender is not None:
            raise JsonableError(_('Invalid mirrored message'))
        sender = user_profile
    data: Dict[str, int] = {}
    sent_message_result = check_send_message(sender, client, recipient_type_name, message_to, topic_name, message_content, forged=forged, forged_timestamp=time, forwarder_user_profile=user_profile, realm=realm, local_id=local_id, sender_queue_id=queue_id, widget_content=widget_content)
    data['id'] = sent_message_result.message_id
    if sent_message_result.automatic_new_visibility_policy:
        data['automatic_new_visibility_policy'] = sent_message_result.automatic_new_visibility_policy
    return json_success(request, data=data)

@has_request_variables
def zcommand_backend(request: HttpRequest, user_profile: UserProfile, command: str=REQ('command')) -> HttpResponse:
    if False:
        print('Hello World!')
    return json_success(request, data=process_zcommands(command, user_profile))

@has_request_variables
def render_message_backend(request: HttpRequest, user_profile: UserProfile, content: str=REQ()) -> HttpResponse:
    if False:
        print('Hello World!')
    message = Message()
    message.sender = user_profile
    message.realm = user_profile.realm
    message.content = content
    client = RequestNotes.get_notes(request).client
    assert client is not None
    message.sending_client = client
    rendering_result = render_markdown(message, content, realm=user_profile.realm)
    return json_success(request, data={'rendered': rendering_result.rendered_content})