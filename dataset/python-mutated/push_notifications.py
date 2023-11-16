import asyncio
import base64
import copy
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union
import gcm
import lxml.html
import orjson
from django.conf import settings
from django.db import IntegrityError, transaction
from django.db.models import F, Q
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import override as override_language
from typing_extensions import TypeAlias, override
from analytics.lib.counts import COUNT_STATS, do_increment_logging_stat
from zerver.lib.avatar import absolute_avatar_url
from zerver.lib.emoji_utils import hex_codepoint_to_emoji
from zerver.lib.exceptions import ErrorCode, JsonableError
from zerver.lib.message import access_message, huddle_users
from zerver.lib.outgoing_http import OutgoingSession
from zerver.lib.remote_server import send_json_to_push_bouncer, send_to_push_bouncer
from zerver.lib.soft_deactivation import soft_reactivate_if_personal_notification
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.models import AbstractPushDeviceToken, ArchivedMessage, Message, NotificationTriggers, PushDeviceToken, Recipient, Stream, UserGroup, UserMessage, UserProfile, get_display_recipient, get_user_profile_by_id
if TYPE_CHECKING:
    import aioapns
logger = logging.getLogger(__name__)
if settings.ZILENCER_ENABLED:
    from zilencer.models import RemotePushDeviceToken, RemoteZulipServer
DeviceToken: TypeAlias = Union[PushDeviceToken, 'RemotePushDeviceToken']

def b64_to_hex(data: str) -> str:
    if False:
        print('Hello World!')
    return base64.b64decode(data).hex()

def hex_to_b64(data: str) -> str:
    if False:
        return 10
    return base64.b64encode(bytes.fromhex(data)).decode()

def get_message_stream_name_from_database(message: Message) -> str:
    if False:
        print('Hello World!')
    '\n    Never use this function outside of the push-notifications\n    codepath. Most of our code knows how to get streams\n    up front in a more efficient manner.\n    '
    stream_id = message.recipient.type_id
    return Stream.objects.get(id=stream_id).name

class UserPushIdentityCompat:
    """Compatibility class for supporting the transition from remote servers
    sending their UserProfile ids to the bouncer to sending UserProfile uuids instead.

    Until we can drop support for receiving user_id, we need this
    class, because a user's identity in the push notification context
    may be represented either by an id or uuid.
    """

    def __init__(self, user_id: Optional[int]=None, user_uuid: Optional[str]=None) -> None:
        if False:
            return 10
        assert user_id is not None or user_uuid is not None
        self.user_id = user_id
        self.user_uuid = user_uuid

    def filter_q(self) -> Q:
        if False:
            i = 10
            return i + 15
        '\n        This aims to support correctly querying for RemotePushDeviceToken.\n        If only one of (user_id, user_uuid) is provided, the situation is trivial,\n        If both are provided, we want to query for tokens matching EITHER the\n        uuid or the id - because the user may have devices with old registrations,\n        so user_id-based, as well as new registration with uuid. Notifications\n        naturally should be sent to both.\n        '
        if self.user_id is not None and self.user_uuid is None:
            return Q(user_id=self.user_id)
        elif self.user_uuid is not None and self.user_id is None:
            return Q(user_uuid=self.user_uuid)
        else:
            assert self.user_id is not None and self.user_uuid is not None
            return Q(user_uuid=self.user_uuid) | Q(user_id=self.user_id)

    @override
    def __str__(self) -> str:
        if False:
            print('Hello World!')
        result = ''
        if self.user_id is not None:
            result += f'<id:{self.user_id}>'
        if self.user_uuid is not None:
            result += f'<uuid:{self.user_uuid}>'
        return result

    @override
    def __eq__(self, other: object) -> bool:
        if False:
            return 10
        if isinstance(other, UserPushIdentityCompat):
            return self.user_id == other.user_id and self.user_uuid == other.user_uuid
        return False

@dataclass
class APNsContext:
    apns: 'aioapns.APNs'
    loop: asyncio.AbstractEventLoop

@lru_cache(maxsize=None)
def get_apns_context() -> Optional[APNsContext]:
    if False:
        for i in range(10):
            print('nop')
    import aioapns
    if settings.APNS_CERT_FILE is None:
        return None
    loop = asyncio.new_event_loop()

    async def err_func(request: aioapns.NotificationRequest, result: aioapns.common.NotificationResult) -> None:
        pass

    async def make_apns() -> aioapns.APNs:
        return aioapns.APNs(client_cert=settings.APNS_CERT_FILE, topic=settings.APNS_TOPIC, max_connection_attempts=APNS_MAX_RETRIES, use_sandbox=settings.APNS_SANDBOX, err_func=err_func)
    apns = loop.run_until_complete(make_apns())
    return APNsContext(apns=apns, loop=loop)

def apns_enabled() -> bool:
    if False:
        print('Hello World!')
    return settings.APNS_CERT_FILE is not None

def modernize_apns_payload(data: Mapping[str, Any]) -> Mapping[str, Any]:
    if False:
        while True:
            i = 10
    "Take a payload in an unknown Zulip version's format, and return in current format."
    if 'message_ids' in data:
        return {'alert': data['alert'], 'badge': 0, 'custom': {'zulip': {'message_ids': data['message_ids']}}}
    else:
        return data
APNS_MAX_RETRIES = 3

def send_apple_push_notification(user_identity: UserPushIdentityCompat, devices: Sequence[DeviceToken], payload_data: Mapping[str, Any], remote: Optional['RemoteZulipServer']=None) -> int:
    if False:
        return 10
    if not devices:
        return 0
    import aioapns
    import aioapns.exceptions
    apns_context = get_apns_context()
    if apns_context is None:
        logger.debug('APNs: Dropping a notification because nothing configured.  Set PUSH_NOTIFICATION_BOUNCER_URL (or APNS_CERT_FILE).')
        return 0
    if remote:
        assert settings.ZILENCER_ENABLED
        DeviceTokenClass: Type[AbstractPushDeviceToken] = RemotePushDeviceToken
    else:
        DeviceTokenClass = PushDeviceToken
    if remote:
        logger.info('APNs: Sending notification for remote user %s:%s to %d devices', remote.uuid, user_identity, len(devices))
    else:
        logger.info('APNs: Sending notification for local user %s to %d devices', user_identity, len(devices))
    payload_data = dict(modernize_apns_payload(payload_data))
    message = {**payload_data.pop('custom', {}), 'aps': payload_data}
    for device in devices:
        if device.ios_app_id is None:
            logger.error('APNs: Missing ios_app_id for user %s device %s', user_identity, device.token)

    async def send_all_notifications() -> Iterable[Tuple[DeviceToken, Union[aioapns.common.NotificationResult, BaseException]]]:
        requests = [aioapns.NotificationRequest(device_token=device.token, message=message, time_to_live=24 * 3600) for device in devices]
        results = await asyncio.gather(*(apns_context.apns.send_notification(request) for request in requests), return_exceptions=True)
        return zip(devices, results)
    results = apns_context.loop.run_until_complete(send_all_notifications())
    successfully_sent_count = 0
    for (device, result) in results:
        if isinstance(result, aioapns.exceptions.ConnectionError):
            logger.error('APNs: ConnectionError sending for user %s to device %s; check certificate expiration', user_identity, device.token)
        elif isinstance(result, BaseException):
            logger.error('APNs: Error sending for user %s to device %s', user_identity, device.token, exc_info=result)
        elif result.is_successful:
            successfully_sent_count += 1
            logger.info('APNs: Success sending for user %s to device %s', user_identity, device.token)
        elif result.description in ['Unregistered', 'BadDeviceToken', 'DeviceTokenNotForTopic']:
            logger.info('APNs: Removing invalid/expired token %s (%s)', device.token, result.description)
            DeviceTokenClass._default_manager.filter(token=device.token, kind=DeviceTokenClass.APNS).delete()
        else:
            logger.warning('APNs: Failed to send for user %s to device %s: %s', user_identity, device.token, result.description)
    return successfully_sent_count

class FCMSession(OutgoingSession):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(role='fcm', timeout=5)

def make_gcm_client() -> gcm.GCM:
    if False:
        print('Hello World!')
    gcm.gcm.GCM_URL = 'https://fcm.googleapis.com/fcm/send'
    return gcm.GCM(settings.ANDROID_GCM_API_KEY)
if settings.ANDROID_GCM_API_KEY:
    gcm_client = make_gcm_client()
else:
    gcm_client = None

def gcm_enabled() -> bool:
    if False:
        for i in range(10):
            print('nop')
    return gcm_client is not None

def send_android_push_notification_to_user(user_profile: UserProfile, data: Dict[str, Any], options: Dict[str, Any]) -> None:
    if False:
        while True:
            i = 10
    devices = list(PushDeviceToken.objects.filter(user=user_profile, kind=PushDeviceToken.GCM))
    send_android_push_notification(UserPushIdentityCompat(user_id=user_profile.id), devices, data, options)

def parse_gcm_options(options: Dict[str, Any], data: Dict[str, Any]) -> str:
    if False:
        while True:
            i = 10
    "\n    Parse GCM options, supplying defaults, and raising an error if invalid.\n\n    The options permitted here form part of the Zulip notification\n    bouncer's API.  They are:\n\n    `priority`: Passed through to GCM; see upstream doc linked below.\n        Zulip servers should always set this; when unset, we guess a value\n        based on the behavior of old server versions.\n\n    Including unrecognized options is an error.\n\n    For details on options' semantics, see this GCM upstream doc:\n      https://firebase.google.com/docs/cloud-messaging/http-server-ref\n\n    Returns `priority`.\n    "
    priority = options.pop('priority', None)
    if priority is None:
        if data.get('event') == 'message':
            priority = 'high'
        else:
            priority = 'normal'
    if priority not in ('normal', 'high'):
        raise JsonableError(_('Invalid GCM option to bouncer: priority {priority!r}').format(priority=priority))
    if options:
        raise JsonableError(_('Invalid GCM options to bouncer: {options}').format(options=orjson.dumps(options).decode()))
    return priority

def send_android_push_notification(user_identity: UserPushIdentityCompat, devices: Sequence[DeviceToken], data: Dict[str, Any], options: Dict[str, Any], remote: Optional['RemoteZulipServer']=None) -> int:
    if False:
        i = 10
        return i + 15
    "\n    Send a GCM message to the given devices.\n\n    See https://firebase.google.com/docs/cloud-messaging/http-server-ref\n    for the GCM upstream API which this talks to.\n\n    data: The JSON object (decoded) to send as the 'data' parameter of\n        the GCM message.\n    options: Additional options to control the GCM message sent.\n        For details, see `parse_gcm_options`.\n    "
    if not devices:
        return 0
    if not gcm_client:
        logger.debug('Skipping sending a GCM push notification since PUSH_NOTIFICATION_BOUNCER_URL and ANDROID_GCM_API_KEY are both unset')
        return 0
    if remote:
        logger.info('GCM: Sending notification for remote user %s:%s to %d devices', remote.uuid, user_identity, len(devices))
    else:
        logger.info('GCM: Sending notification for local user %s to %d devices', user_identity, len(devices))
    reg_ids = [device.token for device in devices]
    priority = parse_gcm_options(options, data)
    try:
        res = gcm_client.json_request(registration_ids=reg_ids, priority=priority, data=data, retries=2, session=FCMSession())
    except OSError:
        logger.warning('Error while pushing to GCM', exc_info=True)
        return 0
    successfully_sent_count = 0
    if res and 'success' in res:
        for (reg_id, msg_id) in res['success'].items():
            logger.info('GCM: Sent %s as %s', reg_id, msg_id)
        successfully_sent_count = len(res['success'].keys())
    if remote:
        assert settings.ZILENCER_ENABLED
        DeviceTokenClass: Type[AbstractPushDeviceToken] = RemotePushDeviceToken
    else:
        DeviceTokenClass = PushDeviceToken
    if 'canonical' in res:
        for (reg_id, new_reg_id) in res['canonical'].items():
            if reg_id == new_reg_id:
                logger.warning('GCM: Got canonical ref but it already matches our ID %s!', reg_id)
            elif not DeviceTokenClass._default_manager.filter(token=new_reg_id, kind=DeviceTokenClass.GCM).exists():
                logger.warning('GCM: Got canonical ref %s replacing %s but new ID not registered! Updating.', new_reg_id, reg_id)
                DeviceTokenClass._default_manager.filter(token=reg_id, kind=DeviceTokenClass.GCM).update(token=new_reg_id)
            else:
                logger.info('GCM: Got canonical ref %s, dropping %s', new_reg_id, reg_id)
                DeviceTokenClass._default_manager.filter(token=reg_id, kind=DeviceTokenClass.GCM).delete()
    if 'errors' in res:
        for (error, reg_ids) in res['errors'].items():
            if error in ['NotRegistered', 'InvalidRegistration']:
                for reg_id in reg_ids:
                    logger.info('GCM: Removing %s', reg_id)
                    DeviceTokenClass._default_manager.filter(token=reg_id, kind=DeviceTokenClass.GCM).delete()
            else:
                for reg_id in reg_ids:
                    logger.warning('GCM: Delivery to %s failed: %s', reg_id, error)
    return successfully_sent_count

def uses_notification_bouncer() -> bool:
    if False:
        while True:
            i = 10
    return settings.PUSH_NOTIFICATION_BOUNCER_URL is not None

def send_notifications_to_bouncer(user_profile: UserProfile, apns_payload: Dict[str, Any], gcm_payload: Dict[str, Any], gcm_options: Dict[str, Any]) -> Tuple[int, int]:
    if False:
        i = 10
        return i + 15
    post_data = {'user_uuid': str(user_profile.uuid), 'user_id': user_profile.id, 'realm_uuid': str(user_profile.realm.uuid), 'apns_payload': apns_payload, 'gcm_payload': gcm_payload, 'gcm_options': gcm_options}
    response_data = send_json_to_push_bouncer('POST', 'push/notify', post_data)
    assert isinstance(response_data['total_android_devices'], int)
    assert isinstance(response_data['total_apple_devices'], int)
    (total_android_devices, total_apple_devices) = (response_data['total_android_devices'], response_data['total_apple_devices'])
    do_increment_logging_stat(user_profile.realm, COUNT_STATS['mobile_pushes_sent::day'], None, timezone_now(), increment=total_android_devices + total_apple_devices)
    return (total_android_devices, total_apple_devices)

def add_push_device_token(user_profile: UserProfile, token_str: str, kind: int, ios_app_id: Optional[str]=None) -> PushDeviceToken:
    if False:
        print('Hello World!')
    logger.info('Registering push device: %d %r %d %r', user_profile.id, token_str, kind, ios_app_id)
    try:
        with transaction.atomic():
            token = PushDeviceToken.objects.create(user_id=user_profile.id, kind=kind, token=token_str, ios_app_id=ios_app_id, last_updated=timezone_now())
    except IntegrityError:
        token = PushDeviceToken.objects.get(user_id=user_profile.id, kind=kind, token=token_str)
    if uses_notification_bouncer():
        post_data = {'server_uuid': settings.ZULIP_ORG_ID, 'user_uuid': str(user_profile.uuid), 'user_id': str(user_profile.id), 'token': token_str, 'token_kind': kind}
        if kind == PushDeviceToken.APNS:
            post_data['ios_app_id'] = ios_app_id
        logger.info('Sending new push device to bouncer: %r', post_data)
        send_to_push_bouncer('POST', 'push/register', post_data)
    return token

def remove_push_device_token(user_profile: UserProfile, token_str: str, kind: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    try:
        token = PushDeviceToken.objects.get(token=token_str, kind=kind, user=user_profile)
        token.delete()
    except PushDeviceToken.DoesNotExist:
        if not uses_notification_bouncer():
            raise JsonableError(_('Token does not exist'))
    if uses_notification_bouncer():
        post_data = {'server_uuid': settings.ZULIP_ORG_ID, 'user_uuid': str(user_profile.uuid), 'user_id': user_profile.id, 'token': token_str, 'token_kind': kind}
        send_to_push_bouncer('POST', 'push/unregister', post_data)

def clear_push_device_tokens(user_profile_id: int) -> None:
    if False:
        print('Hello World!')
    if uses_notification_bouncer():
        user_uuid = str(get_user_profile_by_id(user_profile_id).uuid)
        post_data = {'server_uuid': settings.ZULIP_ORG_ID, 'user_uuid': user_uuid, 'user_id': user_profile_id}
        send_to_push_bouncer('POST', 'push/unregister/all', post_data)
        return
    PushDeviceToken.objects.filter(user_id=user_profile_id).delete()

def push_notifications_enabled() -> bool:
    if False:
        for i in range(10):
            print('nop')
    'True just if this server has configured a way to send push notifications.'
    if uses_notification_bouncer() and settings.ZULIP_ORG_KEY is not None and (settings.ZULIP_ORG_ID is not None):
        return True
    if settings.DEVELOPMENT and (apns_enabled() or gcm_enabled()):
        return True
    elif apns_enabled() and gcm_enabled():
        return True
    return False

def initialize_push_notifications() -> None:
    if False:
        print('Hello World!')
    if not push_notifications_enabled():
        if settings.DEVELOPMENT and (not settings.TEST_SUITE):
            return
        logger.warning('Mobile push notifications are not configured.\n  See https://zulip.readthedocs.io/en/latest/production/mobile-push-notifications.html')

def get_mobile_push_content(rendered_content: str) -> str:
    if False:
        i = 10
        return i + 15

    def get_text(elem: lxml.html.HtmlElement) -> str:
        if False:
            return 10
        classes = elem.get('class', '')
        if 'emoji' in classes:
            match = re.search('emoji-(?P<emoji_code>\\S+)', classes)
            if match:
                emoji_code = match.group('emoji_code')
                return hex_codepoint_to_emoji(emoji_code)
        if elem.tag == 'img':
            return elem.get('alt', '')
        if elem.tag == 'blockquote':
            return ''
        return elem.text or ''

    def format_as_quote(quote_text: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ''.join((f'> {line}\n' for line in quote_text.splitlines() if line))

    def render_olist(ol: lxml.html.HtmlElement) -> str:
        if False:
            return 10
        items = []
        counter = int(ol.get('start')) if ol.get('start') else 1
        nested_levels = sum((1 for ancestor in ol.iterancestors('ol')))
        indent = '\n' + '  ' * nested_levels if nested_levels else ''
        for li in ol:
            items.append(indent + str(counter) + '. ' + process(li).strip())
            counter += 1
        return '\n'.join(items)

    def render_spoiler(elem: lxml.html.HtmlElement) -> str:
        if False:
            return 10
        header = elem.find_class('spoiler-header')[0]
        text = process(header).strip()
        if len(text) == 0:
            return '(…)\n'
        return f'{text} (…)\n'

    def process(elem: lxml.html.HtmlElement) -> str:
        if False:
            for i in range(10):
                print('nop')
        plain_text = ''
        if elem.tag == 'ol':
            plain_text = render_olist(elem)
        elif 'spoiler-block' in elem.get('class', ''):
            plain_text += render_spoiler(elem)
        else:
            plain_text = get_text(elem)
            sub_text = ''
            for child in elem:
                sub_text += process(child)
            if elem.tag == 'blockquote':
                sub_text = format_as_quote(sub_text)
            plain_text += sub_text
            plain_text += elem.tail or ''
        return plain_text
    if settings.PUSH_NOTIFICATION_REDACT_CONTENT:
        return '*' + _('This organization has disabled including message content in mobile push notifications') + '*'
    elem = lxml.html.fragment_fromstring(rendered_content, create_parent=True)
    plain_text = process(elem)
    return plain_text

def truncate_content(content: str) -> Tuple[str, bool]:
    if False:
        while True:
            i = 10
    if len(content) <= 200:
        return (content, False)
    return (content[:200] + '…', True)

def get_base_payload(user_profile: UserProfile) -> Dict[str, Any]:
    if False:
        return 10
    'Common fields for all notification payloads.'
    data: Dict[str, Any] = {}
    data['server'] = settings.EXTERNAL_HOST
    data['realm_id'] = user_profile.realm.id
    data['realm_uri'] = user_profile.realm.uri
    data['user_id'] = user_profile.id
    return data

def get_message_payload(user_profile: UserProfile, message: Message, mentioned_user_group_id: Optional[int]=None, mentioned_user_group_name: Optional[str]=None) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    'Common fields for `message` payloads, for all platforms.'
    data = get_base_payload(user_profile)
    data['sender_id'] = message.sender.id
    data['sender_email'] = message.sender.email
    data['time'] = datetime_to_timestamp(message.date_sent)
    if mentioned_user_group_id is not None:
        assert mentioned_user_group_name is not None
        data['mentioned_user_group_id'] = mentioned_user_group_id
        data['mentioned_user_group_name'] = mentioned_user_group_name
    if message.recipient.type == Recipient.STREAM:
        data['recipient_type'] = 'stream'
        data['stream'] = get_message_stream_name_from_database(message)
        data['stream_id'] = message.recipient.type_id
        data['topic'] = message.topic_name()
    elif message.recipient.type == Recipient.HUDDLE:
        data['recipient_type'] = 'private'
        data['pm_users'] = huddle_users(message.recipient.id)
    else:
        data['recipient_type'] = 'private'
    return data

def get_apns_alert_title(message: Message) -> str:
    if False:
        i = 10
        return i + 15
    '\n    On an iOS notification, this is the first bolded line.\n    '
    if message.recipient.type == Recipient.HUDDLE:
        recipients = get_display_recipient(message.recipient)
        assert isinstance(recipients, list)
        return ', '.join(sorted((r['full_name'] for r in recipients)))
    elif message.is_stream_message():
        stream_name = get_message_stream_name_from_database(message)
        return f'#{stream_name} > {message.topic_name()}'
    return message.sender.full_name

def get_apns_alert_subtitle(message: Message, trigger: str, mentioned_user_group_name: Optional[str]=None) -> str:
    if False:
        while True:
            i = 10
    '\n    On an iOS notification, this is the second bolded line.\n    '
    if trigger == NotificationTriggers.MENTION:
        if mentioned_user_group_name is not None:
            return _('{full_name} mentioned @{user_group_name}:').format(full_name=message.sender.full_name, user_group_name=mentioned_user_group_name)
        else:
            return _('{full_name} mentioned you:').format(full_name=message.sender.full_name)
    elif trigger in (NotificationTriggers.TOPIC_WILDCARD_MENTION_IN_FOLLOWED_TOPIC, NotificationTriggers.STREAM_WILDCARD_MENTION_IN_FOLLOWED_TOPIC, NotificationTriggers.TOPIC_WILDCARD_MENTION, NotificationTriggers.STREAM_WILDCARD_MENTION):
        return _('{full_name} mentioned everyone:').format(full_name=message.sender.full_name)
    elif message.recipient.type == Recipient.PERSONAL:
        return ''
    return message.sender.full_name + ':'

def get_apns_badge_count(user_profile: UserProfile, read_messages_ids: Optional[Sequence[int]]=[]) -> int:
    if False:
        for i in range(10):
            print('nop')
    return 0

def get_apns_badge_count_future(user_profile: UserProfile, read_messages_ids: Optional[Sequence[int]]=[]) -> int:
    if False:
        i = 10
        return i + 15
    return UserMessage.objects.filter(user_profile=user_profile).extra(where=[UserMessage.where_active_push_notification()]).exclude(message_id__in=read_messages_ids).count()

def get_message_payload_apns(user_profile: UserProfile, message: Message, trigger: str, mentioned_user_group_id: Optional[int]=None, mentioned_user_group_name: Optional[str]=None) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    'A `message` payload for iOS, via APNs.'
    zulip_data = get_message_payload(user_profile, message, mentioned_user_group_id, mentioned_user_group_name)
    zulip_data.update(message_ids=[message.id])
    assert message.rendered_content is not None
    with override_language(user_profile.default_language):
        (content, _) = truncate_content(get_mobile_push_content(message.rendered_content))
        apns_data = {'alert': {'title': get_apns_alert_title(message), 'subtitle': get_apns_alert_subtitle(message, trigger, mentioned_user_group_name), 'body': content}, 'sound': 'default', 'badge': get_apns_badge_count(user_profile), 'custom': {'zulip': zulip_data}}
    return apns_data

def get_message_payload_gcm(user_profile: UserProfile, message: Message, mentioned_user_group_id: Optional[int]=None, mentioned_user_group_name: Optional[str]=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if False:
        print('Hello World!')
    'A `message` payload + options, for Android via GCM/FCM.'
    data = get_message_payload(user_profile, message, mentioned_user_group_id, mentioned_user_group_name)
    assert message.rendered_content is not None
    with override_language(user_profile.default_language):
        (content, truncated) = truncate_content(get_mobile_push_content(message.rendered_content))
        data.update(event='message', zulip_message_id=message.id, content=content, content_truncated=truncated, sender_full_name=message.sender.full_name, sender_avatar_url=absolute_avatar_url(message.sender))
    gcm_options = {'priority': 'high'}
    return (data, gcm_options)

def get_remove_payload_gcm(user_profile: UserProfile, message_ids: List[int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if False:
        print('Hello World!')
    'A `remove` payload + options, for Android via GCM/FCM.'
    gcm_payload = get_base_payload(user_profile)
    gcm_payload.update(event='remove', zulip_message_ids=','.join((str(id) for id in message_ids)), zulip_message_id=message_ids[0])
    gcm_options = {'priority': 'normal'}
    return (gcm_payload, gcm_options)

def get_remove_payload_apns(user_profile: UserProfile, message_ids: List[int]) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    zulip_data = get_base_payload(user_profile)
    zulip_data.update(event='remove', zulip_message_ids=','.join((str(id) for id in message_ids)))
    apns_data = {'badge': get_apns_badge_count(user_profile, message_ids), 'custom': {'zulip': zulip_data}}
    return apns_data

def handle_remove_push_notification(user_profile_id: int, message_ids: List[int]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'This should be called when a message that previously had a\n    mobile push notification executed is read.  This triggers a push to the\n    mobile app, when the message is read on the server, to remove the\n    message from the notification.\n    '
    if not push_notifications_enabled():
        return
    user_profile = get_user_profile_by_id(user_profile_id)
    MAX_APNS_MESSAGE_IDS = 200
    truncated_message_ids = sorted(message_ids)[-MAX_APNS_MESSAGE_IDS:]
    (gcm_payload, gcm_options) = get_remove_payload_gcm(user_profile, truncated_message_ids)
    apns_payload = get_remove_payload_apns(user_profile, truncated_message_ids)
    if uses_notification_bouncer():
        send_notifications_to_bouncer(user_profile, apns_payload, gcm_payload, gcm_options)
    else:
        user_identity = UserPushIdentityCompat(user_id=user_profile_id)
        android_devices = list(PushDeviceToken.objects.filter(user=user_profile, kind=PushDeviceToken.GCM))
        apple_devices = list(PushDeviceToken.objects.filter(user=user_profile, kind=PushDeviceToken.APNS))
        android_successfully_sent_count = send_android_push_notification(user_identity, android_devices, gcm_payload, gcm_options)
        apple_successfully_sent_count = send_apple_push_notification(user_identity, apple_devices, apns_payload)
        do_increment_logging_stat(user_profile.realm, COUNT_STATS['mobile_pushes_sent::day'], None, timezone_now(), increment=android_successfully_sent_count + apple_successfully_sent_count)
    with transaction.atomic(savepoint=False):
        UserMessage.select_for_update_query().filter(user_profile_id=user_profile_id, message_id__in=message_ids).update(flags=F('flags').bitand(~UserMessage.flags.active_mobile_push_notification))

def handle_push_notification(user_profile_id: int, missed_message: Dict[str, Any]) -> None:
    if False:
        return 10
    '\n    missed_message is the event received by the\n    zerver.worker.queue_processors.PushNotificationWorker.consume function.\n    '
    if not push_notifications_enabled():
        return
    user_profile = get_user_profile_by_id(user_profile_id)
    if user_profile.is_bot:
        logger.warning('Send-push-notification event found for bot user %s. Skipping.', user_profile_id)
        return
    if not (user_profile.enable_offline_push_notifications or user_profile.enable_online_push_notifications):
        return
    with transaction.atomic(savepoint=False):
        try:
            (message, user_message) = access_message(user_profile, missed_message['message_id'], lock_message=True)
        except JsonableError:
            if ArchivedMessage.objects.filter(id=missed_message['message_id']).exists():
                return
            logging.info('Unexpected message access failure handling push notifications: %s %s', user_profile.id, missed_message['message_id'])
            return
        if user_message is not None:
            if user_message.flags.read or user_message.flags.active_mobile_push_notification:
                return
            user_message.flags.active_mobile_push_notification = True
            user_message.save(update_fields=['flags'])
        elif not user_profile.long_term_idle:
            logger.error('Could not find UserMessage with message_id %s and user_id %s', missed_message['message_id'], user_profile_id, exc_info=True)
            return
    trigger = missed_message['trigger']
    if trigger == 'wildcard_mentioned':
        trigger = NotificationTriggers.STREAM_WILDCARD_MENTION
    if trigger == 'followed_topic_wildcard_mentioned':
        trigger = NotificationTriggers.STREAM_WILDCARD_MENTION_IN_FOLLOWED_TOPIC
    if trigger == 'private_message':
        trigger = NotificationTriggers.DIRECT_MESSAGE
    mentioned_user_group_name = None
    mentioned_user_group_id = missed_message.get('mentioned_user_group_id')
    if mentioned_user_group_id is not None:
        user_group = UserGroup.objects.get(id=mentioned_user_group_id, realm=user_profile.realm)
        mentioned_user_group_name = user_group.name
    soft_reactivate_if_personal_notification(user_profile, {trigger}, mentioned_user_group_name)
    apns_payload = get_message_payload_apns(user_profile, message, trigger, mentioned_user_group_id, mentioned_user_group_name)
    (gcm_payload, gcm_options) = get_message_payload_gcm(user_profile, message, mentioned_user_group_id, mentioned_user_group_name)
    logger.info('Sending push notifications to mobile clients for user %s', user_profile_id)
    if uses_notification_bouncer():
        (total_android_devices, total_apple_devices) = send_notifications_to_bouncer(user_profile, apns_payload, gcm_payload, gcm_options)
        logger.info('Sent mobile push notifications for user %s through bouncer: %s via FCM devices, %s via APNs devices', user_profile_id, total_android_devices, total_apple_devices)
        return
    android_devices = list(PushDeviceToken.objects.filter(user=user_profile, kind=PushDeviceToken.GCM))
    apple_devices = list(PushDeviceToken.objects.filter(user=user_profile, kind=PushDeviceToken.APNS))
    logger.info('Sending mobile push notifications for local user %s: %s via FCM devices, %s via APNs devices', user_profile_id, len(android_devices), len(apple_devices))
    user_identity = UserPushIdentityCompat(user_id=user_profile.id)
    apple_successfully_sent_count = send_apple_push_notification(user_identity, apple_devices, apns_payload)
    android_successfully_sent_count = send_android_push_notification(user_identity, android_devices, gcm_payload, gcm_options)
    do_increment_logging_stat(user_profile.realm, COUNT_STATS['mobile_pushes_sent::day'], None, timezone_now(), increment=apple_successfully_sent_count + android_successfully_sent_count)

def send_test_push_notification_directly_to_devices(user_identity: UserPushIdentityCompat, devices: Sequence[DeviceToken], base_payload: Dict[str, Any], remote: Optional['RemoteZulipServer']=None) -> None:
    if False:
        while True:
            i = 10
    payload = copy.deepcopy(base_payload)
    payload['event'] = 'test-by-device-token'
    apple_devices = [device for device in devices if device.kind == PushDeviceToken.APNS]
    android_devices = [device for device in devices if device.kind == PushDeviceToken.GCM]
    apple_payload = copy.deepcopy(payload)
    android_payload = copy.deepcopy(payload)
    realm_uri = base_payload['realm_uri']
    apns_data = {'alert': {'title': _('Test notification'), 'body': _('This is a test notification from {realm_uri}.').format(realm_uri=realm_uri)}, 'sound': 'default', 'custom': {'zulip': apple_payload}}
    send_apple_push_notification(user_identity, apple_devices, apns_data, remote=remote)
    android_payload['time'] = datetime_to_timestamp(timezone_now())
    gcm_options = {'priority': 'high'}
    send_android_push_notification(user_identity, android_devices, android_payload, gcm_options, remote=remote)

def send_test_push_notification(user_profile: UserProfile, devices: List[PushDeviceToken]) -> None:
    if False:
        for i in range(10):
            print('nop')
    base_payload = get_base_payload(user_profile)
    if uses_notification_bouncer():
        for device in devices:
            post_data = {'user_uuid': str(user_profile.uuid), 'user_id': user_profile.id, 'token': device.token, 'token_kind': device.kind, 'base_payload': base_payload}
            logger.info('Sending test push notification to bouncer: %r', post_data)
            send_json_to_push_bouncer('POST', 'push/test_notification', post_data)
        return
    user_identity = UserPushIdentityCompat(user_id=user_profile.id, user_uuid=str(user_profile.uuid))
    send_test_push_notification_directly_to_devices(user_identity, devices, base_payload, remote=None)

class InvalidPushDeviceTokenError(JsonableError):
    code = ErrorCode.INVALID_PUSH_DEVICE_TOKEN

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        if False:
            i = 10
            return i + 15
        return _('Device not recognized')

class InvalidRemotePushDeviceTokenError(JsonableError):
    code = ErrorCode.INVALID_REMOTE_PUSH_DEVICE_TOKEN

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    @override
    def msg_format() -> str:
        if False:
            return 10
        return _('Device not recognized by the push bouncer')