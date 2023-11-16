import logging
import os
from typing import Any, Dict, List, Set, Tuple
import dateutil.parser
import orjson
from django.conf import settings
from django.forms.models import model_to_dict
from django.utils.timezone import now as timezone_now
from typing_extensions import TypeAlias
from zerver.data_import.import_util import ZerverFieldsT, build_avatar, build_defaultstream, build_message, build_realm, build_recipient, build_stream, build_subscription, build_usermessages, build_zerver_realm, create_converted_data_files, long_term_idle_helper, make_subscriber_map, process_avatars
from zerver.lib.export import MESSAGE_BATCH_CHUNK_SIZE
from zerver.models import Recipient, UserProfile
from zproject.backends import GitHubAuthBackend
GitterDataT: TypeAlias = List[Dict[str, Any]]
realm_id = 0

def gitter_workspace_to_realm(domain_name: str, gitter_data: GitterDataT, realm_subdomain: str) -> Tuple[ZerverFieldsT, List[ZerverFieldsT], Dict[str, int], Dict[str, int]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns:\n    1. realm, converted realm data\n    2. avatars, which is list to map avatars to Zulip avatar records.json\n    3. user_map, which is a dictionary to map from Gitter user id to Zulip user id\n    4. stream_map, which is a dictionary to map from Gitter rooms to Zulip stream id\n    '
    NOW = float(timezone_now().timestamp())
    zerver_realm: List[ZerverFieldsT] = build_zerver_realm(realm_id, realm_subdomain, NOW, 'Gitter')
    realm = build_realm(zerver_realm, realm_id, domain_name)
    realm['zerver_realmauthenticationmethod'] = [{'name': GitHubAuthBackend.auth_backend_name, 'realm': realm_id, 'id': 1}]
    (zerver_userprofile, avatars, user_map) = build_userprofile(int(NOW), domain_name, gitter_data)
    (zerver_stream, zerver_defaultstream, stream_map) = build_stream_map(int(NOW), gitter_data)
    (zerver_recipient, zerver_subscription) = build_recipient_and_subscription(zerver_userprofile, zerver_stream)
    realm['zerver_userprofile'] = zerver_userprofile
    realm['zerver_stream'] = zerver_stream
    realm['zerver_defaultstream'] = zerver_defaultstream
    realm['zerver_recipient'] = zerver_recipient
    realm['zerver_subscription'] = zerver_subscription
    return (realm, avatars, user_map, stream_map)

def build_userprofile(timestamp: Any, domain_name: str, gitter_data: GitterDataT) -> Tuple[List[ZerverFieldsT], List[ZerverFieldsT], Dict[str, int]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns:\n    1. zerver_userprofile, which is a list of user profile\n    2. avatar_list, which is list to map avatars to Zulip avatars records.json\n    3. added_users, which is a dictionary to map from Gitter user id to Zulip id\n    '
    logging.info('######### IMPORTING USERS STARTED #########\n')
    zerver_userprofile = []
    avatar_list: List[ZerverFieldsT] = []
    user_map: Dict[str, int] = {}
    user_id = 0
    for data in gitter_data:
        if get_user_from_message(data) not in user_map:
            user_data = data['fromUser']
            user_map[user_data['id']] = user_id
            email = get_user_email(user_data, domain_name)
            if user_data.get('avatarUrl'):
                build_avatar(user_id, realm_id, email, user_data['avatarUrl'], timestamp, avatar_list)
            userprofile = UserProfile(full_name=user_data['displayName'], id=user_id, email=email, delivery_email=email, avatar_source='U', date_joined=timestamp, last_login=timestamp)
            userprofile_dict = model_to_dict(userprofile)
            userprofile_dict['realm'] = realm_id
            userprofile_dict['short_name'] = user_data['username']
            zerver_userprofile.append(userprofile_dict)
            user_id += 1
    logging.info('######### IMPORTING USERS FINISHED #########\n')
    return (zerver_userprofile, avatar_list, user_map)

def get_user_email(user_data: ZerverFieldsT, domain_name: str) -> str:
    if False:
        i = 10
        return i + 15
    email = '{}@users.noreply.github.com'.format(user_data['username'])
    return email

def build_stream_map(timestamp: Any, gitter_data: GitterDataT) -> Tuple[List[ZerverFieldsT], List[ZerverFieldsT], Dict[str, int]]:
    if False:
        return 10
    '\n    Returns:\n    1. stream, which is the list of streams\n    2. defaultstreams, which is the list of default streams\n    3. stream_map, which is a dictionary to map from Gitter rooms to Zulip stream id\n    '
    logging.info('######### IMPORTING STREAM STARTED #########\n')
    stream_id = 0
    stream: List[ZerverFieldsT] = []
    stream.append(build_stream(timestamp, realm_id, 'from gitter', 'Imported from Gitter', stream_id))
    defaultstream = build_defaultstream(realm_id=realm_id, stream_id=stream_id, defaultstream_id=0)
    stream_id += 1
    stream_map: Dict[str, int] = {}
    for data in gitter_data:
        if 'room' in data and data['room'] not in stream_map:
            stream.append(build_stream(timestamp, realm_id, data['room'], f"Gitter room {data['room']}", stream_id))
            stream_map[data['room']] = stream_id
            stream_id += 1
    logging.info('######### IMPORTING STREAMS FINISHED #########\n')
    return (stream, [defaultstream], stream_map)

def build_recipient_and_subscription(zerver_userprofile: List[ZerverFieldsT], zerver_stream: List[ZerverFieldsT]) -> Tuple[List[ZerverFieldsT], List[ZerverFieldsT]]:
    if False:
        i = 10
        return i + 15
    "\n    Assumes that there is at least one stream with 'stream_id' = 0,\n      and that this stream is the only defaultstream, with 'defaultstream_id' = 0\n    Returns:\n    1. zerver_recipient, which is a list of mapped recipient\n    2. zerver_subscription, which is a list of mapped subscription\n    "
    zerver_recipient = []
    zerver_subscription = []
    recipient_id = subscription_id = 0
    for stream in zerver_stream:
        zerver_recipient.append(build_recipient(recipient_id, recipient_id, Recipient.STREAM))
        recipient_id += 1
    for user in zerver_userprofile:
        zerver_recipient.append(build_recipient(user['id'], recipient_id, Recipient.PERSONAL))
        zerver_subscription.append(build_subscription(recipient_id, user['id'], subscription_id))
        recipient_id += 1
        subscription_id += 1
    for user in zerver_userprofile:
        for stream in zerver_stream:
            zerver_subscription.append(build_subscription(stream['id'], user['id'], subscription_id))
            subscription_id += 1
    return (zerver_recipient, zerver_subscription)

def get_timestamp_from_message(message: ZerverFieldsT) -> float:
    if False:
        return 10
    return float(dateutil.parser.parse(message['sent']).timestamp())

def get_user_from_message(message: ZerverFieldsT) -> str:
    if False:
        i = 10
        return i + 15
    return message['fromUser']['id']

def convert_gitter_workspace_messages(gitter_data: GitterDataT, output_dir: str, subscriber_map: Dict[int, Set[int]], user_map: Dict[str, int], stream_map: Dict[str, int], user_short_name_to_full_name: Dict[str, str], zerver_userprofile: List[ZerverFieldsT], realm_id: int, chunk_size: int=MESSAGE_BATCH_CHUNK_SIZE) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Messages are stored in batches\n    '
    logging.info('######### IMPORTING MESSAGES STARTED #########\n')
    long_term_idle = long_term_idle_helper(iter(gitter_data), get_user_from_message, get_timestamp_from_message, lambda id: user_map[id], iter(user_map.keys()), zerver_userprofile)
    message_id = 0
    low_index = 0
    upper_index = low_index + chunk_size
    dump_file_id = 1
    while True:
        message_json = {}
        zerver_message = []
        zerver_usermessage: List[ZerverFieldsT] = []
        message_data = gitter_data[low_index:upper_index]
        if len(message_data) == 0:
            break
        for message in message_data:
            message_time = get_timestamp_from_message(message)
            mentioned_user_ids = get_usermentions(message, user_map, user_short_name_to_full_name)
            rendered_content = None
            topic_name = 'imported from Gitter' + (f" room {message['room']}" if 'room' in message else '')
            user_id = user_map[get_user_from_message(message)]
            recipient_id = stream_map[message['room']] if 'room' in message else 0
            zulip_message = build_message(topic_name=topic_name, date_sent=message_time, message_id=message_id, content=message['text'], rendered_content=rendered_content, user_id=user_id, recipient_id=recipient_id, realm_id=realm_id)
            zerver_message.append(zulip_message)
            build_usermessages(zerver_usermessage=zerver_usermessage, subscriber_map=subscriber_map, recipient_id=recipient_id, mentioned_user_ids=mentioned_user_ids, message_id=message_id, is_private=False, long_term_idle=long_term_idle)
            message_id += 1
        message_json['zerver_message'] = zerver_message
        message_json['zerver_usermessage'] = zerver_usermessage
        message_filename = os.path.join(output_dir, f'messages-{dump_file_id:06}.json')
        logging.info('Writing messages to %s\n', message_filename)
        write_data_to_file(os.path.join(message_filename), message_json)
        low_index = upper_index
        upper_index = chunk_size + low_index
        dump_file_id += 1
    logging.info('######### IMPORTING MESSAGES FINISHED #########\n')

def get_usermentions(message: Dict[str, Any], user_map: Dict[str, int], user_short_name_to_full_name: Dict[str, str]) -> List[int]:
    if False:
        while True:
            i = 10
    mentioned_user_ids = []
    if 'mentions' in message:
        for mention in message['mentions']:
            if mention.get('userId') in user_map:
                gitter_mention = '@{}'.format(mention['screenName'])
                if mention['screenName'] not in user_short_name_to_full_name:
                    logging.info('Mentioned user %s never sent any messages, so has no full name data', mention['screenName'])
                    full_name = mention['screenName']
                else:
                    full_name = user_short_name_to_full_name[mention['screenName']]
                zulip_mention = f'@**{full_name}**'
                message['text'] = message['text'].replace(gitter_mention, zulip_mention)
                mentioned_user_ids.append(user_map[mention['userId']])
    return mentioned_user_ids

def do_convert_data(gitter_data_file: str, output_dir: str, threads: int=6) -> None:
    if False:
        for i in range(10):
            print('nop')
    realm_subdomain = ''
    domain_name = settings.EXTERNAL_HOST
    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        raise Exception('Output directory should be empty!')
    with open(gitter_data_file, 'rb') as fp:
        gitter_data = orjson.loads(fp.read())
    (realm, avatar_list, user_map, stream_map) = gitter_workspace_to_realm(domain_name, gitter_data, realm_subdomain)
    subscriber_map = make_subscriber_map(zerver_subscription=realm['zerver_subscription'])
    user_short_name_to_full_name = {}
    for userprofile in realm['zerver_userprofile']:
        user_short_name_to_full_name[userprofile['short_name']] = userprofile['full_name']
    convert_gitter_workspace_messages(gitter_data, output_dir, subscriber_map, user_map, stream_map, user_short_name_to_full_name, realm['zerver_userprofile'], realm_id=realm_id)
    avatar_folder = os.path.join(output_dir, 'avatars')
    avatar_realm_folder = os.path.join(avatar_folder, str(realm_id))
    os.makedirs(avatar_realm_folder, exist_ok=True)
    avatar_records = process_avatars(avatar_list, avatar_folder, realm_id, threads)
    attachment: Dict[str, List[Any]] = {'zerver_attachment': []}
    create_converted_data_files(realm, output_dir, '/realm.json')
    create_converted_data_files([], output_dir, '/emoji/records.json')
    create_converted_data_files(avatar_records, output_dir, '/avatars/records.json')
    create_converted_data_files([], output_dir, '/uploads/records.json')
    create_converted_data_files(attachment, output_dir, '/attachment.json')
    logging.info('######### DATA CONVERSION FINISHED #########\n')
    logging.info('Zulip data dump created at %s', output_dir)

def write_data_to_file(output_file: str, data: Any) -> None:
    if False:
        print('Hello World!')
    with open(output_file, 'wb') as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))