import datetime
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from mimetypes import guess_type
from typing import Any, Dict, List, Optional, Tuple
import bmemcached
import orjson
from bs4 import BeautifulSoup
from django.conf import settings
from django.core.cache import cache
from django.core.validators import validate_email
from django.db import connection, transaction
from django.utils.timezone import now as timezone_now
from psycopg2.extras import execute_values
from psycopg2.sql import SQL, Identifier
from analytics.models import RealmCount, StreamCount, UserCount
from zerver.actions.create_realm import set_default_for_realm_permission_group_settings
from zerver.actions.realm_settings import do_change_realm_plan_type
from zerver.actions.user_settings import do_change_avatar_fields
from zerver.lib.avatar_hash import user_avatar_path_from_ids
from zerver.lib.bulk_create import bulk_set_users_or_streams_recipient_fields
from zerver.lib.export import DATE_FIELDS, Field, Path, Record, TableData, TableName
from zerver.lib.markdown import markdown_convert
from zerver.lib.markdown import version as markdown_version
from zerver.lib.message import get_last_message_id
from zerver.lib.server_initialization import create_internal_realm, server_initialized
from zerver.lib.streams import render_stream_description
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.upload import upload_backend
from zerver.lib.upload.base import BadImageError, sanitize_name
from zerver.lib.upload.s3 import get_bucket
from zerver.lib.user_groups import create_system_user_groups_for_realm
from zerver.lib.user_message import UserMessageLite, bulk_insert_ums
from zerver.lib.utils import generate_api_key, process_list_in_batches
from zerver.models import AlertWord, Attachment, BotConfigData, BotStorageData, Client, CustomProfileField, CustomProfileFieldValue, DefaultStream, GroupGroupMembership, Huddle, Message, MutedUser, Reaction, Realm, RealmAuditLog, RealmAuthenticationMethod, RealmDomain, RealmEmoji, RealmFilter, RealmPlayground, RealmUserDefault, Recipient, ScheduledMessage, Service, Stream, Subscription, SystemGroups, UserActivity, UserActivityInterval, UserGroup, UserGroupMembership, UserHotspot, UserMessage, UserPresence, UserProfile, UserStatus, UserTopic, get_huddle_hash, get_realm, get_system_bot, get_user_profile_by_id
realm_tables = [('zerver_realmauthenticationmethod', RealmAuthenticationMethod, 'realmauthenticationmethod'), ('zerver_defaultstream', DefaultStream, 'defaultstream'), ('zerver_realmemoji', RealmEmoji, 'realmemoji'), ('zerver_realmdomain', RealmDomain, 'realmdomain'), ('zerver_realmfilter', RealmFilter, 'realmfilter'), ('zerver_realmplayground', RealmPlayground, 'realmplayground')]
ID_MAP: Dict[str, Dict[int, int]] = {'alertword': {}, 'client': {}, 'user_profile': {}, 'huddle': {}, 'realm': {}, 'stream': {}, 'recipient': {}, 'subscription': {}, 'defaultstream': {}, 'reaction': {}, 'realmauthenticationmethod': {}, 'realmemoji': {}, 'realmdomain': {}, 'realmfilter': {}, 'realmplayground': {}, 'message': {}, 'user_presence': {}, 'userstatus': {}, 'useractivity': {}, 'useractivityinterval': {}, 'usermessage': {}, 'customprofilefield': {}, 'customprofilefieldvalue': {}, 'attachment': {}, 'realmauditlog': {}, 'recipient_to_huddle_map': {}, 'userhotspot': {}, 'usertopic': {}, 'muteduser': {}, 'service': {}, 'usergroup': {}, 'usergroupmembership': {}, 'groupgroupmembership': {}, 'botstoragedata': {}, 'botconfigdata': {}, 'analytics_realmcount': {}, 'analytics_streamcount': {}, 'analytics_usercount': {}, 'realmuserdefault': {}, 'scheduledmessage': {}}
id_map_to_list: Dict[str, Dict[int, List[int]]] = {'huddle_to_user_list': {}}
path_maps: Dict[str, Dict[str, str]] = {'attachment_path': {}}

def update_id_map(table: TableName, old_id: int, new_id: int) -> None:
    if False:
        i = 10
        return i + 15
    if table not in ID_MAP:
        raise Exception(f'\n            Table {table} is not initialized in ID_MAP, which could\n            mean that we have not thought through circular\n            dependencies.\n            ')
    ID_MAP[table][old_id] = new_id

def fix_datetime_fields(data: TableData, table: TableName) -> None:
    if False:
        print('Hello World!')
    for item in data[table]:
        for field_name in DATE_FIELDS[table]:
            if item[field_name] is not None:
                item[field_name] = datetime.datetime.fromtimestamp(item[field_name], tz=datetime.timezone.utc)

def fix_upload_links(data: TableData, message_table: TableName) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Because the URLs for uploaded files encode the realm ID of the\n    organization being imported (which is only determined at import\n    time), we need to rewrite the URLs of links to uploaded files\n    during the import process.\n    '
    for message in data[message_table]:
        if message['has_attachment'] is True:
            for (key, value) in path_maps['attachment_path'].items():
                if key in message['content']:
                    message['content'] = message['content'].replace(key, value)
                    if message['rendered_content']:
                        message['rendered_content'] = message['rendered_content'].replace(key, value)

def fix_streams_can_remove_subscribers_group_column(data: TableData, realm: Realm) -> None:
    if False:
        i = 10
        return i + 15
    table = get_db_table(Stream)
    admins_group = UserGroup.objects.get(name=SystemGroups.ADMINISTRATORS, realm=realm, is_system_group=True)
    for stream in data[table]:
        stream['can_remove_subscribers_group'] = admins_group

def create_subscription_events(data: TableData, realm_id: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    "\n    When the export data doesn't contain the table `zerver_realmauditlog`,\n    this function creates RealmAuditLog objects for `subscription_created`\n    type event for all the existing Stream subscriptions.\n\n    This is needed for all the export tools which do not include the\n    table `zerver_realmauditlog` (Slack, Gitter, etc.) because the appropriate\n    data about when a user was subscribed is not exported by the third-party\n    service.\n    "
    all_subscription_logs = []
    event_last_message_id = get_last_message_id()
    event_time = timezone_now()
    recipient_id_to_stream_id = {d['id']: d['type_id'] for d in data['zerver_recipient'] if d['type'] == Recipient.STREAM}
    for sub in data['zerver_subscription']:
        recipient_id = sub['recipient_id']
        stream_id = recipient_id_to_stream_id.get(recipient_id)
        if stream_id is None:
            continue
        user_id = sub['user_profile_id']
        all_subscription_logs.append(RealmAuditLog(realm_id=realm_id, acting_user_id=user_id, modified_user_id=user_id, modified_stream_id=stream_id, event_last_message_id=event_last_message_id, event_time=event_time, event_type=RealmAuditLog.SUBSCRIPTION_CREATED))
    RealmAuditLog.objects.bulk_create(all_subscription_logs)

def fix_service_tokens(data: TableData, table: TableName) -> None:
    if False:
        i = 10
        return i + 15
    "\n    The tokens in the services are created by 'generate_api_key'.\n    As the tokens are unique, they should be re-created for the imports.\n    "
    for item in data[table]:
        item['token'] = generate_api_key()

def process_huddle_hash(data: TableData, table: TableName) -> None:
    if False:
        return 10
    '\n    Build new huddle hashes with the updated ids of the users\n    '
    for huddle in data[table]:
        user_id_list = id_map_to_list['huddle_to_user_list'][huddle['id']]
        huddle['huddle_hash'] = get_huddle_hash(user_id_list)

def get_huddles_from_subscription(data: TableData, table: TableName) -> None:
    if False:
        print('Hello World!')
    '\n    Extract the IDs of the user_profiles involved in a huddle from the subscription object\n    This helps to generate a unique huddle hash from the updated user_profile ids\n    '
    id_map_to_list['huddle_to_user_list'] = {value: [] for value in ID_MAP['recipient_to_huddle_map'].values()}
    for subscription in data[table]:
        if subscription['recipient'] in ID_MAP['recipient_to_huddle_map']:
            huddle_id = ID_MAP['recipient_to_huddle_map'][subscription['recipient']]
            id_map_to_list['huddle_to_user_list'][huddle_id].append(subscription['user_profile_id'])

def fix_customprofilefield(data: TableData) -> None:
    if False:
        print('Hello World!')
    "\n    In CustomProfileField with 'field_type' like 'USER', the IDs need to be\n    re-mapped.\n    "
    field_type_USER_ids = {item['id'] for item in data['zerver_customprofilefield'] if item['field_type'] == CustomProfileField.USER}
    for item in data['zerver_customprofilefieldvalue']:
        if item['field_id'] in field_type_USER_ids:
            old_user_id_list = orjson.loads(item['value'])
            new_id_list = re_map_foreign_keys_many_to_many_internal(table='zerver_customprofilefieldvalue', field_name='value', related_table='user_profile', old_id_list=old_user_id_list)
            item['value'] = orjson.dumps(new_id_list).decode()

def fix_message_rendered_content(realm: Realm, sender_map: Dict[int, Record], messages: List[Record]) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    This function sets the rendered_content of all the messages\n    after the messages have been imported from a non-Zulip platform.\n    '
    for message in messages:
        if message['rendered_content'] is not None:
            soup = BeautifulSoup(message['rendered_content'], 'html.parser')
            user_mentions = soup.findAll('span', {'class': 'user-mention'})
            if len(user_mentions) != 0:
                user_id_map = ID_MAP['user_profile']
                for mention in user_mentions:
                    if not mention.has_attr('data-user-id'):
                        continue
                    if mention['data-user-id'] == '*':
                        continue
                    old_user_id = int(mention['data-user-id'])
                    if old_user_id in user_id_map:
                        mention['data-user-id'] = str(user_id_map[old_user_id])
                message['rendered_content'] = str(soup)
            stream_mentions = soup.findAll('a', {'class': 'stream'})
            if len(stream_mentions) != 0:
                stream_id_map = ID_MAP['stream']
                for mention in stream_mentions:
                    old_stream_id = int(mention['data-stream-id'])
                    if old_stream_id in stream_id_map:
                        mention['data-stream-id'] = str(stream_id_map[old_stream_id])
                message['rendered_content'] = str(soup)
            user_group_mentions = soup.findAll('span', {'class': 'user-group-mention'})
            if len(user_group_mentions) != 0:
                user_group_id_map = ID_MAP['usergroup']
                for mention in user_group_mentions:
                    old_user_group_id = int(mention['data-user-group-id'])
                    if old_user_group_id in user_group_id_map:
                        mention['data-user-group-id'] = str(user_group_id_map[old_user_group_id])
                message['rendered_content'] = str(soup)
            continue
        try:
            content = message['content']
            sender_id = message['sender_id']
            sender = sender_map[sender_id]
            sent_by_bot = sender['is_bot']
            translate_emoticons = sender['translate_emoticons']
            realm_alert_words_automaton = None
            rendered_content = markdown_convert(content=content, realm_alert_words_automaton=realm_alert_words_automaton, message_realm=realm, sent_by_bot=sent_by_bot, translate_emoticons=translate_emoticons).rendered_content
            message['rendered_content'] = rendered_content
            if 'scheduled_timestamp' not in message:
                message['rendered_content_version'] = markdown_version
        except Exception:
            logging.warning('Error in Markdown rendering for message ID %s; continuing', message['id'])

def current_table_ids(data: TableData, table: TableName) -> List[int]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the ids present in the current table\n    '
    return [item['id'] for item in data[table]]

def idseq(model_class: Any) -> str:
    if False:
        i = 10
        return i + 15
    if model_class == RealmDomain:
        return 'zerver_realmalias_id_seq'
    elif model_class == BotStorageData:
        return 'zerver_botuserstatedata_id_seq'
    elif model_class == BotConfigData:
        return 'zerver_botuserconfigdata_id_seq'
    elif model_class == UserTopic:
        return 'zerver_mutedtopic_id_seq'
    return f'{model_class._meta.db_table}_id_seq'

def allocate_ids(model_class: Any, count: int) -> List[int]:
    if False:
        return 10
    '\n    Increases the sequence number for a given table by the amount of objects being\n    imported into that table. Hence, this gives a reserved range of IDs to import the\n    converted Slack objects into the tables.\n    '
    conn = connection.cursor()
    sequence = idseq(model_class)
    conn.execute('select nextval(%s) from generate_series(1, %s)', [sequence, count])
    query = conn.fetchall()
    conn.close()
    return [item[0] for item in query]

def convert_to_id_fields(data: TableData, table: TableName, field_name: Field) -> None:
    if False:
        i = 10
        return i + 15
    '\n    When Django gives us dict objects via model_to_dict, the foreign\n    key fields are `foo`, but we want `foo_id` for the bulk insert.\n    This function handles the simple case where we simply rename\n    the fields.  For cases where we need to munge ids in the\n    database, see re_map_foreign_keys.\n    '
    for item in data[table]:
        item[field_name + '_id'] = item[field_name]
        del item[field_name]

def re_map_foreign_keys(data: TableData, table: TableName, field_name: Field, related_table: TableName, verbose: bool=False, id_field: bool=False, recipient_field: bool=False) -> None:
    if False:
        while True:
            i = 10
    '\n    This is a wrapper function for all the realm data tables\n    and only avatar and attachment records need to be passed through the internal function\n    because of the difference in data format (TableData corresponding to realm data tables\n    and List[Record] corresponding to the avatar and attachment records)\n    '
    assert 'usermessage' not in related_table
    re_map_foreign_keys_internal(data[table], table, field_name, related_table, verbose, id_field, recipient_field)

def re_map_foreign_keys_internal(data_table: List[Record], table: TableName, field_name: Field, related_table: TableName, verbose: bool=False, id_field: bool=False, recipient_field: bool=False) -> None:
    if False:
        while True:
            i = 10
    '\n    We occasionally need to assign new ids to rows during the\n    import/export process, to accommodate things like existing rows\n    already being in tables.  See bulk_import_client for more context.\n\n    The tricky part is making sure that foreign key references\n    are in sync with the new ids, and this fixer function does\n    the re-mapping.  (It also appends `_id` to the field.)\n    '
    lookup_table = ID_MAP[related_table]
    for item in data_table:
        old_id = item[field_name]
        if recipient_field:
            if related_table == 'stream' and item['type'] == 2:
                pass
            elif related_table == 'user_profile' and item['type'] == 1:
                pass
            elif related_table == 'huddle' and item['type'] == 3:
                ID_MAP['recipient_to_huddle_map'][item['id']] = lookup_table[old_id]
            else:
                continue
        old_id = item[field_name]
        if old_id in lookup_table:
            new_id = lookup_table[old_id]
            if verbose:
                logging.info('Remapping %s %s from %s to %s', table, field_name + '_id', old_id, new_id)
        else:
            new_id = old_id
        if not id_field:
            item[field_name + '_id'] = new_id
            del item[field_name]
        else:
            item[field_name] = new_id

def re_map_realm_emoji_codes(data: TableData, *, table_name: str) -> None:
    if False:
        print('Hello World!')
    '\n    Some tables, including Reaction and UserStatus, contain a form of\n    foreign key reference to the RealmEmoji table in the form of\n    `str(realm_emoji.id)` when `reaction_type="realm_emoji"`.\n\n    See the block comment for emoji_code in the AbstractEmoji\n    definition for more details.\n    '
    realm_emoji_dct = {}
    for row in data['zerver_realmemoji']:
        realm_emoji_dct[row['id']] = row
    for row in data[table_name]:
        if row['reaction_type'] == Reaction.REALM_EMOJI:
            old_realm_emoji_id = int(row['emoji_code'])
            new_realm_emoji_id = ID_MAP['realmemoji'][old_realm_emoji_id]
            realm_emoji_row = realm_emoji_dct[new_realm_emoji_id]
            assert realm_emoji_row['name'] == row['emoji_name']
            row['emoji_code'] = str(new_realm_emoji_id)

def re_map_foreign_keys_many_to_many(data: TableData, table: TableName, field_name: Field, related_table: TableName, verbose: bool=False) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    We need to assign new ids to rows during the import/export\n    process.\n\n    The tricky part is making sure that foreign key references\n    are in sync with the new ids, and this wrapper function does\n    the re-mapping only for ManyToMany fields.\n    '
    for item in data[table]:
        old_id_list = item[field_name]
        new_id_list = re_map_foreign_keys_many_to_many_internal(table, field_name, related_table, old_id_list, verbose)
        item[field_name] = new_id_list
        del item[field_name]

def re_map_foreign_keys_many_to_many_internal(table: TableName, field_name: Field, related_table: TableName, old_id_list: List[int], verbose: bool=False) -> List[int]:
    if False:
        print('Hello World!')
    '\n    This is an internal function for tables with ManyToMany fields,\n    which takes the old ID list of the ManyToMany relation and returns the\n    new updated ID list.\n    '
    lookup_table = ID_MAP[related_table]
    new_id_list = []
    for old_id in old_id_list:
        if old_id in lookup_table:
            new_id = lookup_table[old_id]
            if verbose:
                logging.info('Remapping %s %s from %s to %s', table, field_name + '_id', old_id, new_id)
        else:
            new_id = old_id
        new_id_list.append(new_id)
    return new_id_list

def fix_bitfield_keys(data: TableData, table: TableName, field_name: Field) -> None:
    if False:
        for i in range(10):
            print('nop')
    for item in data[table]:
        item[field_name] = item[field_name + '_mask']
        del item[field_name + '_mask']

def remove_denormalized_recipient_column_from_data(data: TableData) -> None:
    if False:
        i = 10
        return i + 15
    "\n    The recipient column shouldn't be imported, we'll set the correct values\n    when Recipient table gets imported.\n    "
    for stream_dict in data['zerver_stream']:
        if 'recipient' in stream_dict:
            del stream_dict['recipient']
    for user_profile_dict in data['zerver_userprofile']:
        if 'recipient' in user_profile_dict:
            del user_profile_dict['recipient']
    for huddle_dict in data['zerver_huddle']:
        if 'recipient' in huddle_dict:
            del huddle_dict['recipient']

def get_db_table(model_class: Any) -> str:
    if False:
        i = 10
        return i + 15
    "E.g. (RealmDomain -> 'zerver_realmdomain')"
    return model_class._meta.db_table

def update_model_ids(model: Any, data: TableData, related_table: TableName) -> None:
    if False:
        i = 10
        return i + 15
    table = get_db_table(model)
    assert 'usermessage' not in table
    old_id_list = current_table_ids(data, table)
    allocated_id_list = allocate_ids(model, len(data[table]))
    for item in range(len(data[table])):
        update_id_map(related_table, old_id_list[item], allocated_id_list[item])
    re_map_foreign_keys(data, table, 'id', related_table=related_table, id_field=True)

def bulk_import_user_message_data(data: TableData, dump_file_id: int) -> None:
    if False:
        while True:
            i = 10
    model = UserMessage
    table = 'zerver_usermessage'
    lst = data[table]

    def process_batch(items: List[Dict[str, Any]]) -> None:
        if False:
            i = 10
            return i + 15
        ums = [UserMessageLite(user_profile_id=item['user_profile_id'], message_id=item['message_id'], flags=item['flags']) for item in items]
        bulk_insert_ums(ums)
    chunk_size = 10000
    process_list_in_batches(lst=lst, chunk_size=chunk_size, process_batch=process_batch)
    logging.info('Successfully imported %s from %s[%s].', model, table, dump_file_id)

def bulk_import_model(data: TableData, model: Any, dump_file_id: Optional[str]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    table = get_db_table(model)
    model.objects.bulk_create((model(**item) for item in data[table]))
    if dump_file_id is None:
        logging.info('Successfully imported %s from %s.', model, table)
    else:
        logging.info('Successfully imported %s from %s[%s].', model, table, dump_file_id)

def bulk_import_client(data: TableData, model: Any, table: TableName) -> None:
    if False:
        print('Hello World!')
    for item in data[table]:
        try:
            client = Client.objects.get(name=item['name'])
        except Client.DoesNotExist:
            client = Client.objects.create(name=item['name'])
        update_id_map(table='client', old_id=item['id'], new_id=client.id)

def fix_subscriptions_is_user_active_column(data: TableData, user_profiles: List[UserProfile]) -> None:
    if False:
        return 10
    table = get_db_table(Subscription)
    user_id_to_active_status = {user.id: user.is_active for user in user_profiles}
    for sub in data[table]:
        sub['is_user_active'] = user_id_to_active_status[sub['user_profile_id']]

def process_avatars(record: Dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    from zerver.lib.upload import upload_backend
    if record['s3_path'].endswith('.original'):
        user_profile = get_user_profile_by_id(record['user_profile_id'])
        if settings.LOCAL_AVATARS_DIR is not None:
            avatar_path = user_avatar_path_from_ids(user_profile.id, record['realm_id'])
            medium_file_path = os.path.join(settings.LOCAL_AVATARS_DIR, avatar_path) + '-medium.png'
            if os.path.exists(medium_file_path):
                os.remove(medium_file_path)
        try:
            upload_backend.ensure_avatar_image(user_profile=user_profile, is_medium=True)
            if record.get('importer_should_thumbnail'):
                upload_backend.ensure_avatar_image(user_profile=user_profile)
        except BadImageError:
            logging.warning('Could not thumbnail avatar image for user %s; ignoring', user_profile.id)
            do_change_avatar_fields(user_profile, UserProfile.AVATAR_FROM_GRAVATAR, acting_user=None)

def import_uploads(realm: Realm, import_dir: Path, processes: int, default_user_profile_id: Optional[int]=None, processing_avatars: bool=False, processing_emojis: bool=False, processing_realm_icons: bool=False) -> None:
    if False:
        for i in range(10):
            print('nop')
    if processing_avatars and processing_emojis:
        raise AssertionError('Cannot import avatars and emojis at the same time!')
    if processing_avatars:
        logging.info('Importing avatars')
    elif processing_emojis:
        logging.info('Importing emojis')
    elif processing_realm_icons:
        logging.info('Importing realm icons and logos')
    else:
        logging.info('Importing uploaded files')
    records_filename = os.path.join(import_dir, 'records.json')
    with open(records_filename, 'rb') as records_file:
        records: List[Dict[str, Any]] = orjson.loads(records_file.read())
    timestamp = datetime_to_timestamp(timezone_now())
    re_map_foreign_keys_internal(records, 'records', 'realm_id', related_table='realm', id_field=True)
    if not processing_emojis and (not processing_realm_icons):
        re_map_foreign_keys_internal(records, 'records', 'user_profile_id', related_table='user_profile', id_field=True)
    s3_uploads = settings.LOCAL_UPLOADS_DIR is None
    if s3_uploads:
        if processing_avatars or processing_emojis or processing_realm_icons:
            bucket_name = settings.S3_AVATAR_BUCKET
        else:
            bucket_name = settings.S3_AUTH_UPLOADS_BUCKET
        bucket = get_bucket(bucket_name)
    count = 0
    for record in records:
        count += 1
        if count % 1000 == 0:
            logging.info('Processed %s/%s uploads', count, len(records))
        if processing_avatars:
            relative_path = user_avatar_path_from_ids(record['user_profile_id'], record['realm_id'])
            if record['s3_path'].endswith('.original'):
                relative_path += '.original'
            elif not s3_uploads:
                relative_path += '.png'
        elif processing_emojis:
            relative_path = RealmEmoji.PATH_ID_TEMPLATE.format(realm_id=record['realm_id'], emoji_file_name=record['file_name'])
            record['last_modified'] = timestamp
        elif processing_realm_icons:
            icon_name = os.path.basename(record['path'])
            relative_path = os.path.join(str(record['realm_id']), 'realm', icon_name)
            record['last_modified'] = timestamp
        else:
            relative_path = upload_backend.generate_message_upload_path(str(record['realm_id']), sanitize_name(os.path.basename(record['path'])))
            path_maps['attachment_path'][record['s3_path']] = relative_path
        if s3_uploads:
            key = bucket.Object(relative_path)
            metadata = {}
            if 'user_profile_id' not in record:
                assert default_user_profile_id is not None
                metadata['user_profile_id'] = str(default_user_profile_id)
            else:
                user_profile_id = int(record['user_profile_id'])
                if user_profile_id in ID_MAP['user_profile']:
                    logging.info('Uploaded by ID mapped user: %s!', user_profile_id)
                    user_profile_id = ID_MAP['user_profile'][user_profile_id]
                user_profile = get_user_profile_by_id(user_profile_id)
                metadata['user_profile_id'] = str(user_profile.id)
            if 'last_modified' in record:
                metadata['orig_last_modified'] = str(record['last_modified'])
            metadata['realm_id'] = str(record['realm_id'])
            content_type = record.get('content_type')
            if content_type is None:
                content_type = guess_type(record['s3_path'])[0]
                if content_type is None:
                    content_type = 'application/octet-stream'
            key.upload_file(Filename=os.path.join(import_dir, record['path']), ExtraArgs={'ContentType': content_type, 'Metadata': metadata})
        else:
            assert settings.LOCAL_UPLOADS_DIR is not None
            assert settings.LOCAL_AVATARS_DIR is not None
            assert settings.LOCAL_FILES_DIR is not None
            if processing_avatars or processing_emojis or processing_realm_icons:
                file_path = os.path.join(settings.LOCAL_AVATARS_DIR, relative_path)
            else:
                file_path = os.path.join(settings.LOCAL_FILES_DIR, relative_path)
            orig_file_path = os.path.join(import_dir, record['path'])
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            shutil.copy(orig_file_path, file_path)
    if processing_avatars:
        if processes == 1:
            for record in records:
                process_avatars(record)
        else:
            connection.close()
            _cache = cache._cache
            assert isinstance(_cache, bmemcached.Client)
            _cache.disconnect_all()
            with ProcessPoolExecutor(max_workers=processes) as executor:
                for future in as_completed((executor.submit(process_avatars, record) for record in records)):
                    future.result()

def do_import_realm(import_dir: Path, subdomain: str, processes: int=1) -> Realm:
    if False:
        return 10
    logging.info('Importing realm dump %s', import_dir)
    if not os.path.exists(import_dir):
        raise Exception('Missing import directory!')
    realm_data_filename = os.path.join(import_dir, 'realm.json')
    if not os.path.exists(realm_data_filename):
        raise Exception('Missing realm.json file!')
    if not server_initialized():
        create_internal_realm()
    logging.info('Importing realm data from %s', realm_data_filename)
    with open(realm_data_filename, 'rb') as f:
        data = orjson.loads(f.read())
    data['zerver_userprofile'] = data['zerver_userprofile'] + data['zerver_userprofile_mirrordummy']
    del data['zerver_userprofile_mirrordummy']
    data['zerver_userprofile'].sort(key=lambda r: r['id'])
    remove_denormalized_recipient_column_from_data(data)
    sort_by_date = data.get('sort_by_date', False)
    bulk_import_client(data, Client, 'zerver_client')
    update_model_ids(Stream, data, 'stream')
    re_map_foreign_keys(data, 'zerver_realm', 'notifications_stream', related_table='stream')
    re_map_foreign_keys(data, 'zerver_realm', 'signup_notifications_stream', related_table='stream')
    if 'zerver_usergroup' in data:
        update_model_ids(UserGroup, data, 'usergroup')
        for setting_name in Realm.REALM_PERMISSION_GROUP_SETTINGS:
            re_map_foreign_keys(data, 'zerver_realm', setting_name, related_table='usergroup')
    fix_datetime_fields(data, 'zerver_realm')
    data['zerver_realm'][0]['string_id'] = subdomain
    data['zerver_realm'][0]['name'] = subdomain
    update_model_ids(Realm, data, 'realm')
    realm_properties = dict(**data['zerver_realm'][0])
    realm_properties['deactivated'] = True
    with transaction.atomic(durable=True):
        realm = Realm(**realm_properties)
        if 'zerver_usergroup' not in data:
            for permission_configuration in Realm.REALM_PERMISSION_GROUP_SETTINGS.values():
                setattr(realm, permission_configuration.id_field_name, -1)
        realm.save()
        if 'zerver_usergroup' in data:
            re_map_foreign_keys(data, 'zerver_usergroup', 'realm', related_table='realm')
            for setting_name in UserGroup.GROUP_PERMISSION_SETTINGS:
                re_map_foreign_keys(data, 'zerver_usergroup', setting_name, related_table='usergroup')
            bulk_import_model(data, UserGroup)
        role_system_groups_dict: Optional[Dict[int, UserGroup]] = None
        if 'zerver_usergroup' not in data:
            role_system_groups_dict = create_system_user_groups_for_realm(realm)
        fix_datetime_fields(data, 'zerver_stream')
        re_map_foreign_keys(data, 'zerver_stream', 'realm', related_table='realm')
        if role_system_groups_dict is not None:
            fix_streams_can_remove_subscribers_group_column(data, realm)
        else:
            re_map_foreign_keys(data, 'zerver_stream', 'can_remove_subscribers_group', related_table='usergroup')
        for stream in data['zerver_stream']:
            stream['rendered_description'] = render_stream_description(stream['description'], realm)
        bulk_import_model(data, Stream)
        if 'zerver_usergroup' not in data:
            set_default_for_realm_permission_group_settings(realm)
    internal_realm = get_realm(settings.SYSTEM_BOT_REALM)
    for item in data['zerver_userprofile_crossrealm']:
        logging.info('Adding to ID map: %s %s', item['id'], get_system_bot(item['email'], internal_realm.id).id)
        new_user_id = get_system_bot(item['email'], internal_realm.id).id
        update_id_map(table='user_profile', old_id=item['id'], new_id=new_user_id)
        new_recipient_id = Recipient.objects.get(type=Recipient.PERSONAL, type_id=new_user_id).id
        update_id_map(table='recipient', old_id=item['recipient_id'], new_id=new_recipient_id)
    update_message_foreign_keys(import_dir=import_dir, sort_by_date=sort_by_date)
    fix_datetime_fields(data, 'zerver_userprofile')
    update_model_ids(UserProfile, data, 'user_profile')
    re_map_foreign_keys(data, 'zerver_userprofile', 'realm', related_table='realm')
    re_map_foreign_keys(data, 'zerver_userprofile', 'bot_owner', related_table='user_profile')
    re_map_foreign_keys(data, 'zerver_userprofile', 'default_sending_stream', related_table='stream')
    re_map_foreign_keys(data, 'zerver_userprofile', 'default_events_register_stream', related_table='stream')
    re_map_foreign_keys(data, 'zerver_userprofile', 'last_active_message_id', related_table='message', id_field=True)
    for user_profile_dict in data['zerver_userprofile']:
        user_profile_dict['password'] = None
        user_profile_dict['api_key'] = generate_api_key()
        del user_profile_dict['user_permissions']
        del user_profile_dict['groups']
        if 'short_name' in user_profile_dict:
            del user_profile_dict['short_name']
    user_profiles = [UserProfile(**item) for item in data['zerver_userprofile']]
    for user_profile in user_profiles:
        validate_email(user_profile.delivery_email)
        validate_email(user_profile.email)
        user_profile.set_unusable_password()
        user_profile.tos_version = UserProfile.TOS_VERSION_BEFORE_FIRST_LOGIN
    UserProfile.objects.bulk_create(user_profiles)
    re_map_foreign_keys(data, 'zerver_defaultstream', 'stream', related_table='stream')
    re_map_foreign_keys(data, 'zerver_realmemoji', 'author', related_table='user_profile')
    for (table, model, related_table) in realm_tables:
        re_map_foreign_keys(data, table, 'realm', related_table='realm')
        update_model_ids(model, data, related_table)
        bulk_import_model(data, model)
    first_user_profile = UserProfile.objects.filter(realm=realm, is_active=True, role=UserProfile.ROLE_REALM_OWNER).order_by('id').first()
    for realm_emoji in RealmEmoji.objects.filter(realm=realm):
        if realm_emoji.author_id is None:
            assert first_user_profile is not None
            realm_emoji.author_id = first_user_profile.id
            realm_emoji.save(update_fields=['author_id'])
    if 'zerver_huddle' in data:
        update_model_ids(Huddle, data, 'huddle')
    re_map_foreign_keys(data, 'zerver_recipient', 'type_id', related_table='stream', recipient_field=True, id_field=True)
    re_map_foreign_keys(data, 'zerver_recipient', 'type_id', related_table='user_profile', recipient_field=True, id_field=True)
    re_map_foreign_keys(data, 'zerver_recipient', 'type_id', related_table='huddle', recipient_field=True, id_field=True)
    update_model_ids(Recipient, data, 'recipient')
    bulk_import_model(data, Recipient)
    bulk_set_users_or_streams_recipient_fields(Stream, Stream.objects.filter(realm=realm))
    bulk_set_users_or_streams_recipient_fields(UserProfile, UserProfile.objects.filter(realm=realm))
    re_map_foreign_keys(data, 'zerver_subscription', 'user_profile', related_table='user_profile')
    get_huddles_from_subscription(data, 'zerver_subscription')
    re_map_foreign_keys(data, 'zerver_subscription', 'recipient', related_table='recipient')
    update_model_ids(Subscription, data, 'subscription')
    fix_subscriptions_is_user_active_column(data, user_profiles)
    bulk_import_model(data, Subscription)
    if 'zerver_realmauditlog' in data:
        fix_datetime_fields(data, 'zerver_realmauditlog')
        re_map_foreign_keys(data, 'zerver_realmauditlog', 'realm', related_table='realm')
        re_map_foreign_keys(data, 'zerver_realmauditlog', 'modified_user', related_table='user_profile')
        re_map_foreign_keys(data, 'zerver_realmauditlog', 'acting_user', related_table='user_profile')
        re_map_foreign_keys(data, 'zerver_realmauditlog', 'modified_stream', related_table='stream')
        re_map_foreign_keys(data, 'zerver_realmauditlog', 'modified_user_group', related_table='usergroup')
        update_model_ids(RealmAuditLog, data, related_table='realmauditlog')
        bulk_import_model(data, RealmAuditLog)
    else:
        logging.info('about to call create_subscription_events')
        create_subscription_events(data=data, realm_id=realm.id)
        logging.info('done with create_subscription_events')
    if not RealmAuditLog.objects.filter(realm=realm, event_type=RealmAuditLog.REALM_CREATED).exists():
        RealmAuditLog.objects.create(realm=realm, event_type=RealmAuditLog.REALM_CREATED, event_time=realm.date_created, backfilled=True)
    if 'zerver_huddle' in data:
        process_huddle_hash(data, 'zerver_huddle')
        bulk_import_model(data, Huddle)
        for huddle in Huddle.objects.filter(recipient=None):
            recipient = Recipient.objects.get(type=Recipient.HUDDLE, type_id=huddle.id)
            huddle.recipient = recipient
            huddle.save(update_fields=['recipient'])
    if 'zerver_alertword' in data:
        re_map_foreign_keys(data, 'zerver_alertword', 'user_profile', related_table='user_profile')
        re_map_foreign_keys(data, 'zerver_alertword', 'realm', related_table='realm')
        update_model_ids(AlertWord, data, 'alertword')
        bulk_import_model(data, AlertWord)
    if 'zerver_userhotspot' in data:
        fix_datetime_fields(data, 'zerver_userhotspot')
        re_map_foreign_keys(data, 'zerver_userhotspot', 'user', related_table='user_profile')
        update_model_ids(UserHotspot, data, 'userhotspot')
        bulk_import_model(data, UserHotspot)
    if 'zerver_usertopic' in data:
        fix_datetime_fields(data, 'zerver_usertopic')
        re_map_foreign_keys(data, 'zerver_usertopic', 'user_profile', related_table='user_profile')
        re_map_foreign_keys(data, 'zerver_usertopic', 'stream', related_table='stream')
        re_map_foreign_keys(data, 'zerver_usertopic', 'recipient', related_table='recipient')
        update_model_ids(UserTopic, data, 'usertopic')
        bulk_import_model(data, UserTopic)
    if 'zerver_muteduser' in data:
        fix_datetime_fields(data, 'zerver_muteduser')
        re_map_foreign_keys(data, 'zerver_muteduser', 'user_profile', related_table='user_profile')
        re_map_foreign_keys(data, 'zerver_muteduser', 'muted_user', related_table='user_profile')
        update_model_ids(MutedUser, data, 'muteduser')
        bulk_import_model(data, MutedUser)
    if 'zerver_service' in data:
        re_map_foreign_keys(data, 'zerver_service', 'user_profile', related_table='user_profile')
        fix_service_tokens(data, 'zerver_service')
        update_model_ids(Service, data, 'service')
        bulk_import_model(data, Service)
    if 'zerver_usergroup' in data:
        re_map_foreign_keys(data, 'zerver_usergroupmembership', 'user_group', related_table='usergroup')
        re_map_foreign_keys(data, 'zerver_usergroupmembership', 'user_profile', related_table='user_profile')
        update_model_ids(UserGroupMembership, data, 'usergroupmembership')
        bulk_import_model(data, UserGroupMembership)
        re_map_foreign_keys(data, 'zerver_groupgroupmembership', 'supergroup', related_table='usergroup')
        re_map_foreign_keys(data, 'zerver_groupgroupmembership', 'subgroup', related_table='usergroup')
        update_model_ids(GroupGroupMembership, data, 'groupgroupmembership')
        bulk_import_model(data, GroupGroupMembership)
    if role_system_groups_dict is not None:
        add_users_to_system_user_groups(realm, user_profiles, role_system_groups_dict)
    if 'zerver_botstoragedata' in data:
        re_map_foreign_keys(data, 'zerver_botstoragedata', 'bot_profile', related_table='user_profile')
        update_model_ids(BotStorageData, data, 'botstoragedata')
        bulk_import_model(data, BotStorageData)
    if 'zerver_botconfigdata' in data:
        re_map_foreign_keys(data, 'zerver_botconfigdata', 'bot_profile', related_table='user_profile')
        update_model_ids(BotConfigData, data, 'botconfigdata')
        bulk_import_model(data, BotConfigData)
    if 'zerver_realmuserdefault' in data:
        re_map_foreign_keys(data, 'zerver_realmuserdefault', 'realm', related_table='realm')
        update_model_ids(RealmUserDefault, data, 'realmuserdefault')
        bulk_import_model(data, RealmUserDefault)
    if not RealmUserDefault.objects.filter(realm=realm).exists():
        RealmUserDefault.objects.create(realm=realm)
    fix_datetime_fields(data, 'zerver_userpresence')
    re_map_foreign_keys(data, 'zerver_userpresence', 'user_profile', related_table='user_profile')
    re_map_foreign_keys(data, 'zerver_userpresence', 'realm', related_table='realm')
    update_model_ids(UserPresence, data, 'user_presence')
    bulk_import_model(data, UserPresence)
    fix_datetime_fields(data, 'zerver_useractivity')
    re_map_foreign_keys(data, 'zerver_useractivity', 'user_profile', related_table='user_profile')
    re_map_foreign_keys(data, 'zerver_useractivity', 'client', related_table='client')
    update_model_ids(UserActivity, data, 'useractivity')
    bulk_import_model(data, UserActivity)
    fix_datetime_fields(data, 'zerver_useractivityinterval')
    re_map_foreign_keys(data, 'zerver_useractivityinterval', 'user_profile', related_table='user_profile')
    update_model_ids(UserActivityInterval, data, 'useractivityinterval')
    bulk_import_model(data, UserActivityInterval)
    re_map_foreign_keys(data, 'zerver_customprofilefield', 'realm', related_table='realm')
    update_model_ids(CustomProfileField, data, related_table='customprofilefield')
    bulk_import_model(data, CustomProfileField)
    re_map_foreign_keys(data, 'zerver_customprofilefieldvalue', 'user_profile', related_table='user_profile')
    re_map_foreign_keys(data, 'zerver_customprofilefieldvalue', 'field', related_table='customprofilefield')
    fix_customprofilefield(data)
    update_model_ids(CustomProfileFieldValue, data, related_table='customprofilefieldvalue')
    bulk_import_model(data, CustomProfileFieldValue)
    import_uploads(realm, os.path.join(import_dir, 'avatars'), processes, default_user_profile_id=None, processing_avatars=True)
    import_uploads(realm, os.path.join(import_dir, 'uploads'), processes, default_user_profile_id=None)
    if os.path.exists(os.path.join(import_dir, 'emoji')):
        import_uploads(realm, os.path.join(import_dir, 'emoji'), processes, default_user_profile_id=first_user_profile.id if first_user_profile else None, processing_emojis=True)
    if os.path.exists(os.path.join(import_dir, 'realm_icons')):
        import_uploads(realm, os.path.join(import_dir, 'realm_icons'), processes, default_user_profile_id=first_user_profile.id if first_user_profile else None, processing_realm_icons=True)
    sender_map = {user['id']: user for user in data['zerver_userprofile']}
    if 'zerver_scheduledmessage' in data:
        fix_datetime_fields(data, 'zerver_scheduledmessage')
        re_map_foreign_keys(data, 'zerver_scheduledmessage', 'sender', related_table='user_profile')
        re_map_foreign_keys(data, 'zerver_scheduledmessage', 'recipient', related_table='recipient')
        re_map_foreign_keys(data, 'zerver_scheduledmessage', 'sending_client', related_table='client')
        re_map_foreign_keys(data, 'zerver_scheduledmessage', 'stream', related_table='stream')
        re_map_foreign_keys(data, 'zerver_scheduledmessage', 'realm', related_table='realm')
        fix_upload_links(data, 'zerver_scheduledmessage')
        fix_message_rendered_content(realm=realm, sender_map=sender_map, messages=data['zerver_scheduledmessage'])
        update_model_ids(ScheduledMessage, data, 'scheduledmessage')
        bulk_import_model(data, ScheduledMessage)
    import_message_data(realm=realm, sender_map=sender_map, import_dir=import_dir)
    re_map_foreign_keys(data, 'zerver_reaction', 'message', related_table='message')
    re_map_foreign_keys(data, 'zerver_reaction', 'user_profile', related_table='user_profile')
    re_map_realm_emoji_codes(data, table_name='zerver_reaction')
    update_model_ids(Reaction, data, 'reaction')
    bulk_import_model(data, Reaction)
    update_first_message_id_query = SQL('\n    UPDATE zerver_stream\n    SET first_message_id = subquery.first_message_id\n    FROM (\n        SELECT r.type_id id, min(m.id) first_message_id\n        FROM zerver_message m\n        JOIN zerver_recipient r ON\n        r.id = m.recipient_id\n        WHERE r.type = 2 AND m.realm_id = %(realm_id)s\n        GROUP BY r.type_id\n        ) AS subquery\n    WHERE zerver_stream.id = subquery.id\n    ')
    with connection.cursor() as cursor:
        cursor.execute(update_first_message_id_query, {'realm_id': realm.id})
    if 'zerver_userstatus' in data:
        fix_datetime_fields(data, 'zerver_userstatus')
        re_map_foreign_keys(data, 'zerver_userstatus', 'user_profile', related_table='user_profile')
        re_map_foreign_keys(data, 'zerver_userstatus', 'client', related_table='client')
        update_model_ids(UserStatus, data, 'userstatus')
        re_map_realm_emoji_codes(data, table_name='zerver_userstatus')
        bulk_import_model(data, UserStatus)
    fn = os.path.join(import_dir, 'attachment.json')
    if not os.path.exists(fn):
        raise Exception('Missing attachment.json file!')
    logging.info('Importing attachment data from %s', fn)
    with open(fn, 'rb') as f:
        attachment_data = orjson.loads(f.read())
    import_attachments(attachment_data)
    import_analytics_data(realm=realm, import_dir=import_dir)
    if settings.BILLING_ENABLED:
        do_change_realm_plan_type(realm, Realm.PLAN_TYPE_LIMITED, acting_user=None)
    else:
        do_change_realm_plan_type(realm, Realm.PLAN_TYPE_SELF_HOSTED, acting_user=None)
    realm.deactivated = data['zerver_realm'][0]['deactivated']
    realm.save()
    return realm

def update_message_foreign_keys(import_dir: Path, sort_by_date: bool) -> None:
    if False:
        return 10
    old_id_list = get_incoming_message_ids(import_dir=import_dir, sort_by_date=sort_by_date)
    count = len(old_id_list)
    new_id_list = allocate_ids(model_class=Message, count=count)
    for (old_id, new_id) in zip(old_id_list, new_id_list):
        update_id_map(table='message', old_id=old_id, new_id=new_id)

def get_incoming_message_ids(import_dir: Path, sort_by_date: bool) -> List[int]:
    if False:
        print('Hello World!')
    "\n    This function reads in our entire collection of message\n    ids, which can be millions of integers for some installations.\n    And then we sort the list.  This is necessary to ensure\n    that the sort order of incoming ids matches the sort order\n    of date_sent, which isn't always guaranteed by our\n    utilities that convert third party chat data.  We also\n    need to move our ids to a new range if we're dealing\n    with a server that has data for other realms.\n    "
    if sort_by_date:
        tups: List[Tuple[int, int]] = []
    else:
        message_ids: List[int] = []
    dump_file_id = 1
    while True:
        message_filename = os.path.join(import_dir, f'messages-{dump_file_id:06}.json')
        if not os.path.exists(message_filename):
            break
        with open(message_filename, 'rb') as f:
            data = orjson.loads(f.read())
        del data['zerver_usermessage']
        for row in data['zerver_message']:
            message_id = row['id']
            if sort_by_date:
                date_sent = int(row['date_sent'])
                tup = (date_sent, message_id)
                tups.append(tup)
            else:
                message_ids.append(message_id)
        dump_file_id += 1
    if sort_by_date:
        tups.sort()
        message_ids = [tup[1] for tup in tups]
    return message_ids

def import_message_data(realm: Realm, sender_map: Dict[int, Record], import_dir: Path) -> None:
    if False:
        i = 10
        return i + 15
    dump_file_id = 1
    while True:
        message_filename = os.path.join(import_dir, f'messages-{dump_file_id:06}.json')
        if not os.path.exists(message_filename):
            break
        with open(message_filename, 'rb') as f:
            data = orjson.loads(f.read())
        logging.info('Importing message dump %s', message_filename)
        re_map_foreign_keys(data, 'zerver_message', 'sender', related_table='user_profile')
        re_map_foreign_keys(data, 'zerver_message', 'recipient', related_table='recipient')
        re_map_foreign_keys(data, 'zerver_message', 'sending_client', related_table='client')
        fix_datetime_fields(data, 'zerver_message')
        fix_upload_links(data, 'zerver_message')
        message_id_map = ID_MAP['message']
        for row in data['zerver_message']:
            del row['realm']
            row['realm_id'] = realm.id
            row['id'] = message_id_map[row['id']]
        for row in data['zerver_usermessage']:
            assert row['message'] in message_id_map
        fix_message_rendered_content(realm=realm, sender_map=sender_map, messages=data['zerver_message'])
        logging.info('Successfully rendered Markdown for message batch')
        bulk_import_model(data, Message)
        re_map_foreign_keys(data, 'zerver_usermessage', 'message', related_table='message')
        re_map_foreign_keys(data, 'zerver_usermessage', 'user_profile', related_table='user_profile')
        fix_bitfield_keys(data, 'zerver_usermessage', 'flags')
        bulk_import_user_message_data(data, dump_file_id)
        dump_file_id += 1

def import_attachments(data: TableData) -> None:
    if False:
        while True:
            i = 10
    fix_datetime_fields(data, 'zerver_attachment')
    re_map_foreign_keys(data, 'zerver_attachment', 'owner', related_table='user_profile')
    re_map_foreign_keys(data, 'zerver_attachment', 'realm', related_table='realm')
    parent_model = Attachment
    parent_db_table_name = 'zerver_attachment'
    parent_singular = 'attachment'
    parent_id = 'attachment_id'
    update_model_ids(parent_model, data, 'attachment')

    def format_m2m_data(child_singular: str, child_plural: str, m2m_table_name: str, child_id: str) -> Tuple[str, List[Record], str]:
        if False:
            for i in range(10):
                print('nop')
        m2m_rows = [{parent_singular: parent_row['id'], child_singular: ID_MAP[child_singular][fk_id]} for parent_row in data[parent_db_table_name] for fk_id in parent_row[child_plural]]
        m2m_data: TableData = {m2m_table_name: m2m_rows}
        convert_to_id_fields(m2m_data, m2m_table_name, parent_singular)
        convert_to_id_fields(m2m_data, m2m_table_name, child_singular)
        m2m_rows = m2m_data[m2m_table_name]
        for parent_row in data[parent_db_table_name]:
            del parent_row[child_plural]
        return (m2m_table_name, m2m_rows, child_id)
    messages_m2m_tuple = format_m2m_data('message', 'messages', 'zerver_attachment_messages', 'message_id')
    scheduled_messages_m2m_tuple = format_m2m_data('scheduledmessage', 'scheduled_messages', 'zerver_attachment_scheduled_messages', 'scheduledmessage_id')
    for attachment in data[parent_db_table_name]:
        attachment['path_id'] = path_maps['attachment_path'][attachment['path_id']]
    bulk_import_model(data, parent_model)
    with connection.cursor() as cursor:
        for (m2m_table_name, m2m_rows, child_id) in [messages_m2m_tuple, scheduled_messages_m2m_tuple]:
            sql_template = SQL('\n                INSERT INTO {m2m_table_name} ({parent_id}, {child_id}) VALUES %s\n            ').format(m2m_table_name=Identifier(m2m_table_name), parent_id=Identifier(parent_id), child_id=Identifier(child_id))
            tups = [(row[parent_id], row[child_id]) for row in m2m_rows]
            execute_values(cursor.cursor, sql_template, tups)
            logging.info('Successfully imported M2M table %s', m2m_table_name)

def import_analytics_data(realm: Realm, import_dir: Path) -> None:
    if False:
        print('Hello World!')
    analytics_filename = os.path.join(import_dir, 'analytics.json')
    if not os.path.exists(analytics_filename):
        return
    logging.info('Importing analytics data from %s', analytics_filename)
    with open(analytics_filename, 'rb') as f:
        data = orjson.loads(f.read())
    fix_datetime_fields(data, 'analytics_realmcount')
    re_map_foreign_keys(data, 'analytics_realmcount', 'realm', related_table='realm')
    update_model_ids(RealmCount, data, 'analytics_realmcount')
    bulk_import_model(data, RealmCount)
    fix_datetime_fields(data, 'analytics_usercount')
    re_map_foreign_keys(data, 'analytics_usercount', 'realm', related_table='realm')
    re_map_foreign_keys(data, 'analytics_usercount', 'user', related_table='user_profile')
    update_model_ids(UserCount, data, 'analytics_usercount')
    bulk_import_model(data, UserCount)
    fix_datetime_fields(data, 'analytics_streamcount')
    re_map_foreign_keys(data, 'analytics_streamcount', 'realm', related_table='realm')
    re_map_foreign_keys(data, 'analytics_streamcount', 'stream', related_table='stream')
    update_model_ids(StreamCount, data, 'analytics_streamcount')
    bulk_import_model(data, StreamCount)

def add_users_to_system_user_groups(realm: Realm, user_profiles: List[UserProfile], role_system_groups_dict: Dict[int, UserGroup]) -> None:
    if False:
        for i in range(10):
            print('nop')
    full_members_system_group = UserGroup.objects.get(name=SystemGroups.FULL_MEMBERS, realm=realm, is_system_group=True)
    usergroup_memberships = []
    for user_profile in user_profiles:
        user_group = role_system_groups_dict[user_profile.role]
        usergroup_memberships.append(UserGroupMembership(user_profile=user_profile, user_group=user_group))
        if user_profile.role == UserProfile.ROLE_MEMBER and (not user_profile.is_provisional_member):
            usergroup_memberships.append(UserGroupMembership(user_profile=user_profile, user_group=full_members_system_group))
    UserGroupMembership.objects.bulk_create(usergroup_memberships)
    now = timezone_now()
    RealmAuditLog.objects.bulk_create((RealmAuditLog(realm=realm, modified_user=membership.user_profile, modified_user_group=membership.user_group, event_type=RealmAuditLog.USER_GROUP_DIRECT_USER_MEMBERSHIP_ADDED, event_time=now, acting_user=None) for membership in usergroup_memberships))