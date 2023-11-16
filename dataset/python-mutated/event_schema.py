from typing import Dict, List, Sequence, Set, Tuple, Union
from zerver.lib.data_types import DictType, EnumType, Equals, ListType, NumberType, OptionalType, StringDictType, TupleType, UnionType, UrlType, check_data, event_dict_type, make_checker
from zerver.lib.topic import ORIG_TOPIC, TOPIC_LINKS, TOPIC_NAME
from zerver.models import Realm, RealmUserDefault, Stream, UserProfile
default_stream_fields = [('can_remove_subscribers_group', int), ('date_created', int), ('description', str), ('first_message_id', OptionalType(int)), ('history_public_to_subscribers', bool), ('invite_only', bool), ('is_announcement_only', bool), ('is_web_public', bool), ('message_retention_days', OptionalType(int)), ('name', str), ('rendered_description', str), ('stream_id', int), ('stream_post_policy', int)]
basic_stream_fields = [*default_stream_fields, ('stream_weekly_traffic', OptionalType(int))]
subscription_fields: Sequence[Tuple[str, object]] = [*basic_stream_fields, ('audible_notifications', OptionalType(bool)), ('color', str), ('desktop_notifications', OptionalType(bool)), ('email_address', str), ('email_notifications', OptionalType(bool)), ('in_home_view', bool), ('is_muted', bool), ('pin_to_top', bool), ('push_notifications', OptionalType(bool)), ('subscribers', ListType(int)), ('wildcard_mentions_notify', OptionalType(bool))]
value_type = UnionType([bool, int, str])
optional_value_type = UnionType([bool, int, str, Equals(None)])
alert_words_event = event_dict_type(required_keys=[('type', Equals('alert_words')), ('alert_words', ListType(str))])
check_alert_words = make_checker(alert_words_event)
attachment_message_type = DictType(required_keys=[('id', int), ('date_sent', int)])
attachment_type = DictType(required_keys=[('id', int), ('name', str), ('size', int), ('path_id', str), ('create_time', int), ('messages', ListType(attachment_message_type))])
attachment_add_event = event_dict_type(required_keys=[('type', Equals('attachment')), ('op', Equals('add')), ('attachment', attachment_type), ('upload_space_used', int)])
check_attachment_add = make_checker(attachment_add_event)
attachment_remove_event = event_dict_type(required_keys=[('type', Equals('attachment')), ('op', Equals('remove')), ('attachment', DictType([('id', int)])), ('upload_space_used', int)])
check_attachment_remove = make_checker(attachment_remove_event)
attachment_update_event = event_dict_type(required_keys=[('type', Equals('attachment')), ('op', Equals('update')), ('attachment', attachment_type), ('upload_space_used', int)])
check_attachment_update = make_checker(attachment_update_event)
custom_profile_field_type = DictType(required_keys=[('id', int), ('type', int), ('name', str), ('hint', str), ('field_data', str), ('order', int)], optional_keys=[('display_in_profile_summary', bool)])
custom_profile_fields_event = event_dict_type(required_keys=[('type', Equals('custom_profile_fields')), ('fields', ListType(custom_profile_field_type))])
check_custom_profile_fields = make_checker(custom_profile_fields_event)
_check_stream_group = DictType(required_keys=[('name', str), ('id', int), ('description', str), ('streams', ListType(DictType(default_stream_fields)))])
default_stream_groups_event = event_dict_type(required_keys=[('type', Equals('default_stream_groups')), ('default_stream_groups', ListType(_check_stream_group))])
check_default_stream_groups = make_checker(default_stream_groups_event)
default_streams_event = event_dict_type(required_keys=[('type', Equals('default_streams')), ('default_streams', ListType(DictType(default_stream_fields)))])
check_default_streams = make_checker(default_streams_event)
delete_message_event = event_dict_type(required_keys=[('type', Equals('delete_message')), ('message_type', EnumType(['private', 'stream']))], optional_keys=[('message_id', int), ('message_ids', ListType(int)), ('stream_id', int), ('topic', str)])
_check_delete_message = make_checker(delete_message_event)

def check_delete_message(var_name: str, event: Dict[str, object], message_type: str, num_message_ids: int, is_legacy: bool) -> None:
    if False:
        return 10
    _check_delete_message(var_name, event)
    keys = {'id', 'type', 'message_type'}
    assert event['message_type'] == message_type
    if message_type == 'stream':
        keys |= {'stream_id', 'topic'}
    elif message_type == 'private':
        pass
    else:
        raise AssertionError('unexpected message_type')
    if is_legacy:
        assert num_message_ids == 1
        keys.add('message_id')
    else:
        assert isinstance(event['message_ids'], list)
        assert num_message_ids == len(event['message_ids'])
        keys.add('message_ids')
    assert set(event.keys()) == keys
draft_fields = DictType(required_keys=[('id', int), ('type', EnumType(['', 'stream', 'private'])), ('to', ListType(int)), ('topic', str), ('content', str)], optional_keys=[('timestamp', int)])
drafts_add_event = event_dict_type(required_keys=[('type', Equals('drafts')), ('op', Equals('add')), ('drafts', ListType(draft_fields))])
check_draft_add = make_checker(drafts_add_event)
drafts_update_event = event_dict_type(required_keys=[('type', Equals('drafts')), ('op', Equals('update')), ('draft', draft_fields)])
check_draft_update = make_checker(drafts_update_event)
drafts_remove_event = event_dict_type(required_keys=[('type', Equals('drafts')), ('op', Equals('remove')), ('draft_id', int)])
check_draft_remove = make_checker(drafts_remove_event)
has_zoom_token_event = event_dict_type(required_keys=[('type', Equals('has_zoom_token')), ('value', bool)])
_check_has_zoom_token = make_checker(has_zoom_token_event)

def check_has_zoom_token(var_name: str, event: Dict[str, object], value: bool) -> None:
    if False:
        print('Hello World!')
    _check_has_zoom_token(var_name, event)
    assert event['value'] == value
heartbeat_event = event_dict_type(required_keys=[('type', Equals('heartbeat'))])
_check_heartbeat = make_checker(heartbeat_event)

def check_heartbeat(var_name: str, event: Dict[str, object]) -> None:
    if False:
        for i in range(10):
            print('nop')
    _check_heartbeat(var_name, event)
_hotspot = DictType(required_keys=[('name', str), ('title', str), ('description', str), ('delay', NumberType())])
hotspots_event = event_dict_type(required_keys=[('type', Equals('hotspots')), ('hotspots', ListType(_hotspot))])
check_hotspots = make_checker(hotspots_event)
invites_changed_event = event_dict_type(required_keys=[('type', Equals('invites_changed'))])
check_invites_changed = make_checker(invites_changed_event)
muted_topic_type = TupleType([str, str, int])
muted_topics_event = event_dict_type(required_keys=[('type', Equals('muted_topics')), ('muted_topics', ListType(muted_topic_type))])
check_muted_topics = make_checker(muted_topics_event)
user_topic_event = DictType(required_keys=[('id', int), ('type', Equals('user_topic')), ('stream_id', int), ('topic_name', str), ('last_updated', int), ('visibility_policy', int)])
check_user_topic = make_checker(user_topic_event)
muted_user_type = DictType(required_keys=[('id', int), ('timestamp', int)])
muted_users_event = event_dict_type(required_keys=[('type', Equals('muted_users')), ('muted_users', ListType(muted_user_type))])
check_muted_users = make_checker(muted_users_event)
_check_topic_links = DictType(required_keys=[('text', str), ('url', str)])
message_fields = [('avatar_url', OptionalType(str)), ('client', str), ('content', str), ('content_type', Equals('text/html')), ('display_recipient', str), ('id', int), ('is_me_message', bool), ('reactions', ListType(dict)), ('recipient_id', int), ('sender_realm_str', str), ('sender_email', str), ('sender_full_name', str), ('sender_id', int), ('stream_id', int), (TOPIC_NAME, str), (TOPIC_LINKS, ListType(_check_topic_links)), ('submessages', ListType(dict)), ('timestamp', int), ('type', str)]
message_event = event_dict_type(required_keys=[('type', Equals('message')), ('flags', ListType(str)), ('message', DictType(message_fields))])
check_message = make_checker(message_event)
presence_type = DictType(required_keys=[('status', EnumType(['active', 'idle'])), ('timestamp', int), ('client', str), ('pushable', bool)])
presence_event = event_dict_type(required_keys=[('type', Equals('presence')), ('user_id', int), ('server_timestamp', NumberType()), ('presence', StringDictType(presence_type))], optional_keys=[('email', str)])
_check_presence = make_checker(presence_event)

def check_presence(var_name: str, event: Dict[str, object], has_email: bool, presence_key: str, status: str) -> None:
    if False:
        i = 10
        return i + 15
    _check_presence(var_name, event)
    assert ('email' in event) == has_email
    assert isinstance(event['presence'], dict)
    [(event_presence_key, event_presence_value)] = event['presence'].items()
    assert event_presence_key == presence_key
    assert event_presence_value['status'] == status
reaction_legacy_user_type = DictType(required_keys=[('email', str), ('full_name', str), ('user_id', int)])
reaction_add_event = event_dict_type(required_keys=[('type', Equals('reaction')), ('op', Equals('add')), ('message_id', int), ('emoji_name', str), ('emoji_code', str), ('reaction_type', EnumType(['unicode_emoji', 'realm_emoji', 'zulip_extra_emoji'])), ('user_id', int), ('user', reaction_legacy_user_type)])
check_reaction_add = make_checker(reaction_add_event)
reaction_remove_event = event_dict_type(required_keys=[('type', Equals('reaction')), ('op', Equals('remove')), ('message_id', int), ('emoji_name', str), ('emoji_code', str), ('reaction_type', EnumType(['unicode_emoji', 'realm_emoji', 'zulip_extra_emoji'])), ('user_id', int), ('user', reaction_legacy_user_type)])
check_reaction_remove = make_checker(reaction_remove_event)
realm_deactivated_event = event_dict_type(required_keys=[('type', Equals('realm')), ('op', Equals('deactivated')), ('realm_id', int)])
check_realm_deactivated = make_checker(realm_deactivated_event)
bot_services_outgoing_type = DictType(required_keys=[('base_url', UrlType()), ('interface', int), ('token', str)])
config_data_schema = StringDictType(str)
bot_services_embedded_type = DictType(required_keys=[('service_name', str), ('config_data', config_data_schema)])
bot_services_type = ListType(UnionType([bot_services_outgoing_type, bot_services_embedded_type]))
bot_type = DictType(required_keys=[('user_id', int), ('api_key', str), ('avatar_url', str), ('bot_type', int), ('default_all_public_streams', bool), ('default_events_register_stream', OptionalType(str)), ('default_sending_stream', OptionalType(str)), ('email', str), ('full_name', str), ('is_active', bool), ('owner_id', int), ('services', bot_services_type)])
realm_bot_add_event = event_dict_type(required_keys=[('type', Equals('realm_bot')), ('op', Equals('add')), ('bot', bot_type)])
_check_realm_bot_add = make_checker(realm_bot_add_event)

def check_realm_bot_add(var_name: str, event: Dict[str, object]) -> None:
    if False:
        print('Hello World!')
    _check_realm_bot_add(var_name, event)
    assert isinstance(event['bot'], dict)
    bot_type = event['bot']['bot_type']
    services_field = f"{var_name}['bot']['services']"
    services = event['bot']['services']
    if bot_type == UserProfile.DEFAULT_BOT:
        check_data(Equals([]), services_field, services)
    elif bot_type == UserProfile.OUTGOING_WEBHOOK_BOT:
        check_data(ListType(bot_services_outgoing_type, length=1), services_field, services)
    elif bot_type == UserProfile.EMBEDDED_BOT:
        check_data(ListType(bot_services_embedded_type, length=1), services_field, services)
    else:
        raise AssertionError(f'Unknown bot_type: {bot_type}')
bot_type_for_delete = DictType(required_keys=[('user_id', int)])
realm_bot_delete_event = event_dict_type(required_keys=[('type', Equals('realm_bot')), ('op', Equals('delete')), ('bot', bot_type_for_delete)])
check_realm_bot_delete = make_checker(realm_bot_delete_event)
bot_type_for_remove = DictType(required_keys=[('full_name', str), ('user_id', int)])
bot_type_for_update = DictType(required_keys=[('user_id', int)], optional_keys=[('api_key', str), ('avatar_url', str), ('default_all_public_streams', bool), ('default_events_register_stream', OptionalType(str)), ('default_sending_stream', OptionalType(str)), ('full_name', str), ('is_active', bool), ('owner_id', int), ('services', bot_services_type)])
realm_bot_update_event = event_dict_type(required_keys=[('type', Equals('realm_bot')), ('op', Equals('update')), ('bot', bot_type_for_update)])
_check_realm_bot_update = make_checker(realm_bot_update_event)

def check_realm_bot_update(var_name: str, event: Dict[str, object], field: str) -> None:
    if False:
        print('Hello World!')
    _check_realm_bot_update(var_name, event)
    assert isinstance(event['bot'], dict)
    assert {'user_id', field} == set(event['bot'].keys())
realm_domain_type = DictType(required_keys=[('domain', str), ('allow_subdomains', bool)])
realm_domains_add_event = event_dict_type(required_keys=[('type', Equals('realm_domains')), ('op', Equals('add')), ('realm_domain', realm_domain_type)])
check_realm_domains_add = make_checker(realm_domains_add_event)
realm_domains_change_event = event_dict_type(required_keys=[('type', Equals('realm_domains')), ('op', Equals('change')), ('realm_domain', realm_domain_type)])
check_realm_domains_change = make_checker(realm_domains_change_event)
realm_domains_remove_event = event_dict_type(required_keys=[('type', Equals('realm_domains')), ('op', Equals('remove')), ('domain', str)])
check_realm_domains_remove = make_checker(realm_domains_remove_event)
realm_playground_type = DictType(required_keys=[('id', int), ('name', str), ('pygments_language', str), ('url_template', str)])
realm_playgrounds_event = event_dict_type(required_keys=[('type', Equals('realm_playgrounds')), ('realm_playgrounds', ListType(realm_playground_type))])
_check_realm_playgrounds = make_checker(realm_playgrounds_event)

def check_realm_playgrounds(var_name: str, event: Dict[str, object]) -> None:
    if False:
        while True:
            i = 10
    _check_realm_playgrounds(var_name, event)
    assert isinstance(event['realm_playgrounds'], list)
realm_emoji_type = DictType(required_keys=[('id', str), ('name', str), ('source_url', str), ('deactivated', bool), ('author_id', int), ('still_url', OptionalType(str))])
realm_emoji_update_event = event_dict_type(required_keys=[('type', Equals('realm_emoji')), ('op', Equals('update')), ('realm_emoji', StringDictType(realm_emoji_type))])
_check_realm_emoji_update = make_checker(realm_emoji_update_event)

def check_realm_emoji_update(var_name: str, event: Dict[str, object]) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    The way we send realm emojis is kinda clumsy--we\n    send a dict mapping the emoji id to a sub_dict with\n    the fields (including the id).  Ideally we can streamline\n    this and just send a list of dicts.  The clients can make\n    a Map as needed.\n    '
    _check_realm_emoji_update(var_name, event)
    assert isinstance(event['realm_emoji'], dict)
    for (k, v) in event['realm_emoji'].items():
        assert v['id'] == k
export_type = DictType(required_keys=[('id', int), ('export_time', NumberType()), ('acting_user_id', int), ('export_url', OptionalType(str)), ('deleted_timestamp', OptionalType(NumberType())), ('failed_timestamp', OptionalType(NumberType())), ('pending', bool)])
realm_export_event = event_dict_type(required_keys=[('type', Equals('realm_export')), ('exports', ListType(export_type))])
_check_realm_export = make_checker(realm_export_event)

def check_realm_export(var_name: str, event: Dict[str, object], has_export_url: bool, has_deleted_timestamp: bool, has_failed_timestamp: bool) -> None:
    if False:
        print('Hello World!')
    _check_realm_export(var_name, event)
    assert isinstance(event['exports'], list)
    assert len(event['exports']) == 1
    export = event['exports'][0]
    assert has_export_url == (export['export_url'] is not None)
    assert has_deleted_timestamp == (export['deleted_timestamp'] is not None)
    assert has_failed_timestamp == (export['failed_timestamp'] is not None)
realm_linkifier_type = DictType(required_keys=[('pattern', str), ('url_template', str), ('id', int)])
realm_linkifiers_event = event_dict_type([('type', Equals('realm_linkifiers')), ('realm_linkifiers', ListType(realm_linkifier_type))])
check_realm_linkifiers = make_checker(realm_linkifiers_event)
plan_type_extra_data_type = DictType(required_keys=[('upload_quota', int)])
'\nrealm/update events are flexible for values;\nwe will use a more strict checker to check\ntypes in a context-specific manner\n'
realm_update_event = event_dict_type(required_keys=[('type', Equals('realm')), ('op', Equals('update')), ('property', str), ('value', value_type)], optional_keys=[('extra_data', plan_type_extra_data_type)])
_check_realm_update = make_checker(realm_update_event)

def check_realm_update(var_name: str, event: Dict[str, object], prop: str) -> None:
    if False:
        return 10
    '\n    Realm updates have these two fields:\n\n        property\n        value\n\n    We check not only the basic schema, but also that\n    the value people actually matches the type from\n    Realm.property_types that we have configured\n    for the property.\n    '
    _check_realm_update(var_name, event)
    assert prop == event['property']
    value = event['value']
    if prop == 'plan_type':
        assert isinstance(value, int)
        assert 'extra_data' in event
        return
    assert 'extra_data' not in event
    if prop in ['notifications_stream_id', 'signup_notifications_stream_id', 'org_type']:
        assert isinstance(value, int)
        return
    property_type = Realm.property_types[prop]
    if property_type in (bool, int, str):
        assert isinstance(value, property_type)
    elif property_type == (int, type(None)):
        assert isinstance(value, int)
    elif property_type == (str, type(None)):
        assert isinstance(value, str)
    else:
        raise AssertionError(f'Unexpected property type {property_type}')
realm_user_settings_defaults_update_event = event_dict_type(required_keys=[('type', Equals('realm_user_settings_defaults')), ('op', Equals('update')), ('property', str), ('value', value_type)])
_check_realm_default_update = make_checker(realm_user_settings_defaults_update_event)

def check_realm_default_update(var_name: str, event: Dict[str, object], prop: str) -> None:
    if False:
        return 10
    _check_realm_default_update(var_name, event)
    assert prop == event['property']
    assert prop != 'default_language'
    assert prop in RealmUserDefault.property_types
    prop_type = RealmUserDefault.property_types[prop]
    assert isinstance(event['value'], prop_type)
authentication_dict = DictType(required_keys=[('Google', bool), ('Dev', bool), ('LDAP', bool), ('GitHub', bool), ('Email', bool)])
authentication_data = DictType(required_keys=[('authentication_methods', authentication_dict)])
icon_data = DictType(required_keys=[('icon_url', str), ('icon_source', str)])
logo_data = DictType(required_keys=[('logo_url', str), ('logo_source', str)])
allow_message_editing_data = DictType(required_keys=[('allow_message_editing', bool)])
message_content_edit_limit_seconds_data = DictType(required_keys=[('message_content_edit_limit_seconds', OptionalType(int))])
edit_topic_policy_data = DictType(required_keys=[('edit_topic_policy', int)])
night_logo_data = DictType(required_keys=[('night_logo_url', str), ('night_logo_source', str)])
group_setting_update_data_type = DictType(required_keys=[], optional_keys=[('create_multiuse_invite_group', int), ('can_access_all_users_group', int)])
update_dict_data = UnionType([allow_message_editing_data, authentication_data, edit_topic_policy_data, icon_data, logo_data, message_content_edit_limit_seconds_data, night_logo_data, group_setting_update_data_type])
realm_update_dict_event = event_dict_type(required_keys=[('type', Equals('realm')), ('op', Equals('update_dict')), ('property', EnumType(['default', 'icon', 'logo', 'night_logo'])), ('data', update_dict_data)])
_check_realm_update_dict = make_checker(realm_update_dict_event)

def check_realm_update_dict(var_name: str, event: Dict[str, object]) -> None:
    if False:
        i = 10
        return i + 15
    _check_realm_update_dict(var_name, event)
    if event['property'] == 'default':
        assert isinstance(event['data'], dict)
        if 'allow_message_editing' in event['data']:
            sub_type = allow_message_editing_data
        elif 'message_content_edit_limit_seconds' in event['data']:
            sub_type = message_content_edit_limit_seconds_data
        elif 'edit_topic_policy' in event['data']:
            sub_type = edit_topic_policy_data
        elif 'authentication_methods' in event['data']:
            sub_type = authentication_data
        elif any((setting_name in event['data'] for setting_name in Realm.REALM_PERMISSION_GROUP_SETTINGS)):
            sub_type = group_setting_update_data_type
        else:
            raise AssertionError('unhandled fields in data')
    elif event['property'] == 'icon':
        sub_type = icon_data
    elif event['property'] == 'logo':
        sub_type = logo_data
    elif event['property'] == 'night_logo':
        sub_type = night_logo_data
    else:
        raise AssertionError("unhandled property: {event['property']}")
    check_data(sub_type, f"{var_name}['data']", event['data'])
realm_user_type = DictType(required_keys=[('user_id', int), ('email', str), ('avatar_url', OptionalType(str)), ('avatar_version', int), ('full_name', str), ('is_admin', bool), ('is_billing_admin', bool), ('is_owner', bool), ('is_bot', bool), ('is_guest', bool), ('role', EnumType(UserProfile.ROLE_TYPES)), ('is_active', bool), ('profile_data', StringDictType(dict)), ('timezone', str), ('date_joined', str), ('delivery_email', OptionalType(str))])
realm_user_add_event = event_dict_type(required_keys=[('type', Equals('realm_user')), ('op', Equals('add')), ('person', realm_user_type)])
check_realm_user_add = make_checker(realm_user_add_event)
removed_user_type = DictType(required_keys=[('user_id', int), ('full_name', str)])
custom_profile_field_type = DictType(required_keys=[('id', int), ('value', OptionalType(str))], optional_keys=[('rendered_value', str)])
realm_user_person_types = dict(avatar_fields=DictType(required_keys=[('user_id', int), ('avatar_source', str), ('avatar_url', OptionalType(str)), ('avatar_url_medium', OptionalType(str)), ('avatar_version', int)]), bot_owner_id=DictType(required_keys=[('user_id', int), ('bot_owner_id', int)]), custom_profile_field=DictType(required_keys=[('user_id', int), ('custom_profile_field', custom_profile_field_type)]), delivery_email=DictType(required_keys=[('user_id', int), ('delivery_email', OptionalType(str))]), email=DictType(required_keys=[('user_id', int), ('new_email', str)]), full_name=DictType(required_keys=[('user_id', int), ('full_name', str)]), is_billing_admin=DictType(required_keys=[('user_id', int), ('is_billing_admin', bool)]), role=DictType(required_keys=[('user_id', int), ('role', EnumType(UserProfile.ROLE_TYPES))]), timezone=DictType(required_keys=[('user_id', int), ('email', str), ('timezone', str)]), is_active=DictType(required_keys=[('user_id', int), ('is_active', bool)]))
realm_user_update_event = event_dict_type(required_keys=[('type', Equals('realm_user')), ('op', Equals('update')), ('person', UnionType(list(realm_user_person_types.values())))])
_check_realm_user_update = make_checker(realm_user_update_event)

def check_realm_user_update(var_name: str, event: Dict[str, object], person_flavor: str) -> None:
    if False:
        print('Hello World!')
    _check_realm_user_update(var_name, event)
    check_data(realm_user_person_types[person_flavor], f"{var_name}['person']", event['person'])
restart_event = event_dict_type(required_keys=[('type', Equals('restart')), ('zulip_version', str), ('zulip_merge_base', str), ('zulip_feature_level', int), ('server_generation', int), ('immediate', bool)])
check_restart_event = make_checker(restart_event)
scheduled_message_fields = DictType(required_keys=[('scheduled_message_id', int), ('type', EnumType(['stream', 'private'])), ('to', UnionType([ListType(int), int])), ('content', str), ('rendered_content', str), ('scheduled_delivery_timestamp', int), ('failed', bool)], optional_keys=[('topic', str)])
scheduled_messages_add_event = event_dict_type(required_keys=[('type', Equals('scheduled_messages')), ('op', Equals('add')), ('scheduled_messages', ListType(scheduled_message_fields))])
check_scheduled_message_add = make_checker(scheduled_messages_add_event)
scheduled_messages_update_event = event_dict_type(required_keys=[('type', Equals('scheduled_messages')), ('op', Equals('update')), ('scheduled_message', scheduled_message_fields)])
check_scheduled_message_update = make_checker(scheduled_messages_update_event)
scheduled_messages_remove_event = event_dict_type(required_keys=[('type', Equals('scheduled_messages')), ('op', Equals('remove')), ('scheduled_message_id', int)])
check_scheduled_message_remove = make_checker(scheduled_messages_remove_event)
stream_create_event = event_dict_type(required_keys=[('type', Equals('stream')), ('op', Equals('create')), ('streams', ListType(DictType(basic_stream_fields)))])
check_stream_create = make_checker(stream_create_event)
stream_delete_event = event_dict_type(required_keys=[('type', Equals('stream')), ('op', Equals('delete')), ('streams', ListType(DictType(basic_stream_fields)))])
check_stream_delete = make_checker(stream_delete_event)
stream_update_event = event_dict_type(required_keys=[('type', Equals('stream')), ('op', Equals('update')), ('property', str), ('value', optional_value_type), ('name', str), ('stream_id', int)], optional_keys=[('rendered_description', str), ('history_public_to_subscribers', bool), ('is_web_public', bool)])
_check_stream_update = make_checker(stream_update_event)

def check_stream_update(var_name: str, event: Dict[str, object]) -> None:
    if False:
        print('Hello World!')
    _check_stream_update(var_name, event)
    prop = event['property']
    value = event['value']
    extra_keys = set(event.keys()) - {'id', 'type', 'op', 'property', 'value', 'name', 'stream_id'}
    if prop == 'description':
        assert extra_keys == {'rendered_description'}
        assert isinstance(value, str)
    elif prop == 'email_address':
        assert extra_keys == set()
        assert isinstance(value, str)
    elif prop == 'invite_only':
        assert extra_keys == {'history_public_to_subscribers', 'is_web_public'}
        assert isinstance(value, bool)
    elif prop == 'message_retention_days':
        assert extra_keys == set()
        if value is not None:
            assert isinstance(value, int)
    elif prop == 'name':
        assert extra_keys == set()
        assert isinstance(value, str)
    elif prop == 'stream_post_policy':
        assert extra_keys == set()
        assert value in Stream.STREAM_POST_POLICY_TYPES
    elif prop == 'can_remove_subscribers_group':
        assert extra_keys == set()
        assert isinstance(value, int)
    else:
        raise AssertionError(f'Unknown property: {prop}')
submessage_event = event_dict_type(required_keys=[('type', Equals('submessage')), ('message_id', int), ('submessage_id', int), ('sender_id', int), ('msg_type', str), ('content', str)])
check_submessage = make_checker(submessage_event)
single_subscription_type = DictType(required_keys=subscription_fields)
subscription_add_event = event_dict_type(required_keys=[('type', Equals('subscription')), ('op', Equals('add')), ('subscriptions', ListType(single_subscription_type))])
check_subscription_add = make_checker(subscription_add_event)
subscription_peer_add_event = event_dict_type(required_keys=[('type', Equals('subscription')), ('op', Equals('peer_add')), ('user_ids', ListType(int)), ('stream_ids', ListType(int))])
check_subscription_peer_add = make_checker(subscription_peer_add_event)
subscription_peer_remove_event = event_dict_type(required_keys=[('type', Equals('subscription')), ('op', Equals('peer_remove')), ('user_ids', ListType(int)), ('stream_ids', ListType(int))])
check_subscription_peer_remove = make_checker(subscription_peer_remove_event)
remove_sub_type = DictType(required_keys=[('name', str), ('stream_id', int)])
subscription_remove_event = event_dict_type(required_keys=[('type', Equals('subscription')), ('op', Equals('remove')), ('subscriptions', ListType(remove_sub_type))])
check_subscription_remove = make_checker(subscription_remove_event)
subscription_update_event = event_dict_type(required_keys=[('type', Equals('subscription')), ('op', Equals('update')), ('property', str), ('stream_id', int), ('value', value_type)])
_check_subscription_update = make_checker(subscription_update_event)

def check_subscription_update(var_name: str, event: Dict[str, object], property: str, value: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    _check_subscription_update(var_name, event)
    assert event['property'] == property
    assert event['value'] == value
typing_person_type = DictType(required_keys=[('email', str), ('user_id', int)])
equals_direct_or_stream = EnumType(['direct', 'stream'])
typing_start_event = event_dict_type(required_keys=[('type', Equals('typing')), ('op', Equals('start')), ('message_type', equals_direct_or_stream), ('sender', typing_person_type)], optional_keys=[('recipients', ListType(typing_person_type)), ('stream_id', int), ('topic', str)])
check_typing_start = make_checker(typing_start_event)
typing_stop_event = event_dict_type(required_keys=[('type', Equals('typing')), ('op', Equals('stop')), ('message_type', equals_direct_or_stream), ('sender', typing_person_type)], optional_keys=[('recipients', ListType(typing_person_type)), ('stream_id', int), ('topic', str)])
check_typing_stop = make_checker(typing_stop_event)
update_display_settings_event = event_dict_type(required_keys=[('type', Equals('update_display_settings')), ('setting_name', str), ('setting', value_type), ('user', str)], optional_keys=[('language_name', str)])
_check_update_display_settings = make_checker(update_display_settings_event)
user_settings_update_event = event_dict_type(required_keys=[('type', Equals('user_settings')), ('op', Equals('update')), ('property', str), ('value', value_type)], optional_keys=[('language_name', str)])
_check_user_settings_update = make_checker(user_settings_update_event)

def check_update_display_settings(var_name: str, event: Dict[str, object]) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Display setting events have a "setting" field that\n    is more specifically typed according to the\n    UserProfile.property_types dictionary.\n    '
    _check_update_display_settings(var_name, event)
    setting_name = event['setting_name']
    setting = event['setting']
    assert isinstance(setting_name, str)
    if setting_name == 'timezone':
        assert isinstance(setting, str)
    else:
        setting_type = UserProfile.property_types[setting_name]
        assert isinstance(setting, setting_type)
    if setting_name == 'default_language':
        assert 'language_name' in event
    else:
        assert 'language_name' not in event

def check_user_settings_update(var_name: str, event: Dict[str, object]) -> None:
    if False:
        while True:
            i = 10
    _check_user_settings_update(var_name, event)
    setting_name = event['property']
    value = event['value']
    assert isinstance(setting_name, str)
    if setting_name == 'timezone':
        assert isinstance(value, str)
    else:
        setting_type = UserProfile.property_types[setting_name]
        assert isinstance(value, setting_type)
    if setting_name == 'default_language':
        assert 'language_name' in event
    else:
        assert 'language_name' not in event
update_global_notifications_event = event_dict_type(required_keys=[('type', Equals('update_global_notifications')), ('notification_name', str), ('setting', value_type), ('user', str)])
_check_update_global_notifications = make_checker(update_global_notifications_event)

def check_update_global_notifications(var_name: str, event: Dict[str, object], desired_val: Union[bool, int, str]) -> None:
    if False:
        while True:
            i = 10
    '\n    See UserProfile.notification_settings_legacy for\n    more details.\n    '
    _check_update_global_notifications(var_name, event)
    setting_name = event['notification_name']
    setting = event['setting']
    assert setting == desired_val
    assert isinstance(setting_name, str)
    setting_type = UserProfile.notification_settings_legacy[setting_name]
    assert isinstance(setting, setting_type)
update_message_required_fields = [('type', Equals('update_message')), ('user_id', OptionalType(int)), ('edit_timestamp', int), ('message_id', int), ('flags', ListType(str)), ('message_ids', ListType(int)), ('rendering_only', bool)]
update_message_stream_fields: List[Tuple[str, object]] = [('stream_id', int), ('stream_name', str)]
update_message_content_fields: List[Tuple[str, object]] = [('is_me_message', bool), ('orig_content', str), ('orig_rendered_content', str), ('prev_rendered_content_version', int)]
update_message_content_or_embedded_data_fields: List[Tuple[str, object]] = [('content', str), ('rendered_content', str)]
update_message_topic_fields = [(TOPIC_LINKS, ListType(_check_topic_links)), (TOPIC_NAME, str)]
update_message_change_stream_fields: List[Tuple[str, object]] = [('new_stream_id', int)]
update_message_change_stream_or_topic_fields: List[Tuple[str, object]] = [('propagate_mode', EnumType(['change_one', 'change_later', 'change_all'])), (ORIG_TOPIC, str)]
update_message_optional_fields = update_message_stream_fields + update_message_content_fields + update_message_content_or_embedded_data_fields + update_message_topic_fields + update_message_change_stream_fields + update_message_change_stream_or_topic_fields
update_message_event = event_dict_type(required_keys=update_message_required_fields, optional_keys=update_message_optional_fields)
_check_update_message = make_checker(update_message_event)

def check_update_message(var_name: str, event: Dict[str, object], is_stream_message: bool, has_content: bool, has_topic: bool, has_new_stream_id: bool, is_embedded_update_only: bool) -> None:
    if False:
        print('Hello World!')
    _check_update_message(var_name, event)
    actual_keys = set(event.keys())
    expected_keys = {'id'}
    expected_keys.update((tup[0] for tup in update_message_required_fields))
    if is_stream_message:
        expected_keys.update((tup[0] for tup in update_message_stream_fields))
    if has_content:
        expected_keys.update((tup[0] for tup in update_message_content_fields))
        expected_keys.update((tup[0] for tup in update_message_content_or_embedded_data_fields))
    if has_topic:
        expected_keys.update((tup[0] for tup in update_message_topic_fields))
        expected_keys.update((tup[0] for tup in update_message_change_stream_or_topic_fields))
    if has_new_stream_id:
        expected_keys.update((tup[0] for tup in update_message_change_stream_fields))
        expected_keys.update((tup[0] for tup in update_message_change_stream_or_topic_fields))
    if is_embedded_update_only:
        expected_keys.update((tup[0] for tup in update_message_content_or_embedded_data_fields))
        assert event['user_id'] is None
    else:
        assert isinstance(event['user_id'], int)
    assert event['rendering_only'] == is_embedded_update_only
    assert expected_keys == actual_keys
update_message_flags_add_event = event_dict_type(required_keys=[('type', Equals('update_message_flags')), ('op', Equals('add')), ('operation', Equals('add')), ('flag', str), ('messages', ListType(int)), ('all', bool)])
check_update_message_flags_add = make_checker(update_message_flags_add_event)
update_message_flags_remove_event = event_dict_type(required_keys=[('type', Equals('update_message_flags')), ('op', Equals('remove')), ('operation', Equals('remove')), ('flag', str), ('messages', ListType(int)), ('all', bool)], optional_keys=[('message_details', StringDictType(DictType(required_keys=[('type', EnumType(['private', 'stream']))], optional_keys=[('mentioned', bool), ('user_ids', ListType(int)), ('stream_id', int), ('topic', str), ('unmuted_stream_msg', bool)])))])
check_update_message_flags_remove = make_checker(update_message_flags_remove_event)
group_type = DictType(required_keys=[('id', int), ('name', str), ('members', ListType(int)), ('direct_subgroup_ids', ListType(int)), ('description', str), ('is_system_group', bool), ('can_mention_group', int)])
user_group_add_event = event_dict_type(required_keys=[('type', Equals('user_group')), ('op', Equals('add')), ('group', group_type)])
check_user_group_add = make_checker(user_group_add_event)
user_group_add_members_event = event_dict_type(required_keys=[('type', Equals('user_group')), ('op', Equals('add_members')), ('group_id', int), ('user_ids', ListType(int))])
check_user_group_add_members = make_checker(user_group_add_members_event)
user_group_remove_event = event_dict_type(required_keys=[('type', Equals('user_group')), ('op', Equals('remove')), ('group_id', int)])
check_user_group_remove = make_checker(user_group_remove_event)
user_group_remove_members_event = event_dict_type(required_keys=[('type', Equals('user_group')), ('op', Equals('remove_members')), ('group_id', int), ('user_ids', ListType(int))])
check_user_group_remove_members = make_checker(user_group_remove_members_event)
user_group_data_type = DictType(required_keys=[], optional_keys=[('name', str), ('description', str), ('can_mention_group', int)])
user_group_update_event = event_dict_type(required_keys=[('type', Equals('user_group')), ('op', Equals('update')), ('group_id', int), ('data', user_group_data_type)])
_check_user_group_update = make_checker(user_group_update_event)

def check_user_group_update(var_name: str, event: Dict[str, object], field: str) -> None:
    if False:
        return 10
    _check_user_group_update(var_name, event)
    assert isinstance(event['data'], dict)
    assert set(event['data'].keys()) == {field}
user_group_add_subgroups_event = event_dict_type(required_keys=[('type', Equals('user_group')), ('op', Equals('add_subgroups')), ('group_id', int), ('direct_subgroup_ids', ListType(int))])
check_user_group_add_subgroups = make_checker(user_group_add_subgroups_event)
user_group_remove_subgroups_event = event_dict_type(required_keys=[('type', Equals('user_group')), ('op', Equals('remove_subgroups')), ('group_id', int), ('direct_subgroup_ids', ListType(int))])
check_user_group_remove_subgroups = make_checker(user_group_remove_subgroups_event)
user_status_event = event_dict_type(required_keys=[('type', Equals('user_status')), ('user_id', int)], optional_keys=[('away', bool), ('status_text', str), ('emoji_name', str), ('emoji_code', str), ('reaction_type', EnumType(['unicode_emoji', 'realm_emoji', 'zulip_extra_emoji']))])
_check_user_status = make_checker(user_status_event)

def check_user_status(var_name: str, event: Dict[str, object], fields: Set[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    _check_user_status(var_name, event)
    assert set(event.keys()) == {'id', 'type', 'user_id'} | fields