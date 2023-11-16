from typing import Any, Set, Union
import orjson
from django.conf import settings
from django.contrib.auth.hashers import check_password, make_password
from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from django.utils.timezone import now as timezone_now
from zerver.lib.cache import cache_delete, user_profile_by_api_key_cache_key
from zerver.lib.queue import queue_json_publish
from zerver.lib.utils import generate_api_key

def ensure_no_empty_passwords(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        i = 10
        return i + 15
    'With CVE-2019-18933, it was possible for certain users created\n    using social login (e.g. Google/GitHub auth) to have the empty\n    string as their password in the Zulip database, rather than\n    Django\'s "unusable password" (i.e. no password at all).  This was a\n    serious security issue for organizations with both password and\n    Google/GitHub authentication enabled.\n\n    Combined with the code changes to prevent new users from entering\n    this buggy state, this migration sets the intended "no password"\n    state for any users who are in this buggy state, as had been\n    intended.\n\n    While this bug was discovered by our own development team and we\n    believe it hasn\'t been exploited in the wild, out of an abundance\n    of caution, this migration also resets the personal API keys for\n    all users where Zulip\'s database-level logging cannot **prove**\n    that user\'s current personal API key was never accessed using this\n    bug.\n\n    There are a few ways this can be proven: (1) the user\'s password\n    has never been changed and is not the empty string,\n    or (2) the user\'s personal API key has changed since that user last\n    changed their password (which is not \'\'). Both constitute proof\n    because this bug cannot be used to gain the access required to change\n    or reset a user\'s password.\n\n    Resetting those API keys has the effect of logging many users out\n    of the Zulip mobile and terminal apps unnecessarily (e.g. because\n    the user changed their password at any point in the past, even\n    though the user never was affected by the bug), but we\'re\n    comfortable with that cost for ensuring that this bug is\n    completely fixed.\n\n    To avoid this inconvenience for self-hosted servers which don\'t\n    even have EmailAuthBackend enabled, we skip resetting any API keys\n    if the server doesn\'t have EmailAuthBackend configured.\n    '
    UserProfile = apps.get_model('zerver', 'UserProfile')
    RealmAuditLog = apps.get_model('zerver', 'RealmAuditLog')
    event_type_class = RealmAuditLog._meta.get_field('event_type').get_internal_type()
    if event_type_class == 'CharField':
        USER_PASSWORD_CHANGED: Union[int, str] = 'user_password_changed'
        USER_API_KEY_CHANGED: Union[int, str] = 'user_api_key_changed'
    else:
        USER_PASSWORD_CHANGED = 122
        USER_API_KEY_CHANGED = 127
    password_change_user_ids = set(RealmAuditLog.objects.filter(event_type=USER_PASSWORD_CHANGED).values_list('modified_user_id', flat=True))
    password_change_user_ids_api_key_reset_needed: Set[int] = set()
    password_change_user_ids_no_reset_needed: Set[int] = set()
    for user_id in password_change_user_ids:
        query = RealmAuditLog.objects.filter(modified_user=user_id, event_type__in=[USER_PASSWORD_CHANGED, USER_API_KEY_CHANGED]).order_by('event_time')
        earliest_password_change = query.filter(event_type=USER_PASSWORD_CHANGED).first()
        assert earliest_password_change is not None
        latest_api_key_change = query.filter(event_type=USER_API_KEY_CHANGED).last()
        if latest_api_key_change is None:
            password_change_user_ids_api_key_reset_needed.add(user_id)
        elif earliest_password_change.event_time <= latest_api_key_change.event_time:
            password_change_user_ids_no_reset_needed.add(user_id)
        else:
            password_change_user_ids_api_key_reset_needed.add(user_id)
    if password_change_user_ids_no_reset_needed and settings.PRODUCTION:
        with open('/var/log/zulip/0209_password_migration.log', 'w') as log_file:
            line = 'No reset needed, but changed password: {}\n'
            log_file.write(line.format(password_change_user_ids_no_reset_needed))
    AFFECTED_USER_TYPE_EMPTY_PASSWORD = 'empty_password'
    AFFECTED_USER_TYPE_CHANGED_PASSWORD = 'changed_password'
    MIGRATION_ID = '0209_user_profile_no_empty_password'

    def write_realm_audit_log_entry(user_profile: Any, event_time: Any, event_type: Any, affected_user_type: str) -> None:
        if False:
            i = 10
            return i + 15
        RealmAuditLog.objects.create(realm=user_profile.realm, modified_user=user_profile, event_type=event_type, event_time=event_time, extra_data=orjson.dumps({'migration_id': MIGRATION_ID, 'affected_user_type': affected_user_type}).decode())
    email_auth_enabled = 'zproject.backends.EmailAuthBackend' in settings.AUTHENTICATION_BACKENDS
    for user_profile in UserProfile.objects.all():
        event_time = timezone_now()
        if check_password('', user_profile.password):
            user_profile.password = make_password(None)
            update_fields = ['password']
            write_realm_audit_log_entry(user_profile, event_time, USER_PASSWORD_CHANGED, AFFECTED_USER_TYPE_EMPTY_PASSWORD)
            if email_auth_enabled and (not user_profile.is_bot):
                reset_user_api_key(user_profile)
                update_fields.append('api_key')
                event_time = timezone_now()
                write_realm_audit_log_entry(user_profile, event_time, USER_API_KEY_CHANGED, AFFECTED_USER_TYPE_EMPTY_PASSWORD)
            user_profile.save(update_fields=update_fields)
            continue
        elif email_auth_enabled and user_profile.id in password_change_user_ids_api_key_reset_needed:
            reset_user_api_key(user_profile)
            user_profile.save(update_fields=['api_key'])
            write_realm_audit_log_entry(user_profile, event_time, USER_API_KEY_CHANGED, AFFECTED_USER_TYPE_CHANGED_PASSWORD)

def reset_user_api_key(user_profile: Any) -> None:
    if False:
        while True:
            i = 10
    old_api_key = user_profile.api_key
    user_profile.api_key = generate_api_key()
    cache_delete(user_profile_by_api_key_cache_key(old_api_key))
    event = {'type': 'clear_push_device_tokens', 'user_profile_id': user_profile.id}
    queue_json_publish('deferred_work', event)

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0208_add_realm_night_logo_fields')]
    operations = [migrations.RunPython(ensure_no_empty_passwords, reverse_code=migrations.RunPython.noop, elidable=True)]