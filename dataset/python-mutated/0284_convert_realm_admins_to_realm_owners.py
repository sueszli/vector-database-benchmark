from typing import Any, Dict
import orjson
from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from django.db.models import Count
from django.utils.timezone import now as timezone_now

def set_realm_admins_as_realm_owners(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    UserProfile = apps.get_model('zerver', 'UserProfile')
    RealmAuditLog = apps.get_model('zerver', 'RealmAuditLog')
    UserProfile.ROLE_REALM_OWNER = 100
    UserProfile.ROLE_REALM_ADMINISTRATOR = 200
    UserProfile.ROLE_MEMBER = 400
    UserProfile.ROLE_GUEST = 600
    RealmAuditLog.USER_ROLE_CHANGED = 105
    RealmAuditLog.OLD_VALUE = '1'
    RealmAuditLog.NEW_VALUE = '2'
    RealmAuditLog.ROLE_COUNT = '10'
    RealmAuditLog.ROLE_COUNT_HUMANS = '11'
    RealmAuditLog.ROLE_COUNT_BOTS = '12'

    def realm_user_count_by_role(realm: Any) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        human_counts = {str(UserProfile.ROLE_REALM_ADMINISTRATOR): 0, str(UserProfile.ROLE_REALM_OWNER): 0, str(UserProfile.ROLE_MEMBER): 0, str(UserProfile.ROLE_GUEST): 0}
        for value_dict in UserProfile.objects.filter(realm=realm, is_bot=False, is_active=True).values('role').annotate(Count('role')):
            human_counts[str(value_dict['role'])] = value_dict['role__count']
        bot_count = UserProfile.objects.filter(realm=realm, is_bot=True, is_active=True).count()
        return {RealmAuditLog.ROLE_COUNT_HUMANS: human_counts, RealmAuditLog.ROLE_COUNT_BOTS: bot_count}
    objects_to_create = []
    for user in UserProfile.objects.filter(is_active=True, role=UserProfile.ROLE_REALM_ADMINISTRATOR):
        user.role = UserProfile.ROLE_REALM_OWNER
        user.save(update_fields=['role'])
        audit_log_entry = RealmAuditLog(realm=user.realm, modified_user=user, event_type=RealmAuditLog.USER_ROLE_CHANGED, event_time=timezone_now(), extra_data=orjson.dumps({RealmAuditLog.OLD_VALUE: UserProfile.ROLE_REALM_ADMINISTRATOR, RealmAuditLog.NEW_VALUE: UserProfile.ROLE_REALM_OWNER, RealmAuditLog.ROLE_COUNT: realm_user_count_by_role(user.realm)}).decode())
        objects_to_create.append(audit_log_entry)
    RealmAuditLog.objects.bulk_create(objects_to_create)

class Migration(migrations.Migration):
    dependencies = [('zerver', '0283_apple_auth')]
    operations = [migrations.RunPython(set_realm_admins_as_realm_owners, reverse_code=migrations.RunPython.noop, elidable=True)]