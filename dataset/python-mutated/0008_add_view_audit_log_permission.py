from django.apps.registry import Apps
from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from projects.permissions import VIEW_AUDIT_LOG
from permissions.models import ORGANISATION_PERMISSION_TYPE

def create_permissions(apps: Apps, schema_editor: BaseDatabaseSchemaEditor):
    if False:
        while True:
            i = 10
    permission_model_class = apps.get_model('permissions', 'PermissionModel')
    permission_model_class.objects.get_or_create(key=VIEW_AUDIT_LOG, defaults={'description': 'Allows the user to view the audit logs for this organisation.', 'type': ORGANISATION_PERMISSION_TYPE})

def remove_permissions(apps: Apps, schema_editor: BaseDatabaseSchemaEditor):
    if False:
        while True:
            i = 10
    apps.get_model('permissions', 'PermissionModel').objects.filter(key=VIEW_AUDIT_LOG).delete()

class Migration(migrations.Migration):
    dependencies = [('permissions', '0007_add_invite_users_and_manage_user_groups_org_permissions')]
    operations = [migrations.RunPython(create_permissions, reverse_code=remove_permissions)]