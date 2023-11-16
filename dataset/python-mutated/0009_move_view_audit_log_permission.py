from django.apps.registry import Apps
from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from projects.permissions import VIEW_AUDIT_LOG
from permissions.models import ORGANISATION_PERMISSION_TYPE, PROJECT_PERMISSION_TYPE

def move_permission_to_project(apps: Apps, schema_editor: BaseDatabaseSchemaEditor):
    if False:
        print('Hello World!')
    permission_model_class = apps.get_model('permissions', 'PermissionModel')
    permission_model_class.objects.filter(key=VIEW_AUDIT_LOG).update(type=PROJECT_PERMISSION_TYPE, description='Allows the user to view the audit logs for this project.')

def move_permission_to_organisation(apps: Apps, schema_editor: BaseDatabaseSchemaEditor):
    if False:
        for i in range(10):
            print('nop')
    permission_model_class = apps.get_model('permissions', 'PermissionModel')
    permission_model_class.objects.filter(key=VIEW_AUDIT_LOG).update(type=ORGANISATION_PERMISSION_TYPE, description='Allows the user to view the audit logs for this organisation.')

class Migration(migrations.Migration):
    dependencies = [('permissions', '0008_add_view_audit_log_permission')]
    operations = [migrations.RunPython(move_permission_to_project, reverse_code=move_permission_to_organisation)]