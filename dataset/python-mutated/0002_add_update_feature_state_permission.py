from django.db import migrations
from environments.permissions.constants import UPDATE_FEATURE_STATE
from permissions.models import ENVIRONMENT_PERMISSION_TYPE

def add_update_feature_state_permission(apps, schema_editor):
    if False:
        return 10
    PermissionModel = apps.get_model('permissions', 'PermissionModel')
    PermissionModel.objects.get_or_create(key=UPDATE_FEATURE_STATE, description='Update the state or value for a given feature state.', type=ENVIRONMENT_PERMISSION_TYPE)

def remove_update_feature_state_permission(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    PermissionModel = apps.get_model('permissions', 'PermissionModel')
    PermissionModel.objects.filter(key=UPDATE_FEATURE_STATE).delete()

class Migration(migrations.Migration):
    dependencies = [('environment_permissions', '0001_initial'), ('features', '0035_auto_20211109_0603')]
    operations = [migrations.RunPython(add_update_feature_state_permission, reverse_code=remove_update_feature_state_permission)]