from django.db import migrations
from environments.permissions.constants import MANAGE_IDENTITIES
from permissions.models import ENVIRONMENT_PERMISSION_TYPE

def add_manage_identities_permission(apps, schema_editor):
    if False:
        while True:
            i = 10
    PermissionModel = apps.get_model('permissions', 'PermissionModel')
    PermissionModel.objects.get_or_create(key=MANAGE_IDENTITIES, description='Manage identities in the given environment.', type=ENVIRONMENT_PERMISSION_TYPE)

def remove_manage_identities_permission(apps, schema_editor):
    if False:
        return 10
    PermissionModel = apps.get_model('permissions', 'PermissionModel')
    PermissionModel.objects.filter(key=MANAGE_IDENTITIES).delete()

class Migration(migrations.Migration):
    dependencies = [('environment_permissions', '0002_add_update_feature_state_permission')]
    operations = [migrations.RunPython(add_manage_identities_permission, reverse_code=remove_manage_identities_permission)]