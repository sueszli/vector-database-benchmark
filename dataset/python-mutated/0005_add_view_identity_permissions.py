from django.db import migrations
from environments.permissions.constants import VIEW_IDENTITIES, MANAGE_IDENTITIES
from core.migration_helpers import create_new_environment_permissions
from permissions.models import ENVIRONMENT_PERMISSION_TYPE

def add_view_identities_permission(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    PermissionModel = apps.get_model('permissions', 'PermissionModel')
    UserEnvironmentPermission = apps.get_model('environment_permissions', 'UserEnvironmentPermission')
    UserPermissionGroupEnvironmentPermission = apps.get_model('environment_permissions', 'UserPermissionGroupEnvironmentPermission')
    (view_identties_permission, _) = PermissionModel.objects.get_or_create(key=VIEW_IDENTITIES, description='View identities in the given environment.', type=ENVIRONMENT_PERMISSION_TYPE)
    create_new_environment_permissions(MANAGE_IDENTITIES, UserEnvironmentPermission, 'userenvironmentpermission', [view_identties_permission])
    create_new_environment_permissions(MANAGE_IDENTITIES, UserPermissionGroupEnvironmentPermission, 'userpermissiongroupenvironmentpermission', [view_identties_permission])

def remove_view_identities_permission(apps, schema_editor):
    if False:
        return 10
    PermissionModel = apps.get_model('permissions', 'PermissionModel')
    PermissionModel.objects.filter(key=VIEW_IDENTITIES).delete()

class Migration(migrations.Migration):
    dependencies = [('environment_permissions', '0004_add_change_request_permissions')]
    operations = [migrations.RunPython(add_view_identities_permission, reverse_code=remove_view_identities_permission)]