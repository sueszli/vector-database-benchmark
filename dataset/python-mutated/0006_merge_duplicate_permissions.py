from django.db import migrations
from util.migrations import merge_duplicate_permissions

def merge_duplicate_environment_permissions(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    UserEnvironmentPermission = apps.get_model('environment_permissions', 'UserEnvironmentPermission')
    UserPermissionGroupEnvironmentPermission = apps.get_model('environment_permissions', 'UserPermissionGroupEnvironmentPermission')
    merge_duplicate_permissions(UserEnvironmentPermission, ['user', 'environment'])
    merge_duplicate_permissions(UserPermissionGroupEnvironmentPermission, ['group', 'environment'])

class Migration(migrations.Migration):
    dependencies = [('environment_permissions', '0005_add_view_identity_permissions')]
    operations = [migrations.RunPython(merge_duplicate_environment_permissions, reverse_code=migrations.RunPython.noop)]