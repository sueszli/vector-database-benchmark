from django.db import migrations
from util.migrations import merge_duplicate_permissions

def merge_duplicate_project_permissions(apps, schema_editor):
    if False:
        while True:
            i = 10
    UserProjectPermission = apps.get_model('projects', 'UserProjectPermission')
    UserPermissionGroupProjectPermission = apps.get_model('projects', 'UserPermissionGroupProjectPermission')
    merge_duplicate_permissions(UserProjectPermission, ['user', 'project'])
    merge_duplicate_permissions(UserPermissionGroupProjectPermission, ['group', 'project'])

class Migration(migrations.Migration):
    dependencies = [('projects', '0016_soft_delete_projects')]
    operations = [migrations.RunPython(merge_duplicate_project_permissions, reverse_code=migrations.RunPython.noop)]