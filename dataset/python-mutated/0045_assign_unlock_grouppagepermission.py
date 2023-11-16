from django.db import migrations

def assign_unlock_grouppagepermission(apps, schema_editor):
    if False:
        return 10
    GroupPagePermission = apps.get_model('wagtailcore.GroupPagePermission')
    for lock_permission in GroupPagePermission.objects.filter(permission_type='lock'):
        GroupPagePermission.objects.create(group=lock_permission.group, page=lock_permission.page, permission_type='unlock')

class Migration(migrations.Migration):
    dependencies = [('wagtailcore', '0044_add_unlock_grouppagepermission')]
    operations = [migrations.RunPython(assign_unlock_grouppagepermission, migrations.RunPython.noop)]