from django.db import migrations

def create_admin_access_permissions(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    ContentType = apps.get_model('contenttypes.ContentType')
    Permission = apps.get_model('auth.Permission')
    Group = apps.get_model('auth.Group')
    (wagtailadmin_content_type, created) = ContentType.objects.get_or_create(app_label='wagtailadmin', model='admin')
    (admin_permission, created) = Permission.objects.get_or_create(content_type=wagtailadmin_content_type, codename='access_admin', name='Can access Wagtail admin')
    for group in Group.objects.filter(name__in=['Editors', 'Moderators']):
        group.permissions.add(admin_permission)

def remove_admin_access_permissions(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    'Reverse the above additions of permissions.'
    ContentType = apps.get_model('contenttypes.ContentType')
    Permission = apps.get_model('auth.Permission')
    wagtailadmin_content_type = ContentType.objects.get(app_label='wagtailadmin', model='admin')
    Permission.objects.filter(content_type=wagtailadmin_content_type, codename='access_admin').delete()

class Migration(migrations.Migration):
    dependencies = [('wagtailcore', '0026_group_collection_permission')]
    operations = [migrations.RunPython(create_admin_access_permissions, remove_admin_access_permissions)]