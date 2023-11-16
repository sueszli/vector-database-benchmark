from django.db import migrations

def add_choose_permission_to_admin_groups(apps, _schema_editor):
    if False:
        return 10
    ContentType = apps.get_model('contenttypes.ContentType')
    Permission = apps.get_model('auth.Permission')
    Group = apps.get_model('auth.Group')
    (document_content_type, _created) = ContentType.objects.get_or_create(model='document', app_label='wagtaildocs')
    (choose_document_permission, _created) = Permission.objects.get_or_create(content_type=document_content_type, codename='choose_document', defaults={'name': 'Can choose document'})
    for group in Group.objects.filter(permissions__codename='access_admin'):
        group.permissions.add(choose_document_permission)

def remove_choose_permission(apps, _schema_editor):
    if False:
        return 10
    'Reverse the above additions of permissions.'
    ContentType = apps.get_model('contenttypes.ContentType')
    Permission = apps.get_model('auth.Permission')
    document_content_type = ContentType.objects.get(model='document', app_label='wagtaildocs')
    Permission.objects.filter(content_type=document_content_type, codename='choose_document').delete()

def get_choose_permission(apps):
    if False:
        for i in range(10):
            print('nop')
    Permission = apps.get_model('auth.Permission')
    ContentType = apps.get_model('contenttypes.ContentType')
    (document_content_type, _created) = ContentType.objects.get_or_create(model='document', app_label='wagtaildocs')
    return Permission.objects.filter(content_type=document_content_type, codename__in=['choose_document']).first()

def copy_choose_permission_to_collections(apps, _schema_editor):
    if False:
        while True:
            i = 10
    Collection = apps.get_model('wagtailcore.Collection')
    Group = apps.get_model('auth.Group')
    GroupCollectionPermission = apps.get_model('wagtailcore.GroupCollectionPermission')
    root_collection = Collection.objects.get(depth=1)
    permission = get_choose_permission(apps)
    if permission:
        for group in Group.objects.filter(permissions=permission):
            GroupCollectionPermission.objects.create(group=group, collection=root_collection, permission=permission)

def remove_choose_permission_from_collections(apps, _schema_editor):
    if False:
        return 10
    GroupCollectionPermission = apps.get_model('wagtailcore.GroupCollectionPermission')
    choose_permission = get_choose_permission(apps)
    if choose_permission:
        GroupCollectionPermission.objects.filter(permission=choose_permission).delete()

class Migration(migrations.Migration):
    dependencies = [('wagtaildocs', '0010_document_file_hash')]
    operations = [migrations.AlterModelOptions(name='document', options={'permissions': [('choose_document', 'Can choose document')], 'verbose_name': 'document', 'verbose_name_plural': 'documents'}), migrations.RunPython(add_choose_permission_to_admin_groups, remove_choose_permission), migrations.RunPython(copy_choose_permission_to_collections, remove_choose_permission_from_collections)]