from django.db import migrations

def add_choose_permission_to_admin_groups(apps, _schema_editor):
    if False:
        for i in range(10):
            print('nop')
    ContentType = apps.get_model('contenttypes.ContentType')
    Permission = apps.get_model('auth.Permission')
    Group = apps.get_model('auth.Group')
    (image_content_type, _created) = ContentType.objects.get_or_create(model='image', app_label='wagtailimages')
    (choose_image_permission, _created) = Permission.objects.get_or_create(content_type=image_content_type, codename='choose_image', defaults={'name': 'Can choose image'})
    for group in Group.objects.filter(permissions__codename='access_admin'):
        group.permissions.add(choose_image_permission)

def remove_choose_permission(apps, _schema_editor):
    if False:
        i = 10
        return i + 15
    'Reverse the above additions of permissions.'
    ContentType = apps.get_model('contenttypes.ContentType')
    Permission = apps.get_model('auth.Permission')
    image_content_type = ContentType.objects.get(model='image', app_label='wagtailimages')
    Permission.objects.filter(content_type=image_content_type, codename='choose_image').delete()

def get_choose_permission(apps):
    if False:
        print('Hello World!')
    Permission = apps.get_model('auth.Permission')
    ContentType = apps.get_model('contenttypes.ContentType')
    (image_content_type, _created) = ContentType.objects.get_or_create(model='image', app_label='wagtailimages')
    return Permission.objects.filter(content_type=image_content_type, codename__in=['choose_image']).first()

def copy_choose_permission_to_collections(apps, _schema_editor):
    if False:
        for i in range(10):
            print('nop')
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
    dependencies = [('wagtailimages', '0022_uploadedimage')]
    operations = [migrations.AlterModelOptions(name='image', options={'permissions': [('choose_image', 'Can choose image')], 'verbose_name': 'image', 'verbose_name_plural': 'images'}), migrations.RunPython(add_choose_permission_to_admin_groups, remove_choose_permission), migrations.RunPython(copy_choose_permission_to_collections, remove_choose_permission_from_collections)]