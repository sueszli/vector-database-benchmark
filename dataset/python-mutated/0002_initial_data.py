from django.db import migrations

def initial_data(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    ContentType = apps.get_model('contenttypes.ContentType')
    Group = apps.get_model('auth.Group')
    Page = apps.get_model('wagtailcore.Page')
    Site = apps.get_model('wagtailcore.Site')
    GroupPagePermission = apps.get_model('wagtailcore.GroupPagePermission')
    (page_content_type, created) = ContentType.objects.get_or_create(model='page', app_label='wagtailcore')
    root = Page.objects.create(title='Root', slug='root', content_type=page_content_type, path='0001', depth=1, numchild=1, url_path='/')
    homepage = Page.objects.create(title='Welcome to your new Wagtail site!', slug='home', content_type=page_content_type, path='00010001', depth=2, numchild=0, url_path='/home/')
    Site.objects.create(hostname='localhost', root_page_id=homepage.id, is_default_site=True)
    moderators_group = Group.objects.create(name='Moderators')
    editors_group = Group.objects.create(name='Editors')
    GroupPagePermission.objects.create(group=moderators_group, page=root, permission_type='add')
    GroupPagePermission.objects.create(group=moderators_group, page=root, permission_type='edit')
    GroupPagePermission.objects.create(group=moderators_group, page=root, permission_type='publish')
    GroupPagePermission.objects.create(group=editors_group, page=root, permission_type='add')
    GroupPagePermission.objects.create(group=editors_group, page=root, permission_type='edit')

def remove_initial_data(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    "This function does nothing. The below code is commented out together\n    with an explanation of why we don't need to bother reversing any of the\n    initial data"
    pass

class Migration(migrations.Migration):
    dependencies = [('wagtailcore', '0001_initial')]
    operations = [migrations.RunPython(initial_data, remove_initial_data)]