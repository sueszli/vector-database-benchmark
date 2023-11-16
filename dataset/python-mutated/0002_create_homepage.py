from django.db import migrations

def create_homepage(apps, schema_editor):
    if False:
        while True:
            i = 10
    ContentType = apps.get_model('contenttypes.ContentType')
    Page = apps.get_model('wagtailcore.Page')
    Site = apps.get_model('wagtailcore.Site')
    HomePage = apps.get_model('home.HomePage')
    Page.objects.filter(id=2).delete()
    (homepage_content_type, __) = ContentType.objects.get_or_create(model='homepage', app_label='home')
    homepage = HomePage.objects.create(title='Home', draft_title='Home', slug='home', content_type=homepage_content_type, path='00010001', depth=2, numchild=0, url_path='/home/')
    Site.objects.create(hostname='localhost', root_page=homepage, is_default_site=True)

def remove_homepage(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    ContentType = apps.get_model('contenttypes.ContentType')
    HomePage = apps.get_model('home.HomePage')
    HomePage.objects.filter(slug='home', depth=2).delete()
    ContentType.objects.filter(model='homepage', app_label='home').delete()

class Migration(migrations.Migration):
    run_before = [('wagtailcore', '0053_locale_model')]
    dependencies = [('home', '0001_initial')]
    operations = [migrations.RunPython(create_homepage, remove_homepage)]