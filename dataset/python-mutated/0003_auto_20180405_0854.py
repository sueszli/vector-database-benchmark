from django.db import migrations, models

def migrate_slug_to_name(apps, schema_editor):
    if False:
        while True:
            i = 10
    Menu = apps.get_model('menu', 'Menu')
    for menu in Menu.objects.all():
        menu.name = menu.slug
        menu.save()

class Migration(migrations.Migration):
    dependencies = [('menu', '0002_auto_20180319_0412')]
    operations = [migrations.AddField(model_name='menu', name='name', field=models.CharField(default='', max_length=128), preserve_default=False), migrations.RunPython(migrate_slug_to_name, migrations.RunPython.noop), migrations.RemoveField(model_name='menu', name='slug')]