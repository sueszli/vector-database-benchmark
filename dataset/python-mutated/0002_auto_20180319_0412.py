from django.conf import settings
from django.db import migrations

def create_default_menus(apps, schema_editor):
    if False:
        return 10
    Menu = apps.get_model('menu', 'Menu')
    for slug in settings.DEFAULT_MENUS.values():
        Menu.objects.get_or_create(slug=slug)

class Migration(migrations.Migration):
    dependencies = [('menu', '0001_initial')]
    operations = [migrations.RunPython(create_default_menus, migrations.RunPython.noop)]