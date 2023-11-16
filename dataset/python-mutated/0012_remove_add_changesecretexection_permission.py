from django.db import migrations

def migrate_remove_add_changesecretexection_permission(apps, *args):
    if False:
        while True:
            i = 10
    perm_model = apps.get_model('auth', 'Permission')
    perm_model.objects.filter(codename='add_changesecretexection').delete()

class Migration(migrations.Migration):
    dependencies = [('rbac', '0011_remove_redundant_permission')]
    operations = [migrations.RunPython(migrate_remove_add_changesecretexection_permission)]