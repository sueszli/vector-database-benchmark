from django.db import migrations

def migrate_workspace_to_workbench(apps, *args):
    if False:
        i = 10
        return i + 15
    model = apps.get_model('auth', 'Permission')
    model.objects.filter(codename='view_workspace').delete()

class Migration(migrations.Migration):
    dependencies = [('rbac', '0008_auto_20220411_1709')]
    operations = [migrations.RunPython(migrate_workspace_to_workbench)]