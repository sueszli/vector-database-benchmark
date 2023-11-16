from django.db import migrations

def migrate_workspace_to_workbench(apps, *args):
    if False:
        for i in range(10):
            print('nop')
    model = apps.get_model('auth', 'Permission')
    model.objects.filter(codename='view_workspace').delete()

class Migration(migrations.Migration):
    dependencies = [('rbac', '0007_auto_20220314_1525')]
    operations = [migrations.AlterModelOptions(name='menupermission', options={'default_permissions': [], 'permissions': [('view_console', 'Can view console view'), ('view_audit', 'Can view audit view'), ('view_workbench', 'Can view workbench view'), ('view_webterminal', 'Can view web terminal'), ('view_filemanager', 'Can view file manager')], 'verbose_name': 'Menu permission'})]