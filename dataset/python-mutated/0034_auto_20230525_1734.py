from django.db import migrations

def migrate_asset_permission_delete_perm(apps, *args):
    if False:
        return 10
    asset_permission_cls = apps.get_model('perms', 'AssetPermission')
    asset_permission_cls.objects.filter(actions__gte=31).update(actions=63)

class Migration(migrations.Migration):
    dependencies = [('perms', '0033_auto_20221220_1956')]
    operations = [migrations.RunPython(migrate_asset_permission_delete_perm)]