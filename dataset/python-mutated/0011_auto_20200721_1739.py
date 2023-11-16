from django.db import migrations, models
from django.db.models import F

def migrate_asset_permission(apps, schema_editor):
    if False:
        return 10
    asset_permission_model = apps.get_model('perms', 'AssetPermission')
    asset_permission_model.objects.all().update(actions=F('actions').bitor(24))

class Migration(migrations.Migration):
    dependencies = [('perms', '0010_auto_20191218_1705')]
    operations = [migrations.AlterField(model_name='assetpermission', name='actions', field=models.IntegerField(choices=[(255, 'All'), (1, 'Connect'), (2, 'Upload file'), (4, 'Download file'), (6, 'Upload download'), (8, 'Clipboard copy'), (16, 'Clipboard paste'), (24, 'Clipboard copy paste')], default=255, verbose_name='Actions')), migrations.RunPython(migrate_asset_permission)]