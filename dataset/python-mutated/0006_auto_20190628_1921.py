from django.db import migrations, models
from functools import reduce

def migrate_old_actions(apps, schema_editor):
    if False:
        while True:
            i = 10
    from orgs.utils import set_to_root_org
    set_to_root_org()
    perm_model = apps.get_model('perms', 'AssetPermission')
    db_alias = schema_editor.connection.alias
    perms = perm_model.objects.using(db_alias).all()
    actions_map = {'all': 255, 'connect': 1, 'upload_file': 2, 'download_file': 4}
    for perm in perms:
        actions = perm.actions.all()
        if not actions:
            continue
        new_actions = [actions_map.get(action.name, 255) for action in actions]
        new_action = reduce(lambda x, y: x | y, new_actions)
        perm.action = new_action
        perm.save()

class Migration(migrations.Migration):
    dependencies = [('perms', '0005_auto_20190521_1619')]
    operations = [migrations.AddField(model_name='assetpermission', name='action', field=models.IntegerField(choices=[(255, 'All'), (1, 'Connect'), (2, 'Upload file'), (4, 'Download file'), (6, 'Upload download')], default=255, verbose_name='Actions')), migrations.RunPython(migrate_old_actions)]