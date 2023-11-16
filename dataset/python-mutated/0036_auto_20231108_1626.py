from django.db import migrations

def migrate_action_all_value(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    old_max_bit = 63
    new_max_bit = 4095
    asset_permission_model = apps.get_model('perms', 'AssetPermission')
    db_alias = schema_editor.connection.alias
    perms = list(asset_permission_model.objects.using(db_alias).filter(actions=old_max_bit))
    for perm in perms:
        perm.actions = new_max_bit
    asset_permission_model.objects.bulk_update(perms, ['actions'])

class Migration(migrations.Migration):
    dependencies = [('perms', '0035_auto_20231125_1025')]
    operations = [migrations.RunPython(code=migrate_action_all_value)]