import time
from django.db import migrations, models

def migrate_system_to_account(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    apply_asset_ticket_model = apps.get_model('tickets', 'ApplyAssetTicket')
    apply_command_ticket_model = apps.get_model('tickets', 'ApplyCommandTicket')
    apply_login_asset_ticket_model = apps.get_model('tickets', 'ApplyLoginAssetTicket')
    model_system_user_account = ((apply_asset_ticket_model, 'apply_system_users', 'apply_accounts', True), (apply_command_ticket_model, 'apply_run_system_user', 'apply_run_account', False), (apply_login_asset_ticket_model, 'apply_login_system_user', 'apply_login_account', False))
    print('\n  Start migrate system user to account')
    for (model, old_field, new_field, m2m) in model_system_user_account:
        print("\t  - migrate '{}'".format(model.__name__))
        count = 0
        bulk_size = 1000
        while True:
            start = time.time()
            objects = model.objects.all().prefetch_related(old_field)[count:bulk_size]
            if not objects:
                break
            count += len(objects)
            updated = []
            for obj in objects:
                if m2m:
                    old_value = getattr(obj, old_field).all()
                    new_value = [s.username for s in old_value]
                else:
                    old_value = getattr(obj, old_field)
                    new_value = old_value.username if old_value else ''
                setattr(obj, new_field, new_value)
                updated.append(obj)
            model.objects.bulk_update(updated, [new_field])
            print('    Migrate account: {}-{} using: {:.2f}s'.format(count - len(objects), count, time.time() - start))

class Migration(migrations.Migration):
    dependencies = [('tickets', '0019_delete_applyapplicationticket')]
    operations = [migrations.AlterField(model_name='applyassetticket', name='apply_permission_name', field=models.CharField(max_length=128, verbose_name='Permission name')), migrations.AddField(model_name='applyassetticket', name='apply_accounts', field=models.JSONField(default=list, verbose_name='Apply accounts')), migrations.AddField(model_name='applycommandticket', name='apply_run_account', field=models.CharField(default='', max_length=128, verbose_name='Run account')), migrations.AddField(model_name='applyloginassetticket', name='apply_login_account', field=models.CharField(default='', max_length=128, verbose_name='Login account')), migrations.RunPython(migrate_system_to_account), migrations.RemoveField(model_name='applyassetticket', name='apply_system_users'), migrations.RemoveField(model_name='applycommandticket', name='apply_run_system_user'), migrations.RemoveField(model_name='applyloginassetticket', name='apply_login_system_user')]