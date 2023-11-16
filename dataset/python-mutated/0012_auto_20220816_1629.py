from django.db import migrations, models

def migrate_system_user_to_account(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    connection_token_model = apps.get_model('authentication', 'ConnectionToken')
    count = 0
    bulk_size = 10000
    while True:
        connection_tokens = connection_token_model.objects.prefetch_related('system_user')[count:bulk_size]
        if not connection_tokens:
            break
        count += len(connection_tokens)
        updated = []
        for connection_token in connection_tokens:
            if not connection_token.system_user:
                continue
            connection_token.account_username = connection_token.system_user.username
            updated.append(connection_token)
        connection_token_model.objects.bulk_update(updated, ['account_username'])

class Migration(migrations.Migration):
    dependencies = [('authentication', '0011_auto_20220705_1940')]
    operations = [migrations.RemoveField(model_name='connectiontoken', name='application'), migrations.RemoveField(model_name='connectiontoken', name='application_display'), migrations.RemoveField(model_name='connectiontoken', name='system_user_display'), migrations.AddField(model_name='connectiontoken', name='account_username', field=models.CharField(default='', max_length=128, verbose_name='Account')), migrations.RunPython(migrate_system_user_to_account), migrations.RemoveField(model_name='connectiontoken', name='system_user'), migrations.RemoveField(model_name='connectiontoken', name='type')]