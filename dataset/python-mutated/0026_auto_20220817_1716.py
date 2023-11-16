import time
from django.db import migrations

def migrate_account_dirty_data(apps, schema_editor):
    if False:
        while True:
            i = 10
    db_alias = schema_editor.connection.alias
    account_model = apps.get_model('applications', 'Account')
    count = 0
    bulk_size = 1000
    while True:
        accounts = account_model.objects.using(db_alias).filter(org_id='')[count:count + bulk_size]
        if not accounts:
            break
        accounts = list(accounts)
        start = time.time()
        for i in accounts:
            if i.app:
                org_id = i.app.org_id
            elif i.systemuser:
                org_id = i.systemuser.org_id
            else:
                org_id = ''
            if org_id:
                i.org_id = org_id
        account_model.objects.bulk_update(accounts, ['org_id'])
        print('Update account org is empty: {}-{} using: {:.2f}s'.format(count, count + len(accounts), time.time() - start))
        count += len(accounts)

class Migration(migrations.Migration):
    dependencies = [('applications', '0025_auto_20220817_1346'), ('perms', '0031_auto_20220816_1600'), ('ops', '0022_auto_20220817_1346'), ('assets', '0100_auto_20220711_1413'), ('tickets', '0020_auto_20220817_1346')]
    operations = [migrations.RunPython(migrate_account_dirty_data), migrations.DeleteModel(name='Account'), migrations.DeleteModel(name='HistoricalAccount'), migrations.DeleteModel(name='ApplicationUser')]