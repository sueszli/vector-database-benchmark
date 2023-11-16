from django.contrib.postgres.functions import RandomUUID
from django.db import migrations

def fill_missing_uuid_on_users(apps, _schema_editor):
    if False:
        while True:
            i = 10
    User = apps.get_model('account', 'User')
    User.objects.update(uuid=RandomUUID())

class Migration(migrations.Migration):
    dependencies = [('account', '0068_user_uuid')]
    operations = [migrations.RunPython(fill_missing_uuid_on_users, reverse_code=migrations.RunPython.noop)]