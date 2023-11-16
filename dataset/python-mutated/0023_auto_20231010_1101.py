import uuid
import authentication.models.access_key
from django.db import migrations, models

def migrate_access_key_secret(apps, schema_editor):
    if False:
        while True:
            i = 10
    access_key_model = apps.get_model('authentication', 'AccessKey')
    db_alias = schema_editor.connection.alias
    batch_size = 100
    count = 0
    while True:
        access_keys = access_key_model.objects.using(db_alias).all()[count:count + batch_size]
        if not access_keys:
            break
        count += len(access_keys)
        access_keys_updated = []
        for access_key in access_keys:
            s = access_key.secret
            if len(s) != 32 or not s.islower():
                continue
            try:
                access_key.secret = '%s-%s-%s-%s-%s' % (s[:8], s[8:12], s[12:16], s[16:20], s[20:])
                access_keys_updated.append(access_key)
            except (ValueError, IndexError):
                pass
        access_key_model.objects.bulk_update(access_keys_updated, fields=['secret'])

class Migration(migrations.Migration):
    dependencies = [('authentication', '0022_passkey')]
    operations = [migrations.AddField(model_name='accesskey', name='date_last_used', field=models.DateTimeField(blank=True, null=True, verbose_name='Date last used')), migrations.AddField(model_name='privatetoken', name='date_last_used', field=models.DateTimeField(blank=True, null=True, verbose_name='Date last used')), migrations.AlterField(model_name='accesskey', name='secret', field=models.CharField(default=authentication.models.access_key.default_secret, max_length=36, verbose_name='AccessKeySecret')), migrations.RunPython(migrate_access_key_secret)]