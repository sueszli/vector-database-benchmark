import secrets
from django.db import migrations, models

def forwards_func(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    User = apps.get_model('posthog', 'User')
    users = User.objects.all()
    for user in users:
        user.distinct_id = secrets.token_urlsafe(32)
        user.save()

def reverse_func(apps, schema_editor):
    if False:
        while True:
            i = 10
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0020_auto_20200210_0212')]
    operations = [migrations.AddField(model_name='user', name='distinct_id', field=models.CharField(blank=True, max_length=200, null=True)), migrations.RunPython(forwards_func, reverse_func, elidable=True)]