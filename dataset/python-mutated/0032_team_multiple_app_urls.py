import django.contrib.postgres.fields
from django.db import migrations, models

def migrate_to_array(apps, schema_editor):
    if False:
        return 10
    Team = apps.get_model('posthog', 'Team')
    for mm in Team.objects.all():
        mm.app_urls = [mm.app_url]
        mm.save()

def rollback_to_string(apps, schema_editor):
    if False:
        print('Hello World!')
    Team = apps.get_model('posthog', 'Team')
    for mm in Team.objects.all():
        mm.app_url = mm.app_urls[0]
        mm.save()

class Migration(migrations.Migration):
    dependencies = [('posthog', '0031_team_signup_token')]
    operations = [migrations.AddField(model_name='team', name='app_urls', field=django.contrib.postgres.fields.ArrayField(base_field=models.CharField(blank=True, max_length=200, null=True), default=list, size=None)), migrations.RunPython(migrate_to_array, rollback_to_string, elidable=True), migrations.RemoveField(model_name='team', name='app_url')]