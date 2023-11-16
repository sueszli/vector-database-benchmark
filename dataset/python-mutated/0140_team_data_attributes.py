import django.contrib.postgres.fields
from django.db import migrations

def set_default_data_attributes(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Team = apps.get_model('posthog', 'Team')
    Team.objects.update(data_attributes=['data-attr'])

class Migration(migrations.Migration):
    dependencies = [('posthog', '0139_dashboard_tagging')]
    operations = [migrations.AddField(model_name='team', name='data_attributes', field=django.contrib.postgres.fields.jsonb.JSONField(default=['data-attr']))]