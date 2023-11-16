import django.contrib.postgres.fields.jsonb
import django.db.models.deletion
from django.db import migrations
from posthog.constants import TREND_FILTER_TYPE_ACTIONS

def move_funnel_steps(apps, schema_editor):
    if False:
        while True:
            i = 10
    Funnel = apps.get_model('posthog', 'Funnel')
    for funnel in Funnel.objects.all():
        funnel.filters = {'actions': [{'id': step.action_id, 'order': step.order, 'type': TREND_FILTER_TYPE_ACTIONS} for step in funnel.steps.all()]}
        funnel.save()

def revert_funnel_steps(apps, schema_editor):
    if False:
        print('Hello World!')
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0043_slack_webhooks')]
    operations = [migrations.AddField(model_name='funnel', name='filters', field=django.contrib.postgres.fields.jsonb.JSONField(default=dict)), migrations.RunPython(move_funnel_steps, revert_funnel_steps, elidable=True)]