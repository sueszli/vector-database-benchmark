from django.db import migrations, models
from posthog.models.utils import generate_random_token

def add_signup_tokens(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Team = apps.get_model('posthog', 'Team')
    for team in Team.objects.filter(signup_token__isnull=True):
        team.signup_token = generate_random_token(22)
        team.save()

def backwards(apps, schema_editor):
    if False:
        print('Hello World!')
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0030_migrate_dashboard_days')]
    operations = [migrations.AddField(model_name='team', name='signup_token', field=models.CharField(blank=True, max_length=200, null=True)), migrations.RunPython(add_signup_tokens, backwards, elidable=True)]