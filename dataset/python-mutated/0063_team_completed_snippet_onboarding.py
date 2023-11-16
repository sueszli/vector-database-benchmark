from django.db import migrations, models

def set_history_default(apps, schema_editor):
    if False:
        return 10
    Team = apps.get_model('posthog', 'Team')
    teams = Team.objects.all()
    for team in teams:
        team.completed_snippet_onboarding = True
        team.save()

def backwards(apps, schema_editor):
    if False:
        while True:
            i = 10
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0062_team_anonymize_ips')]
    operations = [migrations.AddField(model_name='team', name='completed_snippet_onboarding', field=models.BooleanField(default=False)), migrations.RunPython(set_history_default, reverse_code=backwards, elidable=True)]