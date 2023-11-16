from django.db import migrations

def fix_team_event_names(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Team = apps.get_model('posthog', 'Team')
    for team in Team.objects.all():
        old_event_names = team.event_names
        team.event_names = [event for event in old_event_names if isinstance(event, str)]
        if len(team.event_names) != len(old_event_names):
            from posthog.tasks.calculate_event_property_usage import calculate_event_property_usage_for_team
            team.save()
            calculate_event_property_usage_for_team(team.pk)

def backwards(apps, schema_editor):
    if False:
        while True:
            i = 10
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0113_cohort_is_static')]
    operations = [migrations.RunPython(fix_team_event_names, backwards, elidable=True)]