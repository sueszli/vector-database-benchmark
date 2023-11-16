from django.db import migrations

def reset_team_timezone_to_UTC(apps, _) -> None:
    if False:
        print('Hello World!')
    Team = apps.get_model('posthog', 'Team')
    Team.objects.exclude(timezone='UTC').update(timezone='UTC')

class Migration(migrations.Migration):
    dependencies = [('posthog', '0236_add_instance_setting_model')]

    def reverse(apps, _) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass
    operations = [migrations.RunPython(reset_team_timezone_to_UTC, reverse, elidable=True)]