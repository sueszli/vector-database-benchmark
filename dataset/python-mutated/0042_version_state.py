from django.db import migrations, models

def forwards_func(apps, schema_editor):
    if False:
        print('Hello World!')
    '\n    Migrate external Versions with `active=False` to use `state=Closed` and `active=True`.\n    '
    Version = apps.get_model('builds', 'Version')
    Version.objects.filter(type='external', active=False).update(active=True, state='closed')

class Migration(migrations.Migration):
    dependencies = [('builds', '0041_track_task_id')]
    operations = [migrations.AddField(model_name='version', name='state', field=models.CharField(blank=True, choices=[('open', 'Open'), ('closed', 'Closed')], help_text='State of the PR/MR associated to this version.', max_length=20, null=True, verbose_name='State')), migrations.RunPython(forwards_func)]