from django.db import migrations

def forwards_func(apps, schema_editor):
    if False:
        print('Hello World!')
    'Create supported events for webhooks.'
    WebHookEvent = apps.get_model('projects', 'WebHookEvent')
    for event in ['build:triggered', 'build:failed', 'build:passed']:
        WebHookEvent.objects.get_or_create(name=event)

class Migration(migrations.Migration):
    dependencies = [('projects', '0083_init_generic_webhooks')]
    operations = [migrations.RunPython(forwards_func)]