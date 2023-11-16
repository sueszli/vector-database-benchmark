from django.db import migrations

def forwards_func(apps, schema_editor):
    if False:
        while True:
            i = 10
    "Old models with provider_data='' are being fetched as str instead of json."
    Integration = apps.get_model('integrations', 'Integration')
    Integration.objects.filter(provider_data='').update(provider_data={})

class Migration(migrations.Migration):
    dependencies = [('integrations', '0006_set-default-value-provider-data')]
    operations = [migrations.RunPython(forwards_func)]