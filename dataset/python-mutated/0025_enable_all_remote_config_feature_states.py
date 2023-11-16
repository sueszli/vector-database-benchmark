from django.db import migrations

def enable_all_remote_config_feature_states(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    FeatureState = apps.get_model('features', 'FeatureState')
    FeatureState.objects.filter(feature__type='CONFIG').update(enabled=True)

def reverse(apps, schema_editor):
    if False:
        print('Hello World!')
    pass

class Migration(migrations.Migration):
    dependencies = [('features', '0024_auto_20200917_1032')]
    operations = [migrations.RunPython(enable_all_remote_config_feature_states, reverse_code=reverse)]