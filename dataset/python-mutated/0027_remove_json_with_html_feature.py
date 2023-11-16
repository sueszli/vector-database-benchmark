from django.db import migrations
FEATURE_ID = 'build_json_artifacts_with_html'

def forward_add_feature(apps, schema_editor):
    if False:
        print('Hello World!')
    Feature = apps.get_model('projects', 'Feature')
    try:
        Feature.objects.get(feature_id=FEATURE_ID).delete()
    except Feature.DoesNotExist:
        pass

def reverse_add_feature(apps, schema_editor):
    if False:
        while True:
            i = 10
    Feature = apps.get_model('projects', 'Feature')
    Feature.objects.create(feature_id=FEATURE_ID)

class Migration(migrations.Migration):
    dependencies = [('projects', '0026_ad-free-option')]
    operations = [migrations.RunPython(forward_add_feature, reverse_add_feature)]