from django.db import migrations

def migrate_features(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Feature = apps.get_model('projects', 'Feature')
    if Feature.objects.filter(feature_id='external_version_build').exists():
        for project in Feature.objects.get(feature_id='external_version_build').projects.all():
            project.external_builds_enabled = True
            project.save()

class Migration(migrations.Migration):
    dependencies = [('projects', '0049_add_external_build_enabled')]
    operations = [migrations.RunPython(migrate_features)]