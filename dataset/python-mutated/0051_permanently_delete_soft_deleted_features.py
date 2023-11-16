from django.db import migrations

def permanently_delete_features(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Feature = apps.get_model('features', 'Feature')
    FeatureState = apps.get_model('features', 'FeatureState')
    Feature.objects.filter(deleted_at__isnull=False).delete()
    FeatureState.objects.filter(deleted_at__isnull=False).delete()

class Migration(migrations.Migration):
    dependencies = [('features', '0050_remove_unique_indexes')]
    operations = [migrations.RunPython(migrations.RunPython.noop, reverse_code=permanently_delete_features)]