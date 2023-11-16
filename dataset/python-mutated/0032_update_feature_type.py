from django.db import migrations
from features.feature_types import CONFIG, FLAG, STANDARD

def update_feature_type(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Feature = apps.get_model('features', 'Feature')
    Feature.objects.filter(type__in=[FLAG, CONFIG]).update(type=STANDARD)

def reverse(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    pass

class Migration(migrations.Migration):
    dependencies = [('features', '0031_merge_20210409_1621')]
    operations = [migrations.RunPython(update_feature_type, reverse_code=lambda *args: None)]