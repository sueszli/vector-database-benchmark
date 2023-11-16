from django.db import migrations, models
from features.feature_types import STANDARD

def set_default_feature_type(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    feature_model_class = apps.get_model('features', 'feature')
    feature_model_class.objects.filter(type__isnull=True).update(type=STANDARD)

class Migration(migrations.Migration):
    dependencies = [('features', '0046_add_uuid_field_to_feature_segment')]
    operations = [migrations.RunPython(set_default_feature_type, reverse_code=migrations.RunPython.noop), migrations.AlterField(model_name='feature', name='type', field=models.CharField(blank=True, default='STANDARD', max_length=50)), migrations.AlterField(model_name='historicalfeature', name='type', field=models.CharField(blank=True, default='STANDARD', max_length=50))]