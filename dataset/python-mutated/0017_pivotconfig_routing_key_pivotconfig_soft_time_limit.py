import django.core.validators
from django.db import migrations, models

def migrate(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Config = apps.get_model('pivots_manager', 'PivotConfig')
    for config in Config.objects.all():
        config.soft_time_limit = config.config['soft_time_limit']
        config.routing_key = config.config['queue']
        config.save()

def reverse_migrate(apps, schema_editor):
    if False:
        while True:
            i = 10
    Config = apps.get_model('pivots_manager', 'PivotConfig')
    for config in Config.objects.all():
        config.config = {'soft_time_limit': config.soft_time_limit, 'queue': config.routing_key}
        config.save()

class Migration(migrations.Migration):
    dependencies = [('pivots_manager', '0016_alter_pivotconfig_options_and_more')]
    operations = [migrations.AddField(model_name='pivotconfig', name='routing_key', field=models.CharField(default='default', max_length=50)), migrations.AddField(model_name='pivotconfig', name='soft_time_limit', field=models.IntegerField(default=60, validators=[django.core.validators.MinValueValidator(0)])), migrations.RunPython(migrate, reverse_migrate), migrations.RemoveField(model_name='pivotconfig', name='config')]