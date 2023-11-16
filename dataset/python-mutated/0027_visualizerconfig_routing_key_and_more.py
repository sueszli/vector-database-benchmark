import django.core.validators
from django.db import migrations, models

def migrate(apps, schema_editor):
    if False:
        return 10
    Config = apps.get_model('visualizers_manager', 'VisualizerConfig')
    for config in Config.objects.all():
        config.soft_time_limit = config.config['soft_time_limit']
        config.routing_key = config.config['queue']
        config.save()

def reverse_migrate(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Config = apps.get_model('visualizers_manager', 'VisualizerConfig')
    for config in Config.objects.all():
        config.config = {'soft_time_limit': config.soft_time_limit, 'queue': config.routing_key}
        config.save()

class Migration(migrations.Migration):
    dependencies = [('visualizers_manager', '0026_alter_visualizerconfig_python_module')]
    operations = [migrations.AddField(model_name='visualizerconfig', name='routing_key', field=models.CharField(default='default', max_length=50)), migrations.AddField(model_name='visualizerconfig', name='soft_time_limit', field=models.IntegerField(default=60, validators=[django.core.validators.MinValueValidator(0)])), migrations.RunPython(migrate, reverse_migrate), migrations.RemoveField(model_name='visualizerconfig', name='config')]