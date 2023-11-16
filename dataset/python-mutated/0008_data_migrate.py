import django.db.models.deletion
from django.db import migrations, models

def migrate(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    PivotConfig = apps.get_model('pivots_manager', 'PivotConfig')
    PythonModule = apps.get_model('api_app', 'PythonModule')
    pm = PythonModule.objects.get(module='base.Base', base_path='api_app.pivots_manager.pivots')
    for pivot in PivotConfig.objects.all():
        config = pivot.analyzer_config or pivot.connector_config or pivot.visualizer_config
        pivot.execute_on_python_module = config.python_module
        pivot.python_module = pm
        pivot.save()

def reverse_migrate(apps, schema_editor):
    if False:
        return 10
    ...

class Migration(migrations.Migration):
    dependencies = [('api_app', '0046_remove_pluginconfig_plugin_config_no_config_all_null_and_more'), ('pivots_manager', '0007_pivotreport_rename_pivot_pivotmap_and_more')]
    operations = [migrations.RunPython(migrate, reverse_migrate), migrations.RemoveField(model_name='pivotconfig', name='analyzer_config'), migrations.RemoveField(model_name='pivotconfig', name='connector_config'), migrations.RemoveField(model_name='pivotconfig', name='visualizer_config'), migrations.AlterField(model_name='pivotconfig', name='execute_on_python_module', field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='pivots', to='api_app.pythonmodule')), migrations.AlterField(model_name='pivotconfig', name='python_module', field=models.ForeignKey(null=False, blank=False, on_delete=django.db.models.deletion.PROTECT, related_name='%(class)ss', to='api_app.pythonmodule'))]