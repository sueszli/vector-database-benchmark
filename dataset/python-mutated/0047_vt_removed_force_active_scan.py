from django.db import migrations

def migrate(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Parameter = apps.get_model('api_app', 'Parameter')
    Parameter.objects.filter(name='force_active_scan', python_module__module='vt.vt3_get.VirusTotalv3').delete()

def reverse_migrate(apps, schema_editor):
    if False:
        return 10
    PythonModule = apps.get_model('api_app', 'PythonModule')
    AnalyzerConfig = apps.get_model('analyzers_manager', 'AnalyzerConfig')
    Parameter = apps.get_model('api_app', 'Parameter')
    pm = PythonModule.objects.get(module='vt.vt3_get.VirusTotalv3', base_path='api_app.analyzers_manager.observable_analyzers')
    for config in AnalyzerConfig.objects.filter(python_module=pm):
        Parameter.objects.create(name='force_active_scan', analyzer_config=config, python_module=pm, required=False, type='bool', is_secret=False, description='Force active scan on virus total')

class Migration(migrations.Migration):
    dependencies = [('analyzers_manager', '0046_analyzerconfig_add_requests_timeout_field')]
    operations = [migrations.RunPython(migrate, reverse_migrate)]