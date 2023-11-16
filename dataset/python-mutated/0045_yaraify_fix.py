from django.db import migrations

def migrate(apps, schema_editor):
    if False:
        return 10
    Parameter = apps.get_model('api_app', 'Parameter')
    for param in Parameter.objects.filter(name='api_key_name', python_module__module='yaraify_file_scan.YARAifyFileScan'):
        param.required = False
        param.full_clean()
        param.save()

def reverse_migrate(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Parameter = apps.get_model('api_app', 'Parameter')
    for param in Parameter.objects.filter(name='api_key_name', python_module__module='yaraify_file_scan.YARAifyFileScan'):
        param.required = True
        param.full_clean()
        param.save()

class Migration(migrations.Migration):
    dependencies = [('analyzers_manager', '0044_analyzerconfig_routing_key_and_more')]
    operations = [migrations.RunPython(migrate, reverse_migrate)]