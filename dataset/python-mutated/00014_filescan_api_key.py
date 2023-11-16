from django.db import migrations
from django.db.models import Q

def migrate(apps, schema_editor):
    if False:
        return 10
    AnalyzerConfig = apps.get_model('analyzers_manager', 'AnalyzerConfig')
    for config in AnalyzerConfig.objects.filter(Q(python_module='filescan_search.FileScanSearch') | Q(python_module='filescan.FileScanUpload')):
        config.secrets['api_key'] = {'type': 'str', 'description': 'Api key', 'required': True}
        config.full_clean()
        config.save()

def reverse_migrate(apps, schema_editor):
    if False:
        while True:
            i = 10
    AnalyzerConfig = apps.get_model('analyzers_manager', 'AnalyzerConfig')
    for config in AnalyzerConfig.objects.filter(Q(python_module='filescan_search.FileScanSearch') | Q(python_module='filescan.FileScanUpload')):
        config.secrets = {}
        config.full_clean()
        config.save()

class Migration(migrations.Migration):
    dependencies = [('analyzers_manager', '00013_remove_runtime_configuration')]
    operations = [migrations.RunPython(migrate, reverse_migrate)]