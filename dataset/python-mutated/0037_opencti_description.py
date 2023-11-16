from django.db import migrations

def migrate(apps, schema_editor):
    if False:
        print('Hello World!')
    AnalyzerConfig = apps.get_model('analyzers_manager', 'AnalyzerConfig')
    ac = AnalyzerConfig.objects.get(name='OpenCTI')
    ac.description = 'scan an observable on a custom OpenCTI instance. CARE! This may require additional advanced configuration. Check the docs [here](https://intelowl.readthedocs.io/en/latest/Advanced-Configuration.html#opencti)'
    ac.full_clean()
    ac.save()
    ConnectorConfig = apps.get_model('connectors_manager', 'ConnectorConfig')
    ac = ConnectorConfig.objects.get(name='OpenCTI')
    ac.description = 'Automatically creates an observable and a linked report on your OpenCTI instance, linking the successful analysis on IntelOwl. CARE! This may require additional advanced configuration. Check the docs [here](https://intelowl.readthedocs.io/en/latest/Advanced-Configuration.html#opencti)'
    ac.full_clean()
    ac.save()

def reverse_migrate(apps, schema_editor):
    if False:
        return 10
    pass

class Migration(migrations.Migration):
    dependencies = [('analyzers_manager', '0036_delete_cryptoscam')]
    operations = [migrations.RunPython(migrate, reverse_migrate)]