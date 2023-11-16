from django.db import migrations

def migrate(apps, schema_editor):
    if False:
        while True:
            i = 10
    AnalyzerConfig = apps.get_model('analyzers_manager', 'AnalyzerConfig')
    for analyzer in AnalyzerConfig.objects.filter(python_module='vt.vt3_get.VirusTotalv3', maximum_tlp__in=['AMBER', 'RED']):
        analyzer.params['url_sub_path'] = {'default': '', 'type': 'str', 'description': 'if you want to query a specific subpath of the base endpoint, i.e: `analyses`'}
        analyzer.save()

def reverse_migrate(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    AnalyzerConfig = apps.get_model('analyzers_manager', 'AnalyzerConfig')
    for analyzer in AnalyzerConfig.objects.filter(python_module='vt.vt3_get.VirusTotalv3', maximum_tlp__in=['AMBER', 'RED']):
        analyzer.params.pop('url_sub_path', None)

class Migration(migrations.Migration):
    dependencies = [('analyzers_manager', '00010_tlp')]
    operations = [migrations.RunPython(migrate, reverse_migrate)]