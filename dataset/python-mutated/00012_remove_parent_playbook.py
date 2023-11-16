from django.db import migrations

def migrate(apps, schema_editor):
    if False:
        return 10
    ...

def reverse_migrate(apps, schema_editor):
    if False:
        return 10
    AnalyzerReport = apps.get_model('analyzers_manager', 'AnalyzerReport')
    for report in AnalyzerReport.objects.all():
        report.parent_playbook = report.job.playbook_to_execute
        report.save()

class Migration(migrations.Migration):
    dependencies = [('analyzers_manager', '00011_vt_url_subpath'), ('api_app', '0022_single_playbook_post_migration')]
    operations = [migrations.RunPython(migrate, reverse_migrate), migrations.RemoveField(model_name='analyzerreport', name='parent_playbook')]