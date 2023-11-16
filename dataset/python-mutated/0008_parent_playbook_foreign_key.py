import django.db.models.deletion
from django.db import migrations, models

def migrate(apps, schema_editor):
    if False:
        while True:
            i = 10
    PlaybookConfig = apps.get_model('playbooks_manager', 'PlaybookConfig')
    VisualizerReport = apps.get_model('visualizers_manager', 'VisualizerReport')
    for report in VisualizerReport.objects.all():
        if report.parent_playbook:
            report.parent_playbook2 = PlaybookConfig.objects.get(name=report.parent_playbook)
        else:
            report.parent_playbook2 = None
        report.save()

def backwards_migrate(apps, schema_editor):
    if False:
        while True:
            i = 10
    VisualizerReport = apps.get_model('visualizers_manager', 'VisualizerReport')
    for report in VisualizerReport.objects.all():
        if report.parent_playbook:
            report.parent_playbook = report.parent_playbook2.name
        else:
            report.parent_playbook = ''
        report.save()

class Migration(migrations.Migration):
    dependencies = [('playbooks_manager', '0004_datamigration'), ('visualizers_manager', '0007_auto_20230308_1623')]
    operations = [migrations.AddField(model_name='visualizerreport', name='parent_playbook2', field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='visualizerreports', to='playbooks_manager.playbookconfig')), migrations.RunPython(migrate, backwards_migrate), migrations.RemoveField(model_name='visualizerreport', name='parent_playbook'), migrations.RenameField(model_name='visualizerreport', old_name='parent_playbook2', new_name='parent_playbook')]