from django.db import migrations

def migrate(apps, schema_editor):
    if False:
        print('Hello World!')
    ...

def reverse_migrate(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    VisualizerReport = apps.get_model('visualizers_manager', 'VisualizerReport')
    for report in VisualizerReport.objects.all():
        report.parent_playbook = report.job.playbook_to_execute
        report.save()

class Migration(migrations.Migration):
    dependencies = [('visualizers_manager', '0008_parent_playbook_foreign_key'), ('api_app', '0022_single_playbook_post_migration')]
    operations = [migrations.RunPython(migrate, reverse_migrate), migrations.RemoveField(model_name='visualizerreport', name='parent_playbook')]