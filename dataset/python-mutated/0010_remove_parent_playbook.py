from django.db import migrations

def migrate(apps, schema_editor):
    if False:
        while True:
            i = 10
    ...

def reverse_migrate(apps, schema_editor):
    if False:
        return 10
    ConnectorReport = apps.get_model('connectors_manager', 'ConnectorReport')
    for report in ConnectorReport.objects.all():
        report.parent_playbook = report.job.playbook_to_execute
        report.save()

class Migration(migrations.Migration):
    dependencies = [('connectors_manager', '0009_parent_playbook_foreign_key'), ('api_app', '0022_single_playbook_post_migration')]
    operations = [migrations.RunPython(migrate, reverse_migrate), migrations.RemoveField(model_name='connectorreport', name='parent_playbook')]