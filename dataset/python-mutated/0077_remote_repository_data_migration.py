from django.db import migrations

def migrate_data(apps, schema_editor):
    if False:
        while True:
            i = 10
    RemoteRepository = apps.get_model('oauth', 'RemoteRepository')
    queryset = RemoteRepository.objects.filter(project__isnull=False).select_related('project').only('pk', 'project')
    for rr in queryset.iterator():
        rr.project.remote_repository_id = rr.pk
        rr.project.save()

class Migration(migrations.Migration):
    dependencies = [('projects', '0076_project_remote_repository')]
    operations = [migrations.RunPython(migrate_data)]