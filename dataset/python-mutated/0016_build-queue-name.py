from django.db import migrations

def update_build_queue(apps, schema):
    if False:
        while True:
            i = 10
    'Update project build queue to include the previously implied build-\n    prefix.'
    Project = apps.get_model('projects', 'Project')
    for project in Project.objects.all():
        if project.build_queue is not None:
            if not project.build_queue.startswith('build-'):
                project.build_queue = 'build-{}'.format(project.build_queue)
                project.save()

class Migration(migrations.Migration):
    dependencies = [('projects', '0015_add_project_allow_promos')]
    operations = [migrations.RunPython(update_build_queue)]