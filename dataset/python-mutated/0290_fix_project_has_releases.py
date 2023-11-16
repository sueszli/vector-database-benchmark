from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_project_has_release(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Project = apps.get_model('sentry', 'Project')
    ReleaseProject = apps.get_model('sentry', 'ReleaseProject')
    for project in RangeQuerySetWrapperWithProgressBar(Project.objects.all()):
        if not project.flags.has_releases and ReleaseProject.objects.filter(project=project).exists():
            project.flags.has_releases = True
            project.save(update_fields=['flags'])

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0289_dashboardwidgetquery_convert_orderby_to_field')]
    operations = [migrations.RunPython(backfill_project_has_release, migrations.RunPython.noop, hints={'tables': ['sentry_release', 'sentry_project', 'sentry_releaseproject']})]