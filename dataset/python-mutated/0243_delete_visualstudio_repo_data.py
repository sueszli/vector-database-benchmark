from django.db import migrations
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def delete_vsts_repo_data(apps, schema_editor):
    if False:
        return 10
    '\n    Delete the VSTS plugin rows in the Repository table.\n    '
    Repository = apps.get_model('sentry', 'Repository')
    for repository in RangeQuerySetWrapperWithProgressBar(Repository.objects.all()):
        if repository.provider == 'visualstudio':
            repository.delete()

class Migration(migrations.Migration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0242_delete_removed_plugin_data')]
    operations = [migrations.RunPython(delete_vsts_repo_data, migrations.RunPython.noop, hints={'tables': ['sentry_repository']})]