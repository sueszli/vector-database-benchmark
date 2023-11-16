from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration

def remove_depricated_dynamic_sampling_data(apps, schema_editor):
    if False:
        print('Hello World!')
    "\n    Delete the rows in the ProjectOption table that relate to plugins we've deleted.\n    "
    ProjectOption = apps.get_model('sentry', 'ProjectOption')
    for project_option in ProjectOption.objects.filter(key='sentry:dynamic_sampling'):
        project_option.delete()

class Migration(CheckedMigration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0377_groupedmesssage_type_individual_index')]
    operations = [migrations.RunPython(remove_depricated_dynamic_sampling_data, migrations.RunPython.noop, hints={'tables': ['sentry_projectoptions']})]