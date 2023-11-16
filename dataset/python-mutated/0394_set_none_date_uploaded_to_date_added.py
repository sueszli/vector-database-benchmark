import django.utils.timezone
from django.db import migrations, models
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def copy_date_added_to_date_uploaded(apps, schema_editor):
    if False:
        return 10
    ArtifactBundle = apps.get_model('sentry', 'ArtifactBundle')
    for bundle in RangeQuerySetWrapperWithProgressBar(ArtifactBundle.objects.filter(date_uploaded__isnull=True)):
        bundle.date_uploaded = bundle.date_added
        bundle.save(update_fields=['date_uploaded'])

class Migration(CheckedMigration):
    is_dangerous = False
    checked = False
    dependencies = [('sentry', '0393_create_groupforecast_table')]
    operations = [migrations.SeparateDatabaseAndState(database_operations=[migrations.RunPython(copy_date_added_to_date_uploaded, migrations.RunPython.noop, hints={'tables': ['sentry_artifactbundle']}), migrations.AlterField(model_name='artifactbundle', name='date_uploaded', field=models.DateTimeField())], state_operations=[migrations.AlterField(model_name='artifactbundle', name='date_uploaded', field=models.DateTimeField(default=django.utils.timezone.now))])]