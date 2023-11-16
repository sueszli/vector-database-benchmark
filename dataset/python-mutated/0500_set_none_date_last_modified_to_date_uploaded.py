from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def copy_date_uploaded_to_date_last_modified(apps, schema_editor):
    if False:
        while True:
            i = 10
    ArtifactBundle = apps.get_model('sentry', 'ArtifactBundle')
    for bundle in RangeQuerySetWrapperWithProgressBar(ArtifactBundle.objects.filter(date_last_modified__isnull=True)):
        bundle.date_last_modified = bundle.date_uploaded
        bundle.save(update_fields=['date_last_modified'])

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0499_typed_bitfield_revert')]
    operations = [migrations.RunPython(copy_date_uploaded_to_date_last_modified, migrations.RunPython.noop, hints={'tables': ['sentry_artifactbundle']})]