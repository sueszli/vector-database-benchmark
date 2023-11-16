from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def delete_flat_files(apps, schema_editor):
    if False:
        print('Hello World!')
    FlatFileIndexState = apps.get_model('sentry', 'FlatFileIndexState')
    ArtifactBundleFlatFileIndex = apps.get_model('sentry', 'ArtifactBundleFlatFileIndex')
    FlatFileIndexState.objects.raw('TRUNCATE sentry_flatfileindexstate')
    for obj in RangeQuerySetWrapperWithProgressBar(ArtifactBundleFlatFileIndex.objects.select_related('flat_file_index').all()):
        if obj.flat_file_index:
            obj.flat_file_index.delete()
        obj.delete()

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0527_backfill_next_checkin_latest')]
    operations = [migrations.RunPython(delete_flat_files, migrations.RunPython.noop, hints={'tables': ['sentry_flatfileindexstate', 'sentry_artifactbundleflatfileindex']})]