from django.db import IntegrityError, migrations
from django.db.models.functions import Lower
from sentry.constants import RESERVED_ORGANIZATION_SLUGS
from sentry.db.models.utils import slugify_instance
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def fix_org_slug_casing(apps, schema_editor):
    if False:
        print('Hello World!')
    Organization = apps.get_model('sentry', 'Organization')
    query = Organization.objects.exclude(slug=Lower('slug'))
    for org in RangeQuerySetWrapperWithProgressBar(query):
        try:
            org.slug = org.slug.lower()
            org.save(update_fields=['slug'])
        except IntegrityError:
            slugify_instance(org, org.slug, reserved=RESERVED_ORGANIZATION_SLUGS)
            org.save(update_fields=['slug'])

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0380_backfill_monitor_env_initial')]
    operations = [migrations.RunPython(fix_org_slug_casing, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_organization']})]