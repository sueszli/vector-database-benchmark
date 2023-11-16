from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def delete_organization_mappings(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    OrganizationMapping = apps.get_model('sentry', 'OrganizationMapping')
    for organization_mapping in RangeQuerySetWrapperWithProgressBar(OrganizationMapping.objects.all()):
        organization_mapping.delete()

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0471_dashboard_widget_description')]
    operations = [migrations.RunPython(delete_organization_mappings, migrations.RunPython.noop, hints={'tables': ['sentry_organizationmapping']})]