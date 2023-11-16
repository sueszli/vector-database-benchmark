from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_organization_integration_denormalization(apps, schema_editor):
    if False:
        while True:
            i = 10
    PagerDutyService = apps.get_model('sentry', 'PagerDutyService')
    RepositoryProjectPathConfig = apps.get_model('sentry', 'RepositoryProjectPathConfig')
    OrganizationIntegration = apps.get_model('sentry', 'OrganizationIntegration')

    def backfill(self) -> None:
        if False:
            i = 10
            return i + 15
        if self.organization_id is None or self.integration_id is None:
            oi = OrganizationIntegration.objects.get(id=self.organization_integration_id)
            self.organization_id = oi.organization_id
            self.integration_id = oi.integration_id
        self.save()
    for obj in RangeQuerySetWrapperWithProgressBar(PagerDutyService.objects.all()):
        backfill(obj)
    for obj in RangeQuerySetWrapperWithProgressBar(RepositoryProjectPathConfig.objects.all()):
        backfill(obj)

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0412_org_integration_denormalization')]
    operations = [migrations.RunPython(backfill_organization_integration_denormalization, reverse_code=migrations.RunPython.noop, hints={'tables': ['pagerdutyservice', 'repositoryprojectpathconfig']})]