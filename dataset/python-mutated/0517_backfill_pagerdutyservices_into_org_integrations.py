from django.db import ProgrammingError, migrations, router, transaction
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapper

def as_dict(pds):
    if False:
        for i in range(10):
            print('nop')
    return dict(integration_id=pds.integration_id, integration_key=pds.integration_key, service_name=pds.service_name, id=pds.id)

def backfill_pagerdutyservices(apps, schema_editor):
    if False:
        while True:
            i = 10
    PagerDutyService = apps.get_model('sentry', 'PagerDutyService')
    OrganizationIntegration = apps.get_model('sentry', 'OrganizationIntegration')
    try:
        PagerDutyService.objects.first()
    except ProgrammingError:
        return
    for pds in RangeQuerySetWrapper(PagerDutyService.objects.all()):
        try:
            with transaction.atomic(router.db_for_write(OrganizationIntegration)):
                org_integration = OrganizationIntegration.objects.filter(id=pds.organization_integration_id).select_for_update().get()
                existing = org_integration.config.get('pagerduty_services', [])
                org_integration.config['pagerduty_services'] = [row for row in existing if row['id'] != pds.id] + [as_dict(pds)]
                org_integration.save()
        except OrganizationIntegration.DoesNotExist:
            pass

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0516_switch_pagerduty_silo')]
    operations = [migrations.RunPython(backfill_pagerdutyservices, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_organizationintegration']})]