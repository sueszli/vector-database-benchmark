import logging
from enum import Enum
from django.db import migrations
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

class TransactionMetric(Enum):
    DURATION = 1
    LCP = 2

def migrate_project_transaction_thresholds(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    "\n    Migrate an org's apdex thresholds (if they exist) to the\n    projects transaction thresholds in the org.\n    "
    OrganizationOption = apps.get_model('sentry', 'OrganizationOption')
    Project = apps.get_model('sentry', 'Project')
    ProjectTransactionThreshold = apps.get_model('sentry', 'ProjectTransactionThreshold')
    for option in RangeQuerySetWrapperWithProgressBar(OrganizationOption.objects.all()):
        if option.key != 'sentry:apdex_threshold':
            continue
        for project in Project.objects.filter(organization_id=option.organization_id):
            try:
                ProjectTransactionThreshold.objects.get_or_create(organization_id=option.organization_id, project_id=project.id, defaults={'threshold': option.value, 'metric': TransactionMetric.DURATION.value})
            except Exception:
                logging.exception(f'Error migrating project {project.id} for organization {option.organization_id}')

class Migration(migrations.Migration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0209_avatar_remove_file_fk')]
    operations = [migrations.RunPython(migrate_project_transaction_thresholds, migrations.RunPython.noop, hints={'tables': ['sentry_projecttransactionthreshold']})]