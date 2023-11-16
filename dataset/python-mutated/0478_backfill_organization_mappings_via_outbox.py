from enum import IntEnum
from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

class OutboxCategory(IntEnum):
    USER_UPDATE = 0
    WEBHOOK_PROXY = 1
    ORGANIZATION_UPDATE = 2
    ORGANIZATION_MEMBER_UPDATE = 3
    VERIFY_ORGANIZATION_MAPPING = 4
    AUDIT_LOG_EVENT = 5
    USER_IP_EVENT = 6
    INTEGRATION_UPDATE = 7
    PROJECT_UPDATE = 8
    API_APPLICATION_UPDATE = 9
    SENTRY_APP_INSTALLATION_UPDATE = 10
    TEAM_UPDATE = 11
    ORGANIZATION_INTEGRATION_UPDATE = 12
    ORGANIZATION_MEMBER_CREATE = 13

class OutboxScope(IntEnum):
    ORGANIZATION_SCOPE = 0
    USER_SCOPE = 1
    WEBHOOK_SCOPE = 2
    AUDIT_LOG_SCOPE = 3
    USER_IP_SCOPE = 4
    INTEGRATION_SCOPE = 5
    APP_SCOPE = 6
    TEAM_SCOPE = 7

class OrganizationStatus(IntEnum):
    ACTIVE = 0
    PENDING_DELETION = 1
    DELETION_IN_PROGRESS = 2

def backfill_org_mapping_via_outbox(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Organization = apps.get_model('sentry', 'Organization')
    RegionOutbox = apps.get_model('sentry', 'RegionOutbox')
    for org in RangeQuerySetWrapperWithProgressBar(Organization.objects.all()):
        if org.status != OrganizationStatus.DELETION_IN_PROGRESS:
            RegionOutbox(shard_scope=OutboxScope.ORGANIZATION_SCOPE, shard_identifier=org.id, category=OutboxCategory.ORGANIZATION_UPDATE, object_identifier=org.id).save()

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0477_control_avatars')]
    operations = [migrations.RunPython(backfill_org_mapping_via_outbox, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_organization', 'sentry_regionoutbox']})]