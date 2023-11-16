from __future__ import annotations
from enum import Enum
from typing import Optional
from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar
'\nThe below code was copied over from src/sentry/types/integrations.py\n and src/sentry/notifications/types.py because some of these enums will be deleted\n'

class ExternalProviders(Enum):
    EMAIL = 100
    SLACK = 110
    MSTEAMS = 120

class ExternalProviderEnum(Enum):
    EMAIL = 'email'
    SLACK = 'slack'
    MSTEAMS = 'msteams'
EXTERNAL_PROVIDERS = {ExternalProviders.EMAIL: ExternalProviderEnum.EMAIL.value, ExternalProviders.SLACK: ExternalProviderEnum.SLACK.value, ExternalProviders.MSTEAMS: ExternalProviderEnum.MSTEAMS.value}

def get_provider_name(value: int) -> Optional[str]:
    if False:
        return 10
    return EXTERNAL_PROVIDERS.get(ExternalProviders(value))
"\nTODO(postgres): We've encoded these enums as integers to facilitate\ncommunication with the DB. We'd prefer to encode them as strings to facilitate\ncommunication with the API and plan to do so as soon as we use native enums in\nPostgres. In the meantime each enum has an adjacent object that maps the\nintegers to their string values.\n"

def get_notification_setting_type_name(value: int | NotificationSettingTypes) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    return NOTIFICATION_SETTING_TYPES.get(NotificationSettingTypes(value))

def get_notification_scope_name(value: int) -> Optional[str]:
    if False:
        while True:
            i = 10
    return NOTIFICATION_SCOPE_TYPE.get(NotificationScopeType(value))

class NotificationSettingTypes(Enum):
    """
    Each of these categories of Notification settings has at least an option for
    "on" or "off". Workflow also includes SUBSCRIBE_ONLY and Deploy also
    includes COMMITTED_ONLY and both of these values are described below.
    """
    DEFAULT = 0
    DEPLOY = 10
    ISSUE_ALERTS = 20
    WORKFLOW = 30
    ACTIVE_RELEASE = 31
    APPROVAL = 40
    QUOTA = 50
    QUOTA_ERRORS = 51
    QUOTA_TRANSACTIONS = 52
    QUOTA_ATTACHMENTS = 53
    QUOTA_REPLAYS = 56
    QUOTA_WARNINGS = 54
    QUOTA_SPEND_ALLOCATIONS = 55
    SPIKE_PROTECTION = 60
    MISSING_MEMBERS = 70
    REPORTS = -1

class NotificationSettingEnum(Enum):
    DEFAULT = 'default'
    DEPLOY = 'deploy'
    ISSUE_ALERTS = 'alerts'
    WORKFLOW = 'workflow'
    ACTIVE_RELEASE = 'activeRelease'
    APPROVAL = 'approval'
    QUOTA = 'quota'
    QUOTA_ERRORS = 'quotaErrors'
    QUOTA_TRANSACTIONS = 'quotaTransactions'
    QUOTA_ATTACHMENTS = 'quotaAttachments'
    QUOTA_REPLAYS = 'quotaReplays'
    QUOTA_WARNINGS = 'quotaWarnings'
    QUOTA_SPEND_ALLOCATIONS = 'quotaSpendAllocations'
    SPIKE_PROTECTION = 'spikeProtection'
    MISSING_MEMBERS = 'missingMembers'
    REPORTS = 'reports'
NOTIFICATION_SETTING_TYPES = {NotificationSettingTypes.DEFAULT: NotificationSettingEnum.DEFAULT.value, NotificationSettingTypes.DEPLOY: NotificationSettingEnum.DEPLOY.value, NotificationSettingTypes.ISSUE_ALERTS: NotificationSettingEnum.ISSUE_ALERTS.value, NotificationSettingTypes.WORKFLOW: NotificationSettingEnum.WORKFLOW.value, NotificationSettingTypes.ACTIVE_RELEASE: NotificationSettingEnum.ACTIVE_RELEASE.value, NotificationSettingTypes.APPROVAL: NotificationSettingEnum.APPROVAL.value, NotificationSettingTypes.QUOTA: NotificationSettingEnum.QUOTA.value, NotificationSettingTypes.QUOTA_ERRORS: NotificationSettingEnum.QUOTA_ERRORS.value, NotificationSettingTypes.QUOTA_TRANSACTIONS: NotificationSettingEnum.QUOTA_TRANSACTIONS.value, NotificationSettingTypes.QUOTA_ATTACHMENTS: NotificationSettingEnum.QUOTA_ATTACHMENTS.value, NotificationSettingTypes.QUOTA_REPLAYS: NotificationSettingEnum.QUOTA_REPLAYS.value, NotificationSettingTypes.QUOTA_WARNINGS: NotificationSettingEnum.QUOTA_WARNINGS.value, NotificationSettingTypes.QUOTA_SPEND_ALLOCATIONS: NotificationSettingEnum.QUOTA_SPEND_ALLOCATIONS.value, NotificationSettingTypes.SPIKE_PROTECTION: NotificationSettingEnum.SPIKE_PROTECTION.value, NotificationSettingTypes.REPORTS: NotificationSettingEnum.REPORTS.value}

class NotificationSettingOptionValues(Enum):
    """
    An empty row in the DB should be represented as
    NotificationSettingOptionValues.DEFAULT.
    """
    DEFAULT = 0
    NEVER = 10
    ALWAYS = 20
    SUBSCRIBE_ONLY = 30
    COMMITTED_ONLY = 40

class NotificationSettingsOptionEnum(Enum):
    DEFAULT = 'default'
    NEVER = 'never'
    ALWAYS = 'always'
    SUBSCRIBE_ONLY = 'subscribe_only'
    COMMITTED_ONLY = 'committed_only'
NOTIFICATION_SETTING_OPTION_VALUES = {NotificationSettingOptionValues.DEFAULT: NotificationSettingsOptionEnum.DEFAULT.value, NotificationSettingOptionValues.NEVER: NotificationSettingsOptionEnum.NEVER.value, NotificationSettingOptionValues.ALWAYS: NotificationSettingsOptionEnum.ALWAYS.value, NotificationSettingOptionValues.SUBSCRIBE_ONLY: NotificationSettingsOptionEnum.SUBSCRIBE_ONLY.value, NotificationSettingOptionValues.COMMITTED_ONLY: NotificationSettingsOptionEnum.COMMITTED_ONLY.value}

class NotificationScopeEnum(Enum):
    USER = 'user'
    ORGANIZATION = 'organization'
    PROJECT = 'project'
    TEAM = 'team'

class NotificationScopeType(Enum):
    USER = 0
    ORGANIZATION = 10
    PROJECT = 20
    TEAM = 30
NOTIFICATION_SCOPE_TYPE = {NotificationScopeType.USER: NotificationScopeEnum.USER.value, NotificationScopeType.ORGANIZATION: NotificationScopeEnum.ORGANIZATION.value, NotificationScopeType.PROJECT: NotificationScopeEnum.PROJECT.value, NotificationScopeType.TEAM: NotificationScopeEnum.TEAM.value}
'\nEnd of copied over code\n'

def backfill_notification_settings(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    NotificationSetting = apps.get_model('sentry', 'NotificationSetting')
    NotificationSettingOption = apps.get_model('sentry', 'NotificationSettingOption')
    NotificationSettingProvider = apps.get_model('sentry', 'NotificationSettingProvider')
    for setting in RangeQuerySetWrapperWithProgressBar(NotificationSetting.objects.all()):
        related_settings = NotificationSetting.objects.filter(scope_type=setting.scope_type, scope_identifier=setting.scope_identifier, user_id=setting.user_id, team_id=setting.team_id, type=setting.type)
        enabled_providers = []
        all_providers = []
        enabled_value = None
        for related_setting in related_settings:
            if related_setting.value != NotificationSettingOptionValues.NEVER.value:
                enabled_providers.append(related_setting.provider)
                enabled_value = related_setting.value
            all_providers.append(related_setting.provider)
        update_args = {'type': get_notification_setting_type_name(related_setting.type), 'user_id': related_setting.user_id, 'team_id': related_setting.team_id, 'scope_type': get_notification_scope_name(related_setting.scope_type), 'scope_identifier': related_setting.scope_identifier}
        if len(enabled_providers) == 0:
            NotificationSettingOption.objects.update_or_create(**update_args, defaults={'value': NotificationSettingsOptionEnum.NEVER.value})
        else:
            NotificationSettingOption.objects.update_or_create(**update_args, defaults={'value': NOTIFICATION_SETTING_OPTION_VALUES[NotificationSettingOptionValues(enabled_value)]})
        if related_setting.scope_type in [NotificationScopeType.USER.value, NotificationScopeType.TEAM.value]:
            for provider in enabled_providers:
                NotificationSettingProvider.objects.update_or_create(**update_args, provider=get_provider_name(provider), defaults={'value': NotificationSettingsOptionEnum.ALWAYS.value})
            disabled_providers = set(all_providers) - set(enabled_providers)
            for provider in disabled_providers:
                NotificationSettingProvider.objects.update_or_create(**update_args, provider=get_provider_name(provider), defaults={'value': NotificationSettingsOptionEnum.NEVER.value})

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0560_add_monitorincident_table')]
    operations = [migrations.RunPython(backfill_notification_settings, reverse_code=migrations.RunPython.noop, hints={'tables': ['sentry_notificationsetting', 'sentry_notificationsettingoption', 'sentry_notificationsettingprovider']})]