from __future__ import annotations

from collections import namedtuple
from typing import Any, Iterable, Mapping, Optional, Union

from sentry.models.user import User
from sentry.notifications.types import (
    FineTuningAPIKey,
    NotificationScopeType,
    NotificationSettingOptionValues,
    NotificationSettingTypes,
    UserOptionsSettingsKey,
)
from sentry.services.hybrid_cloud.user.model import RpcUser

LegacyUserOptionClone = namedtuple(
    "LegacyUserOptionClone",
    [
        "user",
        "project_id",
        "organization_id",
        "key",
        "value",
    ],
)

USER_OPTION_SETTINGS = {
    UserOptionsSettingsKey.DEPLOY: {
        "key": "deploy-emails",
        "default": "3",
        "type": int,
    },
    UserOptionsSettingsKey.SELF_ACTIVITY: {
        "key": "self_notifications",
        "default": "0",
        "type": bool,
    },
    UserOptionsSettingsKey.SELF_ASSIGN: {
        "key": "self_assign_issue",
        "default": "0",
        "type": bool,
    },
    UserOptionsSettingsKey.SUBSCRIBE_BY_DEFAULT: {
        "key": "subscribe_by_default",
        "default": "1",
        "type": bool,
    },
    UserOptionsSettingsKey.WORKFLOW: {
        "key": "workflow:notifications",
        "default": "1",
        "type": int,
    },
}

KEYS_TO_LEGACY_KEYS = {
    NotificationSettingTypes.DEPLOY: "deploy-emails",
    NotificationSettingTypes.ISSUE_ALERTS: "mail:alert",
    NotificationSettingTypes.WORKFLOW: "workflow:notifications",
}


KEY_VALUE_TO_LEGACY_VALUE = {
    NotificationSettingTypes.DEPLOY: {
        NotificationSettingOptionValues.ALWAYS: 2,
        NotificationSettingOptionValues.COMMITTED_ONLY: 3,
        NotificationSettingOptionValues.NEVER: 4,
    },
    NotificationSettingTypes.ISSUE_ALERTS: {
        NotificationSettingOptionValues.ALWAYS: 1,
        NotificationSettingOptionValues.NEVER: 0,
    },
    NotificationSettingTypes.WORKFLOW: {
        NotificationSettingOptionValues.ALWAYS: 0,
        NotificationSettingOptionValues.SUBSCRIBE_ONLY: 1,
        NotificationSettingOptionValues.NEVER: 2,
    },
}

LEGACY_VALUE_TO_KEY = {
    NotificationSettingTypes.DEPLOY: {
        -1: NotificationSettingOptionValues.DEFAULT,
        2: NotificationSettingOptionValues.ALWAYS,
        3: NotificationSettingOptionValues.COMMITTED_ONLY,
        4: NotificationSettingOptionValues.NEVER,
    },
    NotificationSettingTypes.ISSUE_ALERTS: {
        -1: NotificationSettingOptionValues.DEFAULT,
        0: NotificationSettingOptionValues.NEVER,
        1: NotificationSettingOptionValues.ALWAYS,
    },
    NotificationSettingTypes.WORKFLOW: {
        -1: NotificationSettingOptionValues.DEFAULT,
        0: NotificationSettingOptionValues.ALWAYS,
        1: NotificationSettingOptionValues.SUBSCRIBE_ONLY,
        2: NotificationSettingOptionValues.NEVER,
    },
}


def get_legacy_key(type: NotificationSettingTypes, scope_type: NotificationScopeType) -> str | None:
    """Temporary mapping from new enum types to legacy strings."""
    if scope_type == NotificationScopeType.USER and type == NotificationSettingTypes.ISSUE_ALERTS:
        return "subscribe_by_default"

    return KEYS_TO_LEGACY_KEYS.get(type)


def get_legacy_value(type: NotificationSettingTypes, value: NotificationSettingOptionValues) -> str:
    """
    Temporary mapping from new enum types to legacy strings. Each type has a separate mapping.
    """

    return str(KEY_VALUE_TO_LEGACY_VALUE.get(type, {}).get(value))


def get_option_value_from_boolean(value: bool) -> NotificationSettingOptionValues:
    if value:
        return NotificationSettingOptionValues.ALWAYS
    else:
        return NotificationSettingOptionValues.NEVER


def get_option_value_from_int(
    type: NotificationSettingTypes, value: int
) -> NotificationSettingOptionValues | None:
    return LEGACY_VALUE_TO_KEY.get(type, {}).get(value)


def get_type_from_fine_tuning_key(key: FineTuningAPIKey) -> NotificationSettingTypes | None:
    return {
        FineTuningAPIKey.ALERTS: NotificationSettingTypes.ISSUE_ALERTS,
        FineTuningAPIKey.DEPLOY: NotificationSettingTypes.DEPLOY,
        FineTuningAPIKey.WORKFLOW: NotificationSettingTypes.WORKFLOW,
    }.get(key)


def get_type_from_user_option_settings_key(
    key: UserOptionsSettingsKey,
) -> NotificationSettingTypes | None:
    return {
        UserOptionsSettingsKey.DEPLOY: NotificationSettingTypes.DEPLOY,
        UserOptionsSettingsKey.WORKFLOW: NotificationSettingTypes.WORKFLOW,
        UserOptionsSettingsKey.SUBSCRIBE_BY_DEFAULT: NotificationSettingTypes.ISSUE_ALERTS,
    }.get(key)


def get_key_from_legacy(key: str) -> NotificationSettingTypes | None:
    return {
        "deploy-emails": NotificationSettingTypes.DEPLOY,
        "mail:alert": NotificationSettingTypes.ISSUE_ALERTS,
        "subscribe_by_default": NotificationSettingTypes.ISSUE_ALERTS,
        "workflow:notifications": NotificationSettingTypes.WORKFLOW,
    }.get(key)


def get_key_value_from_legacy(
    key: str, value: Any
) -> tuple[NotificationSettingTypes | None, NotificationSettingOptionValues | None]:
    type = get_key_from_legacy(key)
    if type not in LEGACY_VALUE_TO_KEY:
        return None, None
    option_value = LEGACY_VALUE_TO_KEY.get(type, {}).get(int(value))

    return type, option_value


def get_legacy_object(
    notification_setting: Any,
    user_mapping: Optional[Mapping[int, Union[User, RpcUser]]] = None,
) -> Any:
    type = NotificationSettingTypes(notification_setting.type)
    value = NotificationSettingOptionValues(notification_setting.value)
    scope_type = NotificationScopeType(notification_setting.scope_type)
    key = get_legacy_key(type, scope_type)

    data = {
        "key": key,
        "value": get_legacy_value(type, value),
        "user": user_mapping.get(notification_setting.user_id) if user_mapping else None,
        "project_id": None,
        "organization_id": None,
    }

    if scope_type == NotificationScopeType.PROJECT:
        data["project_id"] = notification_setting.scope_identifier
    if scope_type == NotificationScopeType.ORGANIZATION:
        data["organization_id"] = notification_setting.scope_identifier

    return LegacyUserOptionClone(**data)


def map_notification_settings_to_legacy(
    notification_settings: Iterable[Any],
    user_mapping: Mapping[int, Union[User, RpcUser]],
) -> list[Any]:
    """A hack for legacy serializers. Pretend a list of NotificationSettings is a list of UserOptions."""
    return [
        get_legacy_object(notification_setting, user_mapping)
        for notification_setting in notification_settings
    ]


def get_parent_mappings(
    notification_settings: Iterable[Any],
) -> tuple[Mapping[int, Any], Mapping[int, Any]]:
    """Prefetch a list of Project or Organization objects for the Serializer."""
    from sentry.models.organization import Organization
    from sentry.models.project import Project

    project_ids = []
    organization_ids = []
    for notification_setting in notification_settings:
        if notification_setting.scope_type == NotificationScopeType.PROJECT.value:
            project_ids.append(notification_setting.scope_identifier)
        if notification_setting.scope_type == NotificationScopeType.ORGANIZATION.value:
            organization_ids.append(notification_setting.scope_identifier)

    projects = Project.objects.filter(id__in=project_ids)
    organizations = Organization.objects.filter(id__in=organization_ids)

    project_mapping = {project.id: project for project in projects}
    organization_mapping = {organization.id: organization for organization in organizations}

    return project_mapping, organization_mapping
