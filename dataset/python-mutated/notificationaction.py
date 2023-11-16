from __future__ import annotations
import logging
from abc import ABCMeta, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, MutableMapping, Optional, TypeVar
from django.db import models
from sentry.backup.scopes import RelocationScope
from sentry.db.models import FlexibleForeignKey, Model, sane_repr
from sentry.db.models.base import region_silo_only_model
from sentry.db.models.fields.hybrid_cloud_foreign_key import HybridCloudForeignKey
from sentry.models.organization import Organization
from sentry.services.hybrid_cloud.integration import RpcIntegration
from sentry.types.integrations import ExternalProviders
from sentry.utils.json import JSONData
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from sentry.api.serializers.rest_framework.notification_action import NotificationActionInputData

class FlexibleIntEnum(IntEnum):

    @classmethod
    def as_choices(cls) -> tuple[tuple[int, str], ...]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @classmethod
    def get_name(cls, value: int) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        return dict(cls.as_choices()).get(value)

    @classmethod
    def get_value(cls, name: str) -> int | None:
        if False:
            print('Hello World!')
        invert_choices = {v: k for (k, v) in cls.as_choices()}
        return invert_choices.get(name)

class ActionService(FlexibleIntEnum):
    """
    The available services to fire action notifications
    """
    EMAIL = 0
    PAGERDUTY = 1
    SLACK = 2
    MSTEAMS = 3
    SENTRY_APP = 4
    SENTRY_NOTIFICATION = 5
    OPSGENIE = 6
    DISCORD = 7

    @classmethod
    def as_choices(cls) -> tuple[tuple[int, str], ...]:
        if False:
            print('Hello World!')
        assert ExternalProviders.EMAIL.name is not None
        assert ExternalProviders.PAGERDUTY.name is not None
        assert ExternalProviders.SLACK.name is not None
        assert ExternalProviders.MSTEAMS.name is not None
        assert ExternalProviders.OPSGENIE.name is not None
        assert ExternalProviders.DISCORD.name is not None
        return ((cls.EMAIL.value, ExternalProviders.EMAIL.name), (cls.PAGERDUTY.value, ExternalProviders.PAGERDUTY.name), (cls.SLACK.value, ExternalProviders.SLACK.name), (cls.MSTEAMS.value, ExternalProviders.MSTEAMS.name), (cls.SENTRY_APP.value, 'sentry_app'), (cls.SENTRY_NOTIFICATION.value, 'sentry_notification'), (cls.OPSGENIE.value, ExternalProviders.OPSGENIE.name), (cls.DISCORD.value, ExternalProviders.DISCORD.name))

class ActionTarget(FlexibleIntEnum):
    """
    Explains the contents of target_identifier
    """
    SPECIFIC = 0
    USER = 1
    TEAM = 2
    SENTRY_APP = 3

    @classmethod
    def as_choices(cls) -> tuple[tuple[int, str], ...]:
        if False:
            return 10
        return ((cls.SPECIFIC.value, 'specific'), (cls.USER.value, 'user'), (cls.TEAM.value, 'team'), (cls.SENTRY_APP.value, 'sentry_app'))

class AbstractNotificationAction(Model):
    """
    Abstract model meant to retroactively create a contract for notification actions
    (e.g. metric alerts, spike protection, etc.)
    """
    integration_id = HybridCloudForeignKey('sentry.Integration', blank=True, null=True, on_delete='CASCADE')
    sentry_app_id = HybridCloudForeignKey('sentry.SentryApp', blank=True, null=True, on_delete='CASCADE')
    type = models.SmallIntegerField(choices=ActionService.as_choices())
    target_type = models.SmallIntegerField(choices=ActionTarget.as_choices())
    target_identifier = models.TextField(null=True)
    target_display = models.TextField(null=True)

    @property
    def service_type(self) -> int:
        if False:
            print('Hello World!')
        '\n        Used for disambiguity of self.type\n        '
        return self.type

    class Meta:
        abstract = True

class ActionTrigger(FlexibleIntEnum):
    """
    The possible sources of action notifications.
    Use values less than 100 here to avoid conflicts with getsentry's trigger values.
    """
    AUDIT_LOG = 0

    @classmethod
    def as_choices(cls) -> tuple[tuple[int, str], ...]:
        if False:
            i = 10
            return i + 15
        return ((cls.AUDIT_LOG.value, 'audit-log'),)

class TriggerGenerator:
    """
    Allows NotificationAction.trigger_type to enforce extra triggers via
    NotificationAction.register_trigger_type
    """

    def __iter__(self):
        if False:
            return 10
        yield from NotificationAction._trigger_types

@region_silo_only_model
class NotificationActionProject(Model):
    __relocation_scope__ = {RelocationScope.Global, RelocationScope.Organization}
    project = FlexibleForeignKey('sentry.Project')
    action = FlexibleForeignKey('sentry.NotificationAction')

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_notificationactionproject'

    def get_relocation_scope(self) -> RelocationScope:
        if False:
            print('Hello World!')
        action = NotificationAction.objects.get(id=self.action_id)
        return action.get_relocation_scope()

class ActionRegistration(metaclass=ABCMeta):

    def __init__(self, action: NotificationAction):
        if False:
            for i in range(10):
                print('nop')
        self.action = action

    @abstractmethod
    def fire(self, data: Any) -> None:
        if False:
            while True:
                i = 10
        '\n        Handles delivering the message via the service from the action and specified data.\n        '
        pass

    @classmethod
    def validate_action(cls, data: NotificationActionInputData) -> None:
        if False:
            while True:
                i = 10
        '\n        Optional function to provide increased validation when saving incoming NotificationActions. See NotificationActionSerializer.\n\n        :param data: The input data sent to the API before updating/creating NotificationActions\n        :raises serializers.ValidationError: Indicates that the incoming action would apply to this registration but is not valid.\n        '
        pass

    @classmethod
    def serialize_available(cls, organization: Organization, integrations: Optional[List[RpcIntegration]]=None) -> List[JSONData]:
        if False:
            return 10
        "\n        Optional class method to serialize this registration's available actions to an organization. See NotificationActionsAvailableEndpoint.\n\n        :param organization: The relevant organization which will receive the serialized available action in their response.\n        :param integrations: A list of integrations which are set up for the organization.\n        "
        return []
ActionRegistrationT = TypeVar('ActionRegistrationT', bound=ActionRegistration)

@region_silo_only_model
class NotificationAction(AbstractNotificationAction):
    """
    Generic notification action model to programmatically route depending on the trigger (or source) for the notification
    """
    __relocation_scope__ = {RelocationScope.Global, RelocationScope.Organization}
    __repr__ = sane_repr('id', 'trigger_type', 'service_type', 'target_display')
    _trigger_types: tuple[tuple[int, str], ...] = ActionTrigger.as_choices()
    _registry: MutableMapping[str, type[ActionRegistration]] = {}
    organization = FlexibleForeignKey('sentry.Organization')
    projects = models.ManyToManyField('sentry.Project', through=NotificationActionProject)
    trigger_type = models.SmallIntegerField(choices=TriggerGenerator())

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_notificationaction'

    @classmethod
    def register_trigger_type(cls, value: int, display_text: str) -> None:
        if False:
            return 10
        '\n        This method is used for adding trigger types to this model from getsentry.\n        If the trigger is relevant to sentry as well, directly modify ActionTrigger.\n        '
        cls._trigger_types += ((value, display_text),)

    @classmethod
    def register_action(cls, trigger_type: int, service_type: int, target_type: int):
        if False:
            print('Hello World!')
        '\n        Register a new trigger/service/target combination for NotificationActions.\n        For example, allowing audit-logs (trigger) to fire actions to slack (service) channels (target)\n\n        :param trigger_type: The registered trigger_type integer value saved to the database\n        :param service_type: The service_type integer value which must exist on ActionService\n        :param target_type: The target_type integer value which must exist on ActionTarget\n        :param registration: A subclass of `ActionRegistration`.\n        '

        def inner(registration: type[ActionRegistrationT]) -> type[ActionRegistrationT]:
            if False:
                return 10
            if trigger_type not in dict(cls._trigger_types):
                raise AttributeError(f'Trigger type of {trigger_type} is not registered. Modify ActionTrigger or call register_trigger_type().')
            if service_type not in dict(ActionService.as_choices()):
                raise AttributeError(f'Service type of {service_type} is not registered. Modify ActionService.')
            if target_type not in dict(ActionTarget.as_choices()):
                raise AttributeError(f'Target type of {target_type} is not registered. Modify ActionTarget.')
            key = cls.get_registry_key(trigger_type, service_type, target_type)
            if cls._registry.get(key) is not None:
                raise AttributeError(f'Existing registration found for trigger:{trigger_type}, service:{service_type}, target:{target_type}.')
            cls._registry[key] = registration
            return registration
        return inner

    @classmethod
    def get_trigger_types(cls):
        if False:
            while True:
                i = 10
        return cls._trigger_types

    @classmethod
    def get_trigger_text(self, trigger_type: int) -> str:
        if False:
            print('Hello World!')
        return dict(NotificationAction.get_trigger_types())[trigger_type]

    @classmethod
    def get_registry_key(self, trigger_type: int, service_type: int, target_type: int) -> str:
        if False:
            print('Hello World!')
        return f'{trigger_type}:{service_type}:{target_type}'

    @classmethod
    def get_registry(cls) -> Mapping[str, type[ActionRegistration]]:
        if False:
            for i in range(10):
                print('nop')
        return cls._registry

    @classmethod
    def get_registration(cls, trigger_type: int, service_type: int, target_type: int) -> type[ActionRegistration] | None:
        if False:
            while True:
                i = 10
        key = cls.get_registry_key(trigger_type, service_type, target_type)
        return cls._registry.get(key)

    def get_audit_log_data(self) -> Dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns audit log data for NOTIFICATION_ACTION_ADD, NOTIFICATION_ACTION_EDIT\n        and NOTIFICATION_ACTION_REMOVE events\n        '
        return {'trigger': NotificationAction.get_trigger_text(self.trigger_type)}

    def fire(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        registration = NotificationAction.get_registration(self.trigger_type, self.service_type, self.target_type)
        if registration:
            logger.info('fire_action', extra={'action_id': self.id, 'trigger': NotificationAction.get_trigger_text(self.trigger_type), 'service': ActionService.get_name(self.service_type), 'target': ActionTarget.get_name(self.target_type)})
            return registration(action=self).fire(*args, **kwargs)
        else:
            logger.error('missing_registration', extra={'id': self.id, 'service_type': self.service_type, 'trigger_type': self.trigger_type, 'target_type': self.target_type})

    def get_relocation_scope(self) -> RelocationScope:
        if False:
            while True:
                i = 10
        if self.integration_id is not None or self.sentry_app_id is not None:
            return RelocationScope.Global
        return RelocationScope.Organization