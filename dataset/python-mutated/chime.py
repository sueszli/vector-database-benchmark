from __future__ import annotations
from functools import cached_property
from typing import TYPE_CHECKING
from airflow.exceptions import AirflowOptionalProviderFeatureException
from airflow.providers.amazon.aws.hooks.chime import ChimeWebhookHook
if TYPE_CHECKING:
    from airflow.utils.context import Context
try:
    from airflow.notifications.basenotifier import BaseNotifier
except ImportError:
    raise AirflowOptionalProviderFeatureException('Failed to import BaseNotifier. This feature is only available in Airflow versions >= 2.6.0')

class ChimeNotifier(BaseNotifier):
    """
    Chime notifier to send messages to a chime room via callbacks.

    :param chime_conn_id: The chime connection to use with Endpoint as "https://hooks.chime.aws" and
        the webhook token in the form of ```{webhook.id}?token{webhook.token}```
    :param message: The message to send to the chime room associated with the webhook.

    """
    template_fields = ('message',)

    def __init__(self, *, chime_conn_id: str, message: str='This is the default chime notifier message'):
        if False:
            while True:
                i = 10
        super().__init__()
        self.chime_conn_id = chime_conn_id
        self.message = message

    @cached_property
    def hook(self):
        if False:
            i = 10
            return i + 15
        'To reduce overhead cache the hook for the notifier.'
        return ChimeWebhookHook(chime_conn_id=self.chime_conn_id)

    def notify(self, context: Context) -> None:
        if False:
            while True:
                i = 10
        'Send a message to a Chime Chat Room.'
        self.hook.send_message(message=self.message)
send_chime_notification = ChimeNotifier