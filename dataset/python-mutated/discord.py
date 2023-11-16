from __future__ import annotations
from functools import cached_property
from airflow.exceptions import AirflowOptionalProviderFeatureException
try:
    from airflow.notifications.basenotifier import BaseNotifier
except ImportError:
    raise AirflowOptionalProviderFeatureException('Failed to import BaseNotifier. This feature is only available in Airflow versions >= 2.6.0')
from airflow.providers.discord.hooks.discord_webhook import DiscordWebhookHook
ICON_URL: str = 'https://raw.githubusercontent.com/apache/airflow/main/airflow/www/static/pin_100.png'

class DiscordNotifier(BaseNotifier):
    """
    Discord BaseNotifier.

    :param discord_conn_id: Http connection ID with host as "https://discord.com/api/" and
                         default webhook endpoint in the extra field in the form of
                         {"webhook_endpoint": "webhooks/{webhook.id}/{webhook.token}"}
    :param text: The content of the message
    :param username: The username to send the message as. Optional
    :param avatar_url: The URL of the avatar to use for the message. Optional
    :param tts: Text to speech.
    """
    template_fields = ('discord_conn_id', 'text', 'username', 'avatar_url', 'tts')

    def __init__(self, discord_conn_id: str='discord_webhook_default', text: str='This is a default message', username: str='Airflow', avatar_url: str=ICON_URL, tts: bool=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.discord_conn_id = discord_conn_id
        self.text = text
        self.username = username
        self.avatar_url = avatar_url
        self.tts = tts

    @cached_property
    def hook(self) -> DiscordWebhookHook:
        if False:
            for i in range(10):
                print('nop')
        'Discord Webhook Hook.'
        return DiscordWebhookHook(http_conn_id=self.discord_conn_id)

    def notify(self, context):
        if False:
            return 10
        'Send a message to a Discord channel.'
        self.hook.username = self.username
        self.hook.message = self.text
        self.hook.avatar_url = self.avatar_url
        self.hook.tts = self.tts
        self.hook.execute()