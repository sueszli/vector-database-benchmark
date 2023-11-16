from __future__ import annotations
import json
import re
from typing import Any
from airflow.exceptions import AirflowException
from airflow.providers.http.hooks.http import HttpHook

class DiscordWebhookHook(HttpHook):
    """
    This hook allows you to post messages to Discord using incoming webhooks.

    Takes a Discord connection ID with a default relative webhook endpoint. The
    default endpoint can be overridden using the webhook_endpoint parameter
    (https://discordapp.com/developers/docs/resources/webhook).

    Each Discord webhook can be pre-configured to use a specific username and
    avatar_url. You can override these defaults in this hook.

    :param http_conn_id: Http connection ID with host as "https://discord.com/api/" and
                         default webhook endpoint in the extra field in the form of
                         {"webhook_endpoint": "webhooks/{webhook.id}/{webhook.token}"}
    :param webhook_endpoint: Discord webhook endpoint in the form of
                             "webhooks/{webhook.id}/{webhook.token}"
    :param message: The message you want to send to your Discord channel
                    (max 2000 characters)
    :param username: Override the default username of the webhook
    :param avatar_url: Override the default avatar of the webhook
    :param tts: Is a text-to-speech message
    :param proxy: Proxy to use to make the Discord webhook call
    """
    conn_name_attr = 'http_conn_id'
    default_conn_name = 'discord_default'
    conn_type = 'discord'
    hook_name = 'Discord'

    def __init__(self, http_conn_id: str | None=None, webhook_endpoint: str | None=None, message: str='', username: str | None=None, avatar_url: str | None=None, tts: bool=False, proxy: str | None=None, *args: Any, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.http_conn_id: Any = http_conn_id
        self.webhook_endpoint = self._get_webhook_endpoint(http_conn_id, webhook_endpoint)
        self.message = message
        self.username = username
        self.avatar_url = avatar_url
        self.tts = tts
        self.proxy = proxy

    def _get_webhook_endpoint(self, http_conn_id: str | None, webhook_endpoint: str | None) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the default webhook endpoint or override if a webhook_endpoint is manually supplied.\n\n        :param http_conn_id: The provided connection ID\n        :param webhook_endpoint: The manually provided webhook endpoint\n        :return: Webhook endpoint (str) to use\n        '
        if webhook_endpoint:
            endpoint = webhook_endpoint
        elif http_conn_id:
            conn = self.get_connection(http_conn_id)
            extra = conn.extra_dejson
            endpoint = extra.get('webhook_endpoint', '')
        else:
            raise AirflowException('Cannot get webhook endpoint: No valid Discord webhook endpoint or http_conn_id supplied.')
        if not re.fullmatch('webhooks/[0-9]+/[a-zA-Z0-9_-]+', endpoint):
            raise AirflowException('Expected Discord webhook endpoint in the form of "webhooks/{webhook.id}/{webhook.token}".')
        return endpoint

    def _build_discord_payload(self) -> str:
        if False:
            print('Hello World!')
        '\n        Combine all relevant parameters into a valid Discord JSON payload.\n\n        :return: Discord payload (str) to send\n        '
        payload: dict[str, Any] = {}
        if self.username:
            payload['username'] = self.username
        if self.avatar_url:
            payload['avatar_url'] = self.avatar_url
        payload['tts'] = self.tts
        if len(self.message) <= 2000:
            payload['content'] = self.message
        else:
            raise AirflowException('Discord message length must be 2000 or fewer characters.')
        return json.dumps(payload)

    def execute(self) -> None:
        if False:
            print('Hello World!')
        'Execute the Discord webhook call.'
        proxies = {}
        if self.proxy:
            proxies = {'https': self.proxy}
        discord_payload = self._build_discord_payload()
        self.run(endpoint=self.webhook_endpoint, data=discord_payload, headers={'Content-type': 'application/json'}, extra_options={'proxies': proxies})