"""Hook for Telegram."""
from __future__ import annotations
import asyncio
import telegram
import tenacity
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook

class TelegramHook(BaseHook):
    """
    This hook allows you to post messages to Telegram using the telegram python-telegram-bot library.

    The library can be found here: https://github.com/python-telegram-bot/python-telegram-bot
    It accepts both telegram bot API token directly or connection that has telegram bot API token.
    If both supplied, token parameter will be given precedence, otherwise 'password' field in the connection
    from telegram_conn_id will be used.
    chat_id can also be provided in the connection using 'host' field in connection.
    Following is the details of a telegram_connection:
    name: 'telegram-connection-name'
    conn_type: 'http'
    password: 'TELEGRAM_TOKEN'
    host: 'chat_id' (optional)
    Examples:
    .. code-block:: python

        # Create hook
        telegram_hook = TelegramHook(telegram_conn_id="telegram_default")
        # or telegram_hook = TelegramHook(telegram_conn_id='telegram_default', chat_id='-1xxx')
        # or telegram_hook = TelegramHook(token='xxx:xxx', chat_id='-1xxx')

        # Call method from telegram bot client
        telegram_hook.send_message(None, {"text": "message", "chat_id": "-1xxx"})
        # or telegram_hook.send_message(None', {"text": "message"})

    :param telegram_conn_id: connection that optionally has Telegram API token in the password field
    :param token: optional telegram API token
    :param chat_id: optional chat_id of the telegram chat/channel/group
    """

    def __init__(self, telegram_conn_id: str | None=None, token: str | None=None, chat_id: str | None=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.token = self.__get_token(token, telegram_conn_id)
        self.chat_id = self.__get_chat_id(chat_id, telegram_conn_id)
        self.connection = self.get_conn()

    def get_conn(self) -> telegram.Bot:
        if False:
            return 10
        '\n        Returns the telegram bot client.\n\n        :return: telegram bot client\n        '
        return telegram.Bot(self.token)

    def __get_token(self, token: str | None, telegram_conn_id: str | None) -> str:
        if False:
            print('Hello World!')
        '\n        Returns the telegram API token.\n\n        :param token: telegram API token\n        :param telegram_conn_id: telegram connection name\n        :return: telegram API token\n        '
        if token is not None:
            return token
        if telegram_conn_id is not None:
            conn = self.get_connection(telegram_conn_id)
            if not conn.password:
                raise AirflowException('Missing token(password) in Telegram connection')
            return conn.password
        raise AirflowException('Cannot get token: No valid Telegram connection supplied.')

    def __get_chat_id(self, chat_id: str | None, telegram_conn_id: str | None) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the telegram chat ID for a chat/channel/group.\n\n        :param chat_id: optional chat ID\n        :param telegram_conn_id: telegram connection name\n        :return: telegram chat ID\n        '
        if chat_id is not None:
            return chat_id
        if telegram_conn_id is not None:
            conn = self.get_connection(telegram_conn_id)
            return conn.host
        return None

    @tenacity.retry(retry=tenacity.retry_if_exception_type(telegram.error.TelegramError), stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_fixed(1))
    def send_message(self, api_params: dict) -> None:
        if False:
            return 10
        '\n        Sends the message to a telegram channel or chat.\n\n        :param api_params: params for telegram_instance.send_message. It can also be used to override chat_id\n        '
        kwargs = {'chat_id': self.chat_id, 'parse_mode': telegram.constants.ParseMode.HTML, 'disable_web_page_preview': True}
        kwargs.update(api_params)
        if 'text' not in kwargs or kwargs['text'] is None:
            raise AirflowException("'text' must be provided for telegram message")
        if kwargs['chat_id'] is None:
            raise AirflowException("'chat_id' must be provided for telegram message")
        response = asyncio.run(self.connection.send_message(**kwargs))
        self.log.debug(response)