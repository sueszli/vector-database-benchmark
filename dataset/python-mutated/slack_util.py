"""
DEPRECATION NOTICE: this module is deprecated and will be removed on 2.0.
"""
import logging
from io import IOBase
from typing import cast, Optional, Union
import backoff
from flask import current_app
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.web.slack_response import SlackResponse
logger = logging.getLogger('tasks.slack_util')

@backoff.on_exception(backoff.expo, SlackApiError, factor=10, base=2, max_tries=5)
def deliver_slack_msg(slack_channel: str, subject: str, body: str, file: Optional[Union[str, IOBase, bytes]]) -> None:
    if False:
        for i in range(10):
            print('nop')
    config = current_app.config
    token = config['SLACK_API_TOKEN']
    if callable(token):
        token = token()
    client = WebClient(token=token, proxy=config['SLACK_PROXY'])
    if file:
        response = cast(SlackResponse, client.files_upload(channels=slack_channel, file=file, initial_comment=body, title=subject))
        assert response['file'], str(response)
    else:
        response = cast(SlackResponse, client.chat_postMessage(channel=slack_channel, text=body))
        assert response['message']['text'], str(response)
    logger.info('Sent the report to the slack %s', slack_channel)