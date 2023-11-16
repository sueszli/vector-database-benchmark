import logging
logging.basicConfig(level=logging.DEBUG)
import os
import time
from slack_sdk.web import WebClient
from slack_sdk.errors import SlackApiError
client = WebClient(token=os.environ['SLACK_API_TOKEN'])

def send_slack_message(channel, message):
    if False:
        for i in range(10):
            print('nop')
    return client.chat_postMessage(channel=channel, text=message)
channel = '#random'
message = 'Hello, from Python!'
while True:
    try:
        response = send_slack_message(channel, message)
    except SlackApiError as e:
        if e.response['error'] == 'ratelimited':
            delay = int(e.response.headers['Retry-After'])
            print(f'Rate limited. Retrying in {delay} seconds')
            time.sleep(delay)
            response = send_slack_message(channel, message)
        else:
            raise e