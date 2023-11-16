import logging
logging.basicConfig(level=logging.DEBUG)
import os
from slack_sdk.rtm import RTMClient
from slack_sdk.errors import SlackApiError

@RTMClient.run_on(event='message')
def say_hello(**payload):
    if False:
        print('Hello World!')
    data = payload['data']
    web_client = payload['web_client']
    rtm_client = payload['rtm_client']
    if 'text' in data and 'Hello' in data.get('text', []):
        channel_id = data['channel']
        thread_ts = data['ts']
        user = data['user']
        try:
            response = web_client.chat_postMessage(channel=channel_id, text=f'Hi <@{user}>!', thread_ts=thread_ts)
        except SlackApiError as e:
            assert e.response['ok'] is False
            assert e.response['error']
            print(f"Got an error: {e.response['error']}")
rtm_client = RTMClient(token=os.environ['SLACK_API_TOKEN'])
rtm_client.start()