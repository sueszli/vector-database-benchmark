import logging
logging.basicConfig(level=logging.DEBUG)
import os
from slack_sdk.rtm import RTMClient
logger = logging.getLogger(__name__)
global_state = {}

@RTMClient.run_on(event='open')
def open(**payload):
    if False:
        print('Hello World!')
    web_client = payload['web_client']
    auth_result = web_client.auth_test()
    global_state.update({'bot_id': auth_result['bot_id']})
    logger.info(f'cached: {global_state}')

@RTMClient.run_on(event='message')
def message(**payload):
    if False:
        return 10
    data = payload['data']
    if data.get('bot_id', None) == global_state['bot_id']:
        logger.debug("Skipped as it's me")
        return
    web_client = payload['web_client']
    message = web_client.chat_postMessage(channel=data['channel'], text="What's up?")
    logger.info(f"message: {message['ts']}")
rtm_client = RTMClient(token=os.environ['SLACK_API_TOKEN'])
rtm_client.start()