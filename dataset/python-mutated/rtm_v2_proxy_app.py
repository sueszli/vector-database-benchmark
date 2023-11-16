import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(filename)s (%(lineno)s): %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
import os
from slack_sdk.rtm.v2 import RTMClient
from integration_tests.env_variable_names import SLACK_SDK_TEST_CLASSIC_APP_BOT_TOKEN
proxy_url = 'http://localhost:9000'
if __name__ == '__main__':
    rtm = RTMClient(token=os.environ.get(SLACK_SDK_TEST_CLASSIC_APP_BOT_TOKEN), trace_enabled=True, all_message_trace_enabled=True, proxy=proxy_url)

    @rtm.on('message')
    def handle(client: RTMClient, event: dict):
        if False:
            for i in range(10):
                print('nop')
        client.web_client.reactions_add(channel=event['channel'], timestamp=event['ts'], name='eyes')

    @rtm.on('*')
    def handle(client: RTMClient, event: dict):
        if False:
            print('Hello World!')
        logger.info(event)
    rtm.start()