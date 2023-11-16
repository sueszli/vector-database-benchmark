import logging
logging.basicConfig(level=logging.DEBUG)
import os
from slack_bolt.app import App
from slack_bolt.context import BoltContext
bot_token = os.environ.get('SLACK_SDK_TEST_SOCKET_MODE_BOT_TOKEN')
app = App(signing_secret='will-be-removed-soon', token=bot_token)

@app.event('app_mention')
def mention(context: BoltContext):
    if False:
        print('Hello World!')
    context.say(':wave: Hi there!')

@app.event('message')
def message(context: BoltContext, event: dict):
    if False:
        while True:
            i = 10
    context.client.reactions_add(channel=event['channel'], timestamp=event['ts'], name='eyes')

@app.command('/hello-socket-mode')
def hello_command(ack, body):
    if False:
        return 10
    user_id = body['user_id']
    ack(f'Hi <@{user_id}>!')
if __name__ == '__main__':
    from bolt_adapter.builtin import SocketModeHandler
    app_token = os.environ.get('SLACK_SDK_TEST_SOCKET_MODE_APP_TOKEN')
    SocketModeHandler(app, app_token).start()