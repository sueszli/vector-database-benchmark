import logging
logging.basicConfig(level=logging.DEBUG)
import os
from slack_bolt.app import App
from slack_bolt.context import BoltContext
from slack_bolt.oauth.oauth_settings import OAuthSettings
app = App(signing_secret=os.environ['SLACK_SIGNING_SECRET'], oauth_settings=OAuthSettings(client_id=os.environ['SLACK_CLIENT_ID'], client_secret=os.environ['SLACK_CLIENT_SECRET'], scopes=os.environ['SLACK_SCOPES'].split(',')))

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
        print('Hello World!')
    user_id = body['user_id']
    ack(f'Hi <@{user_id}>!')
if __name__ == '__main__':
    from bolt_adapter.websocket_client import SocketModeHandler
    app_token = os.environ.get('SLACK_APP_TOKEN')
    SocketModeHandler(app, app_token).connect()
    app.start()