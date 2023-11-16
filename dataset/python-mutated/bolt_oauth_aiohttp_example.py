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
        return 10
    context.say(':wave: Hi there!')

@app.event('message')
def message(context: BoltContext, event: dict):
    if False:
        i = 10
        return i + 15
    context.client.reactions_add(channel=event['channel'], timestamp=event['ts'], name='eyes')

@app.command('/hello-socket-mode')
def hello_command(ack, body):
    if False:
        return 10
    user_id = body['user_id']
    ack(f'Hi <@{user_id}>!')
if __name__ == '__main__':

    def run_socket_mode_app():
        if False:
            for i in range(10):
                print('nop')
        import asyncio
        from bolt_adapter.aiohttp import AsyncSocketModeHandler

        async def socket_mode_app():
            app_token = os.environ.get('SLACK_APP_TOKEN')
            await AsyncSocketModeHandler(app, app_token).connect_async()
            await asyncio.sleep(float('inf'))
        asyncio.run(socket_mode_app())
    from concurrent.futures.thread import ThreadPoolExecutor
    socket_mode_thread = ThreadPoolExecutor(1)
    socket_mode_thread.submit(run_socket_mode_app)
    app.start()