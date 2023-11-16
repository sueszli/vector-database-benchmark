import logging
logging.basicConfig(level=logging.DEBUG)
import os
from threading import Event
from slack_sdk.web import WebClient
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode import SocketModeClient
proxy_url = 'http://user:pass%2Fword@localhost:9000'
client = SocketModeClient(app_token=os.environ.get('SLACK_SDK_TEST_SOCKET_MODE_APP_TOKEN'), web_client=WebClient(token=os.environ.get('SLACK_SDK_TEST_SOCKET_MODE_BOT_TOKEN'), proxy=proxy_url), proxy=proxy_url, trace_enabled=True, all_message_trace_enabled=True)
if __name__ == '__main__':

    def process(client: SocketModeClient, req: SocketModeRequest):
        if False:
            i = 10
            return i + 15
        if req.type == 'events_api':
            response = SocketModeResponse(envelope_id=req.envelope_id)
            client.send_socket_mode_response(response)
            client.web_client.reactions_add(name='eyes', channel=req.payload['event']['channel'], timestamp=req.payload['event']['ts'])
    client.socket_mode_request_listeners.append(process)
    client.connect()
    Event().wait()