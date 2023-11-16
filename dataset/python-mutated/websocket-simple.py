"""Process individual messages from a WebSocket connection."""
import logging
import re
from mitmproxy import http

def websocket_message(flow: http.HTTPFlow):
    if False:
        while True:
            i = 10
    assert flow.websocket is not None
    message = flow.websocket.messages[-1]
    if message.from_client:
        logging.info(f'Client sent a message: {message.content!r}')
    else:
        logging.info(f'Server sent a message: {message.content!r}')
    message.content = re.sub(b'^Hello', b'HAPPY', message.content)
    if b'FOOBAR' in message.content:
        message.drop()