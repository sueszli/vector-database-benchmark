import asyncio
import logging
logging.basicConfig(level=logging.DEBUG)
from flask import Flask, make_response
app = Flask(__name__)
logger = logging.getLogger(__name__)
import os
from slack_sdk.web import WebClient
from slack_sdk.errors import SlackApiError
singleton_client = WebClient(token=os.environ['SLACK_BOT_TOKEN'], run_async=False)
singleton_loop = asyncio.new_event_loop()
singleton_async_client = WebClient(token=os.environ['SLACK_BOT_TOKEN'], run_async=True, loop=singleton_loop)

@app.route('/sync/singleton', methods=['GET'])
def singleton():
    if False:
        for i in range(10):
            print('nop')
    try:
        response = singleton_client.chat_postMessage(channel='#random', text='You used the singleton WebClient for posting this message!')
        return str(response)
    except SlackApiError as e:
        return make_response(str(e), 400)

@app.route('/sync/per-request', methods=['GET'])
def per_request():
    if False:
        print('Hello World!')
    try:
        client = WebClient(token=os.environ['SLACK_BOT_TOKEN'], run_async=False)
        response = client.chat_postMessage(channel='#random', text='You used a new WebClient for posting this message!')
        return str(response)
    except SlackApiError as e:
        return make_response(str(e), 400)

@app.route('/async/singleton', methods=['GET'])
def singleton_async():
    if False:
        i = 10
        return i + 15
    try:
        future = singleton_async_client.chat_postMessage(channel='#random', text='You used the singleton WebClient for posting this message!')
        response = singleton_loop.run_until_complete(future)
        return str(response)
    except SlackApiError as e:
        return make_response(str(e), 400)

@app.route('/async/per-request', methods=['GET'])
def per_request_async():
    if False:
        i = 10
        return i + 15
    try:
        loop_for_this_request = asyncio.new_event_loop()
        async_client = WebClient(token=os.environ['SLACK_BOT_TOKEN'], run_async=True, loop=loop_for_this_request)
        future = async_client.chat_postMessage(channel='#random', text='You used the singleton WebClient for posting this message!')
        response = loop_for_this_request.run_until_complete(future)
        return str(response)
    except SlackApiError as e:
        return make_response(str(e), 400)
if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=3000)