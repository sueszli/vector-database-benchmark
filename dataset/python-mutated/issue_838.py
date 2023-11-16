import json
import logging
logging.basicConfig(level=logging.DEBUG)
import os
from slack_sdk.web import WebClient
from slack_sdk.signature import SignatureVerifier
app_token_client = WebClient(token=os.environ['SLACK_APP_TOKEN'])
signature_verifier = SignatureVerifier(os.environ['SLACK_SIGNING_SECRET'])
from flask import Flask, request, make_response
app = Flask(__name__)

@app.route('/slack/events', methods=['POST'])
def slack_app():
    if False:
        return 10
    request_body = request.get_data()
    if not signature_verifier.is_valid_request(request_body, request.headers):
        return make_response('invalid request', 403)
    if request.headers['content-type'] == 'application/json':
        body = json.loads(request_body)
        response = app_token_client.apps_event_authorizations_list(event_context=body['event_context'])
        print(response)
        return make_response('', 200)
    return make_response('', 404)
if __name__ == '__main__':
    app.run('localhost', 3000)