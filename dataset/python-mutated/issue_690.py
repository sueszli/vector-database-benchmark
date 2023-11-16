import logging
logging.basicConfig(level=logging.DEBUG)
import os
from flask import Flask, make_response, request
app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.route('/slack/oauth/callback', methods=['GET'])
def endpoint():
    if False:
        return 10
    code = request.args.get('code')
    from slack_sdk.web import WebClient
    from slack_sdk.errors import SlackApiError
    try:
        client = WebClient(token='')
        client_id = os.environ['SLACK_CLIENT_ID']
        client_secret = os.environ['SLACK_CLIENT_SECRET']
        response = client.oauth_v2_access(client_id=client_id, client_secret=client_secret, code=code)
        result = response.get('error', 'success!')
        return str(result)
    except SlackApiError as e:
        return make_response(str(e), 400)
if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=3000)