"""Send slack notifications"""
import os
import sys
import requests

def send_notification():
    if False:
        i = 10
        return i + 15
    'Create a slack message'
    webhook = os.getenv('SLACK_WEBHOOK')
    if not webhook:
        raise Exception('Unable to retrieve SLACK_WEBHOOK')
    nightly_slack_messages = {'tag': 'to create a tag', 'python': 'on python tests', 'js': 'on javascript tests', 'py_prod': 'on python prod dependencies test', 'cypress': 'on cypress tests', 'playwright': 'on playwright tests', 'build': 'to release'}
    run_id = os.getenv('RUN_ID')
    workflow = sys.argv[1]
    message_key = sys.argv[2]
    payload = None
    if workflow == 'nightly':
        failure = nightly_slack_messages[message_key]
        payload = {'text': f':blobonfire: Nightly build failed {failure} - <https://github.com/streamlit/streamlit/actions/runs/{run_id}|Link to run>'}
    if workflow == 'candidate':
        if message_key == 'success':
            payload = {'text': ':rocket: Release Candidate was successful!'}
        else:
            payload = {'text': f':blobonfire: Release Candidate failed - <https://github.com/streamlit/streamlit/actions/runs/{run_id}|Link to run>'}
    if workflow == 'release':
        if message_key == 'success':
            payload = {'text': ':rocket: Release was successful!'}
        else:
            payload = {'text': f':blobonfire: Release failed - <https://github.com/streamlit/streamlit/actions/runs/{run_id}|Link to run>'}
    if payload:
        response = requests.post(webhook, json=payload)
        if response.status_code != 200:
            raise Exception(f'Unable to send slack message, HTTP response: {response.text}')

def main():
    if False:
        while True:
            i = 10
    send_notification()
if __name__ == '__main__':
    main()