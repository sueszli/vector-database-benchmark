""" DialogFlow CX: webhook to log session ID for each request."""
import re
import functions_framework

@functions_framework.http
def log_session_id_for_troubleshooting(request):
    if False:
        for i in range(10):
            print('nop')
    'Webhook will log session id corresponding to request.'
    req = request.get_json()
    session_id_regex = '.+\\/sessions\\/(.+)'
    session = req['sessionInfo']['session']
    regex_match = re.search(session_id_regex, session)
    session_id = regex_match.group(1)
    print(f'Debug Node: session ID = {session_id}')
    res = {'fulfillment_response': {'messages': [{'text': {'text': [f'Request Session ID: {session_id}']}}]}}
    return res