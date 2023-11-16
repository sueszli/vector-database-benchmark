"""
Module for sending messages to google chat.

.. versionadded:: 2019.2.0

To use this module you need to configure a webhook in the google chat room
where you would like the message to be sent, see:

    https://developers.google.com/hangouts/chat/how-tos/webhooks
"""
import json
__virtualname__ = 'google_chat'

def __virtual__():
    if False:
        i = 10
        return i + 15
    return __virtualname__

def send_message(url, message):
    if False:
        return 10
    '\n    Send a message to the google chat room specified in the webhook url.\n\n    .. code-block:: bash\n\n        salt \'*\' google_chat.send_message "https://chat.googleapis.com/v1/spaces/example_space/messages?key=example_key" "This is a test message"\n    '
    headers = {'Content-Type': 'application/json'}
    data = {'text': message}
    result = __utils__['http.query'](url, 'POST', data=json.dumps(data), header_dict=headers, decode=True, status=True)
    if result.get('status', 0) == 200:
        return True
    else:
        return False