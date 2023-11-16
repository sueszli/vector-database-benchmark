"""
Module for sending messages to Slack

.. versionadded:: 2015.5.0

:configuration: This module can be used by either passing an api key and version
    directly or by specifying both in a configuration profile in the salt
    master/minion config.

    For example:

    .. code-block:: yaml

        slack:
          api_key: peWcBiMOS9HrZG15peWcBiMOS9HrZG15
"""
import logging
import urllib.parse
import salt.utils.json
import salt.utils.slack
from salt.exceptions import SaltInvocationError
log = logging.getLogger(__name__)
__virtualname__ = 'slack'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Return virtual name of the module.\n\n    :return: The virtual name of the module.\n    '
    return __virtualname__

def _get_api_key():
    if False:
        print('Hello World!')
    api_key = __salt__['config.get']('slack.api_key') or __salt__['config.get']('slack:api_key')
    if not api_key:
        raise SaltInvocationError('No Slack API key found.')
    return api_key

def _get_hook_id():
    if False:
        while True:
            i = 10
    url = __salt__['config.get']('slack.hook') or __salt__['config.get']('slack:hook')
    if not url:
        raise SaltInvocationError('No Slack WebHook url found')
    return url

def list_rooms(api_key=None):
    if False:
        while True:
            i = 10
    "\n    List all Slack rooms.\n\n    :param api_key: The Slack admin api key.\n    :return: The room list.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' slack.list_rooms\n\n        salt '*' slack.list_rooms api_key=peWcBiMOS9HrZG15peWcBiMOS9HrZG15\n    "
    if not api_key:
        api_key = _get_api_key()
    return salt.utils.slack.query(function='rooms', api_key=api_key, opts=__opts__)

def list_users(api_key=None):
    if False:
        i = 10
        return i + 15
    "\n    List all Slack users.\n\n    :param api_key: The Slack admin api key.\n    :return: The user list.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' slack.list_users\n\n        salt '*' slack.list_users api_key=peWcBiMOS9HrZG15peWcBiMOS9HrZG15\n    "
    if not api_key:
        api_key = _get_api_key()
    return salt.utils.slack.query(function='users', api_key=api_key, opts=__opts__)

def find_room(name, api_key=None):
    if False:
        print('Hello World!')
    '\n    Find a room by name and return it.\n\n    :param name:    The room name.\n    :param api_key: The Slack admin api key.\n    :return:        The room object.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' slack.find_room name="random"\n\n        salt \'*\' slack.find_room name="random" api_key=peWcBiMOS9HrZG15peWcBiMOS9HrZG15\n    '
    if not api_key:
        api_key = _get_api_key()
    if name.startswith('#'):
        name = name[1:]
    ret = list_rooms(api_key)
    if ret['res']:
        rooms = ret['message']
        if rooms:
            for room in rooms:
                if room['name'] == name:
                    return room
    return False

def find_user(name, api_key=None):
    if False:
        i = 10
        return i + 15
    '\n    Find a user by name and return it.\n\n    :param name:        The user name.\n    :param api_key:     The Slack admin api key.\n    :return:            The user object.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' slack.find_user name="ThomasHatch"\n\n        salt \'*\' slack.find_user name="ThomasHatch" api_key=peWcBiMOS9HrZG15peWcBiMOS9HrZG15\n    '
    if not api_key:
        api_key = _get_api_key()
    ret = list_users(api_key)
    if ret['res']:
        users = ret['message']
        if users:
            for user in users:
                if user['name'] == name:
                    return user
    return False

def post_message(channel, message, from_name, api_key=None, icon=None, attachments=None, blocks=None):
    if False:
        i = 10
        return i + 15
    '\n    Send a message to a Slack channel.\n\n    .. versionchanged:: 3003\n        Added `attachments` and `blocks` kwargs\n\n    :param channel:     The channel name, either will work.\n    :param message:     The message to send to the Slack channel.\n    :param from_name:   Specify who the message is from.\n    :param api_key:     The Slack api key, if not specified in the configuration.\n    :param icon:        URL to an image to use as the icon for this message\n    :param attachments: Any attachments to be sent with the message.\n    :param blocks:      Any blocks to be sent with the message.\n    :return:            Boolean if message was sent successfully.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' slack.post_message channel="Development Room" message="Build is done" from_name="Build Server"\n\n    '
    if not api_key:
        api_key = _get_api_key()
    if not channel:
        log.error('channel is a required option.')
    if not channel.startswith('#') and (not channel.startswith('@')):
        log.warning('Channel name must start with a hash or @. Prepending a hash and using "#%s" as channel name instead of %s', channel, channel)
        channel = '#{}'.format(channel)
    if not from_name:
        log.error('from_name is a required option.')
    if not message:
        log.error('message is a required option.')
    if not from_name:
        log.error('from_name is a required option.')
    parameters = {'channel': channel, 'username': from_name, 'text': message, 'attachments': attachments or [], 'blocks': blocks or []}
    if icon is not None:
        parameters['icon_url'] = icon
    result = salt.utils.slack.query(function='message', api_key=api_key, method='POST', header_dict={'Content-Type': 'application/x-www-form-urlencoded'}, data=urllib.parse.urlencode(parameters), opts=__opts__)
    if result['res']:
        return True
    else:
        return result

def call_hook(message, attachment=None, color='good', short=False, identifier=None, channel=None, username=None, icon_emoji=None):
    if False:
        return 10
    "\n    Send message to Slack incoming webhook.\n\n    :param message:     The topic of message.\n    :param attachment:  The message to send to the Slack WebHook.\n    :param color:       The color of border of left side\n    :param short:       An optional flag indicating whether the value is short\n                        enough to be displayed side-by-side with other values.\n    :param identifier:  The identifier of WebHook.\n    :param channel:     The channel to use instead of the WebHook default.\n    :param username:    Username to use instead of WebHook default.\n    :param icon_emoji:  Icon to use instead of WebHook default.\n    :return:            Boolean if message was sent successfully.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' slack.call_hook message='Hello, from SaltStack'\n\n    "
    base_url = 'https://hooks.slack.com/services/'
    if not identifier:
        identifier = _get_hook_id()
    url = urllib.parse.urljoin(base_url, identifier)
    if not message:
        log.error('message is required option')
    if attachment:
        payload = {'attachments': [{'fallback': message, 'color': color, 'pretext': message, 'fields': [{'value': attachment, 'short': short}]}]}
    else:
        payload = {'text': message}
    if channel:
        payload['channel'] = channel
    if username:
        payload['username'] = username
    if icon_emoji:
        payload['icon_emoji'] = icon_emoji
    data = urllib.parse.urlencode({'payload': salt.utils.json.dumps(payload)})
    result = salt.utils.http.query(url, method='POST', data=data, status=True)
    if result['status'] <= 201:
        return True
    else:
        return {'res': False, 'message': result.get('body', result['status'])}