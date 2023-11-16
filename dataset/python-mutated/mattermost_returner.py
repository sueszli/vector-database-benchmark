"""
Return salt data via mattermost

.. versionadded:: 2017.7.0

The following fields can be set in the minion conf file:

.. code-block:: yaml

    mattermost.hook (required)
    mattermost.username (optional)
    mattermost.channel (optional)

Alternative configuration values can be used by prefacing the configuration.
Any values not found in the alternative configuration will be pulled from
the default location:

.. code-block:: yaml

    mattermost.channel
    mattermost.hook
    mattermost.username

mattermost settings may also be configured as:

.. code-block:: yaml

    mattermost:
      channel: RoomName
      hook: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      username: user

To use the mattermost returner, append '--return mattermost' to the salt command.

.. code-block:: bash

    salt '*' test.ping --return mattermost

To override individual configuration items, append --return_kwargs '{'key:': 'value'}' to the salt command.

.. code-block:: bash

    salt '*' test.ping --return mattermost --return_kwargs '{'channel': '#random'}'
"""
import logging
import salt.returners
import salt.utils.json
import salt.utils.mattermost
log = logging.getLogger(__name__)
__virtualname__ = 'mattermost'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return virtual name of the module.\n\n    :return: The virtual name of the module.\n    '
    return __virtualname__

def _get_options(ret=None):
    if False:
        return 10
    '\n    Get the mattermost options from salt.\n    '
    attrs = {'channel': 'channel', 'username': 'username', 'hook': 'hook', 'api_url': 'api_url'}
    _options = salt.returners.get_returner_options(__virtualname__, ret, attrs, __salt__=__salt__, __opts__=__opts__)
    log.debug('Options: %s', _options)
    return _options

def returner(ret):
    if False:
        while True:
            i = 10
    '\n    Send an mattermost message with the data\n    '
    _options = _get_options(ret)
    api_url = _options.get('api_url')
    channel = _options.get('channel')
    username = _options.get('username')
    hook = _options.get('hook')
    if not hook:
        log.error('mattermost.hook not defined in salt config')
        return
    returns = ret.get('return')
    message = 'id: {}\r\nfunction: {}\r\nfunction args: {}\r\njid: {}\r\nreturn: {}\r\n'.format(ret.get('id'), ret.get('fun'), ret.get('fun_args'), ret.get('jid'), returns)
    mattermost = post_message(channel, message, username, api_url, hook)
    return mattermost

def event_return(events):
    if False:
        for i in range(10):
            print('nop')
    '\n    Send the events to a mattermost room.\n\n    :param events:      List of events\n    :return:            Boolean if messages were sent successfully.\n    '
    _options = _get_options()
    api_url = _options.get('api_url')
    channel = _options.get('channel')
    username = _options.get('username')
    hook = _options.get('hook')
    is_ok = True
    for event in events:
        log.debug('Event: %s', event)
        log.debug('Event data: %s', event['data'])
        message = 'tag: {}\r\n'.format(event['tag'])
        for (key, value) in event['data'].items():
            message += '{}: {}\r\n'.format(key, value)
        result = post_message(channel, message, username, api_url, hook)
        if not result:
            is_ok = False
    return is_ok

def post_message(channel, message, username, api_url, hook):
    if False:
        i = 10
        return i + 15
    '\n    Send a message to a mattermost room.\n\n    :param channel:     The room name.\n    :param message:     The message to send to the mattermost room.\n    :param username:    Specify who the message is from.\n    :param hook:        The mattermost hook, if not specified in the configuration.\n    :return:            Boolean if message was sent successfully.\n    '
    parameters = dict()
    if channel:
        parameters['channel'] = channel
    if username:
        parameters['username'] = username
    parameters['text'] = '```' + message + '```'
    log.debug('Parameters: %s', parameters)
    result = salt.utils.mattermost.query(api_url=api_url, hook=hook, data='payload={}'.format(salt.utils.json.dumps(parameters)))
    log.debug('result %s', result)
    return bool(result)