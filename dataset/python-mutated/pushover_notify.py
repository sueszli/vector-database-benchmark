"""
Module for sending messages to Pushover (https://www.pushover.net)

.. versionadded:: 2016.3.0

:configuration: This module can be used by either passing an api key and version
    directly or by specifying both in a configuration profile in the salt
    master/minion config.

    For example:

    .. code-block:: yaml

        pushover:
          token: abAHuZyCLtdH8P4zhmFZmgUHUsv1ei8
"""
import logging
import urllib.parse
import salt.utils.pushover
from salt.exceptions import SaltInvocationError
log = logging.getLogger(__name__)
__virtualname__ = 'pushover'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Return virtual name of the module.\n\n    :return: The virtual name of the module.\n    '
    return __virtualname__

def post_message(user=None, device=None, message=None, title=None, priority=None, expire=None, retry=None, sound=None, api_version=1, token=None):
    if False:
        return 10
    "\n    Send a message to a Pushover user or group.\n\n    :param user:        The user or group to send to, must be key of user or group not email address.\n    :param message:     The message to send to the PushOver user or group.\n    :param title:       Specify who the message is from.\n    :param priority:    The priority of the message, defaults to 0.\n    :param expire:      The message should expire after N number of seconds.\n    :param retry:       The number of times the message should be retried.\n    :param sound:       The sound to associate with the message.\n    :param api_version: The PushOver API version, if not specified in the configuration.\n    :param token:       The PushOver token, if not specified in the configuration.\n    :return:            Boolean if message was sent successfully.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pushover.post_message user='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' title='Message from Salt' message='Build is done'\n\n        salt '*' pushover.post_message user='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' title='Message from Salt' message='Build is done' priority='2' expire='720' retry='5'\n\n    "
    if not token:
        token = __salt__['config.get']('pushover.token') or __salt__['config.get']('pushover:token')
        if not token:
            raise SaltInvocationError('Pushover token is unavailable.')
    if not user:
        user = __salt__['config.get']('pushover.user') or __salt__['config.get']('pushover:user')
        if not user:
            raise SaltInvocationError('Pushover user key is unavailable.')
    if not message:
        raise SaltInvocationError('Required parameter "message" is missing.')
    user_validate = salt.utils.pushover.validate_user(user, device, token)
    if not user_validate['result']:
        return user_validate
    if not title:
        title = 'Message from SaltStack'
    parameters = dict()
    parameters['user'] = user
    parameters['device'] = device
    parameters['token'] = token
    parameters['title'] = title
    parameters['priority'] = priority
    parameters['expire'] = expire
    parameters['retry'] = retry
    parameters['message'] = message
    if sound and salt.utils.pushover.validate_sound(sound, token)['res']:
        parameters['sound'] = sound
    result = salt.utils.pushover.query(function='message', method='POST', header_dict={'Content-Type': 'application/x-www-form-urlencoded'}, data=urllib.parse.urlencode(parameters), opts=__opts__)
    if result['res']:
        return True
    else:
        return result