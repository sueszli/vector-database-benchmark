"""
Module for sending messages to MS Teams

.. versionadded:: 2017.7.0

:configuration: This module can be used by either passing a hook_url
    directly or by specifying it in a configuration profile in the salt
    master/minion config. For example:

.. code-block:: yaml

    msteams:
      hook_url: https://outlook.office.com/webhook/837
"""
import logging
import salt.utils.json
from salt.exceptions import SaltInvocationError
log = logging.getLogger(__name__)
__virtualname__ = 'msteams'

def __virtual__():
    if False:
        return 10
    '\n    Return virtual name of the module.\n    :return: The virtual name of the module.\n    '
    return __virtualname__

def _get_hook_url():
    if False:
        return 10
    '\n    Return hook_url from minion/master config file\n    or from pillar\n    '
    hook_url = __salt__['config.get']('msteams.hook_url') or __salt__['config.get']('msteams:hook_url')
    if not hook_url:
        raise SaltInvocationError('No MS Teams hook_url found.')
    return hook_url

def post_card(message, hook_url=None, title=None, theme_color=None):
    if False:
        return 10
    '\n    Send a message to an MS Teams channel.\n    :param message:     The message to send to the MS Teams channel.\n    :param hook_url:    The Teams webhook URL, if not specified in the configuration.\n    :param title:       Optional title for the posted card\n    :param theme_color:  Optional hex color highlight for the posted card\n    :return:            Boolean if message was sent successfully.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' msteams.post_card message="Build is done"\n    '
    if not hook_url:
        hook_url = _get_hook_url()
    if not message:
        log.error('message is a required option.')
    payload = {'text': message, 'title': title, 'themeColor': theme_color}
    headers = {'Content-Type': 'application/json'}
    result = salt.utils.http.query(hook_url, method='POST', header_dict=headers, data=salt.utils.json.dumps(payload), status=True)
    if result['status'] <= 201:
        return True
    else:
        return {'res': False, 'message': result.get('body', result['status'])}