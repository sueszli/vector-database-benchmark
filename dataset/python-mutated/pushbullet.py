"""
Module for sending messages to Pushbullet (https://www.pushbullet.com)

.. versionadded:: 2015.8.0

Requires an ``api_key`` in ``/etc/salt/minion``:

.. code-block:: yaml

    pushbullet:
      api_key: 'ABC123abc123ABC123abc123ABC123ab'

For example:

.. code-block:: yaml

    pushbullet:
      device: "Chrome"
      title: "Example push message"
      body: "Message body."

"""
import logging
try:
    import pushbullet
    HAS_PUSHBULLET = True
except ImportError:
    HAS_PUSHBULLET = False
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        print('Hello World!')
    if not HAS_PUSHBULLET:
        return (False, 'Missing pushbullet library.')
    if not __salt__['config.get']('pushbullet.api_key') and (not __salt__['config.get']('pushbullet:api_key')):
        return (False, 'Pushbullet API Key Unavailable, not loading.')
    return True

class _SaltPushbullet:

    def __init__(self, device_name):
        if False:
            print('Hello World!')
        api_key = __salt__['config.get']('pushbullet.api_key') or __salt__['config.get']('pushbullet:api_key')
        self.pb = pushbullet.Pushbullet(api_key)
        self.target = self._find_device_by_name(device_name)

    def push_note(self, title, body):
        if False:
            while True:
                i = 10
        push = self.pb.push_note(title, body, device=self.target)
        return push

    def _find_device_by_name(self, name):
        if False:
            return 10
        for dev in self.pb.devices:
            if dev.nickname == name:
                return dev

def push_note(device=None, title=None, body=None):
    if False:
        while True:
            i = 10
    '\n    Pushing a text note.\n\n    :param device:   Pushbullet target device\n    :param title:    Note title\n    :param body:     Note body\n\n    :return:            Boolean if message was sent successfully.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt "*" pushbullet.push_note device="Chrome" title="Example title" body="Example body."\n    '
    spb = _SaltPushbullet(device)
    res = spb.push_note(title, body)
    return res