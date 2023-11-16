"""
Example beacon to use with salt-proxy

.. code-block:: yaml

    beacons:
      proxy_example:
        endpoint: beacon
"""
import logging
import salt.utils.beacons
import salt.utils.http
__proxyenabled__ = ['*']
__virtualname__ = 'proxy_example'
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Trivially let the beacon load for the test example.\n    For a production beacon we should probably have some expression here.\n    '
    return True

def validate(config):
    if False:
        for i in range(10):
            print('nop')
    '\n    Validate the beacon configuration\n    '
    if not isinstance(config, list):
        return (False, 'Configuration for proxy_example beacon must be a list.')
    return (True, 'Valid beacon configuration')

def beacon(config):
    if False:
        print('Hello World!')
    '\n    Called several times each second\n    https://docs.saltproject.io/en/latest/topics/beacons/#the-beacon-function\n\n    .. code-block:: yaml\n\n        beacons:\n          proxy_example:\n            - endpoint: beacon\n    '
    config = salt.utils.beacons.list_to_dict(config)
    beacon_url = '{}{}'.format(__opts__['proxy']['url'], config['endpoint'])
    ret = salt.utils.http.query(beacon_url, decode_type='json', decode=True)
    return [ret['dict']]