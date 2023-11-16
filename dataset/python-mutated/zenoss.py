"""
Module for working with the Zenoss API

.. versionadded:: 2016.3.0

:configuration: This module requires a 'zenoss' entry in the master/minion config.

    For example:

    .. code-block:: yaml

        zenoss:
          hostname: https://zenoss.example.com
          username: admin
          password: admin123
          verify_ssl: True
          ca_bundle: /etc/ssl/certs/ca-certificates.crt
"""
import logging
import re
import salt.utils.http
import salt.utils.json
log = logging.getLogger(__name__)
__virtualname__ = 'zenoss'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if requests is installed\n    '
    return __virtualname__
ROUTERS = {'MessagingRouter': 'messaging', 'EventsRouter': 'evconsole', 'ProcessRouter': 'process', 'ServiceRouter': 'service', 'DeviceRouter': 'device', 'NetworkRouter': 'network', 'TemplateRouter': 'template', 'DetailNavRouter': 'detailnav', 'ReportRouter': 'report', 'MibRouter': 'mib', 'ZenPackRouter': 'zenpack'}

def _session():
    if False:
        while True:
            i = 10
    '\n    Create a session to be used when connecting to Zenoss.\n    '
    config = __salt__['config.option']('zenoss')
    return salt.utils.http.session(user=config.get('username'), password=config.get('password'), verify_ssl=config.get('verify_ssl', True), ca_bundle=config.get('ca_bundle'), headers={'Content-type': 'application/json; charset=utf-8'})

def _router_request(router, method, data=None):
    if False:
        i = 10
        return i + 15
    '\n    Make a request to the Zenoss API router\n    '
    if router not in ROUTERS:
        return False
    req_data = salt.utils.json.dumps([dict(action=router, method=method, data=data, type='rpc', tid=1)])
    config = __salt__['config.option']('zenoss')
    log.debug('Making request to router %s with method %s', router, method)
    url = '{}/zport/dmd/{}_router'.format(config.get('hostname'), ROUTERS[router])
    response = _session().post(url, data=req_data)
    if re.search('name="__ac_name"', response.content):
        log.error('Request failed. Bad username/password.')
        raise Exception('Request failed. Bad username/password.')
    return salt.utils.json.loads(response.content).get('result', None)

def _determine_device_class():
    if False:
        while True:
            i = 10
    '\n    If no device class is given when adding a device, this helps determine\n    '
    if __salt__['grains.get']('kernel') == 'Linux':
        return '/Server/Linux'

def find_device(device=None):
    if False:
        while True:
            i = 10
    "\n    Find a device in Zenoss. If device not found, returns None.\n\n    Parameters:\n        device:         (Optional) Will use the grain 'fqdn' by default\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zenoss.find_device\n    "
    data = [{'uid': '/zport/dmd/Devices', 'params': {}, 'limit': None}]
    all_devices = _router_request('DeviceRouter', 'getDevices', data=data)
    for dev in all_devices['devices']:
        if dev['name'] == device:
            dev['hash'] = all_devices['hash']
            log.info('Found device %s in Zenoss', device)
            return dev
    log.info('Unable to find device %s in Zenoss', device)
    return None

def device_exists(device=None):
    if False:
        while True:
            i = 10
    "\n    Check to see if a device already exists in Zenoss.\n\n    Parameters:\n        device:         (Optional) Will use the grain 'fqdn' by default\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zenoss.device_exists\n    "
    if not device:
        device = __salt__['grains.get']('fqdn')
    if find_device(device):
        return True
    return False

def add_device(device=None, device_class=None, collector='localhost', prod_state=1000):
    if False:
        while True:
            i = 10
    "\n    A function to connect to a zenoss server and add a new device entry.\n\n    Parameters:\n        device:         (Optional) Will use the grain 'fqdn' by default.\n        device_class:   (Optional) The device class to use. If none, will determine based on kernel grain.\n        collector:      (Optional) The collector to use for this device. Defaults to 'localhost'.\n        prod_state:     (Optional) The prodState to set on the device. If none, defaults to 1000 ( production )\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zenoss.add_device\n    "
    if not device:
        device = __salt__['grains.get']('fqdn')
    if not device_class:
        device_class = _determine_device_class()
    log.info('Adding device %s to zenoss', device)
    data = dict(deviceName=device, deviceClass=device_class, model=True, collector=collector, productionState=prod_state)
    response = _router_request('DeviceRouter', 'addDevice', data=[data])
    return response

def set_prod_state(prod_state, device=None):
    if False:
        return 10
    "\n    A function to set the prod_state in zenoss.\n\n    Parameters:\n        prod_state:     (Required) Integer value of the state\n        device:         (Optional) Will use the grain 'fqdn' by default.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt zenoss.set_prod_state 1000 hostname\n    "
    if not device:
        device = __salt__['grains.get']('fqdn')
    device_object = find_device(device)
    if not device_object:
        return 'Unable to find a device in Zenoss for {}'.format(device)
    log.info('Setting prodState to %d on %s device', prod_state, device)
    data = dict(uids=[device_object['uid']], prodState=prod_state, hashcheck=device_object['hash'])
    return _router_request('DeviceRouter', 'setProductionState', [data])