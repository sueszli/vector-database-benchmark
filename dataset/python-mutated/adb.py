"""
Beacon to emit adb device state changes for Android devices

.. versionadded:: 2016.3.0
"""
import logging
import salt.utils.beacons
import salt.utils.path
log = logging.getLogger(__name__)
__virtualname__ = 'adb'
last_state = {}
last_state_extra = {'value': False, 'no_devices': False}

def __virtual__():
    if False:
        print('Hello World!')
    which_result = salt.utils.path.which('adb')
    if which_result is None:
        err_msg = 'adb is missing.'
        log.error('Unable to load %s beacon: %s', __virtualname__, err_msg)
        return (False, err_msg)
    else:
        return __virtualname__

def validate(config):
    if False:
        i = 10
        return i + 15
    '\n    Validate the beacon configuration\n    '
    if not isinstance(config, list):
        log.info('Configuration for adb beacon must be a list.')
        return (False, 'Configuration for adb beacon must be a list.')
    config = salt.utils.beacons.list_to_dict(config)
    if 'states' not in config:
        log.info('Configuration for adb beacon must include a states array.')
        return (False, 'Configuration for adb beacon must include a states array.')
    elif not isinstance(config['states'], list):
        log.info('Configuration for adb beacon must include a states array.')
        return (False, 'Configuration for adb beacon must include a states array.')
    else:
        states = ['offline', 'bootloader', 'device', 'host', 'recovery', 'no permissions', 'sideload', 'unauthorized', 'unknown', 'missing']
        if any((s not in states for s in config['states'])):
            log.info('Need a one of the following adb states: %s', ', '.join(states))
            return (False, 'Need a one of the following adb states: {}'.format(', '.join(states)))
    return (True, 'Valid beacon configuration')

def beacon(config):
    if False:
        i = 10
        return i + 15
    '\n    Emit the status of all devices returned by adb\n\n    Specify the device states that should emit an event,\n    there will be an event for each device with the\n    event type and device specified.\n\n    .. code-block:: yaml\n\n        beacons:\n          adb:\n            - states:\n                - offline\n                - unauthorized\n                - missing\n            - no_devices_event: True\n            - battery_low: 25\n\n    '
    log.trace('adb beacon starting')
    ret = []
    config = salt.utils.beacons.list_to_dict(config)
    out = __salt__['cmd.run']('adb devices', runas=config.get('user', None))
    lines = out.split('\n')[1:]
    last_state_devices = list(last_state.keys())
    found_devices = []
    for line in lines:
        try:
            (device, state) = line.split('\t')
            found_devices.append(device)
            if device not in last_state_devices or ('state' in last_state[device] and last_state[device]['state'] != state):
                if state in config['states']:
                    ret.append({'device': device, 'state': state, 'tag': state})
                    last_state[device] = {'state': state}
            if 'battery_low' in config:
                val = last_state.get(device, {})
                cmd = 'adb -s {} shell cat /sys/class/power_supply/*/capacity'.format(device)
                battery_levels = __salt__['cmd.run'](cmd, runas=config.get('user', None)).split('\n')
                for l in battery_levels:
                    battery_level = int(l)
                    if 0 < battery_level < 100:
                        if 'battery' not in val or battery_level != val['battery']:
                            if ('battery' not in val or val['battery'] > config['battery_low']) and battery_level <= config['battery_low']:
                                ret.append({'device': device, 'battery_level': battery_level, 'tag': 'battery_low'})
                        if device not in last_state:
                            last_state[device] = {}
                        last_state[device].update({'battery': battery_level})
        except ValueError:
            continue
    for device in last_state_devices:
        if device not in found_devices:
            if 'missing' in config['states']:
                ret.append({'device': device, 'state': 'missing', 'tag': 'missing'})
            del last_state[device]
    if 'no_devices_event' in config and config['no_devices_event'] is True:
        if not found_devices and (not last_state_extra['no_devices']):
            ret.append({'tag': 'no_devices'})
    last_state_extra['no_devices'] = not found_devices
    return ret