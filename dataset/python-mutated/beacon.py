"""
Management of the Salt beacons
==============================

.. versionadded:: 2015.8.0

.. code-block:: yaml

    ps:
      beacon.present:
        - save: True
        - enable: False
        - services:
            salt-master: running
            apache2: stopped

    sh:
      beacon.present: []

    load:
      beacon.present:
        - averages:
            1m:
              - 0.0
              - 2.0
            5m:
              - 0.0
              - 1.5
            15m:
              - 0.1
              - 1.0

    .. versionadded:: 3000

    Beginning in the 3000 release, multiple copies of a beacon can be configured
    using the ``beacon_module`` parameter.

    inotify_infs:
      beacon.present:
        - save: True
        - enable: True
        - files:
           /etc/infs.conf:
             mask:
               - create
               - delete
               - modify
             recurse: True
             auto_add: True
        - interval: 10
        - beacon_module: inotify
        - disable_during_state_run: True

    inotify_ntp:
      beacon.present:
        - save: True
        - enable: True
        - files:
           /etc/ntp.conf:
             mask:
               - create
               - delete
               - modify
             recurse: True
             auto_add: True
        - interval: 10
        - beacon_module: inotify
        - disable_during_state_run: True
"""
import logging
log = logging.getLogger(__name__)

def present(name, save=False, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Ensure beacon is configured with the included beacon data.\n\n    name\n        The name of the beacon to ensure is configured.\n    save\n        True/False, if True the beacons.conf file be updated too. Default is False.\n\n    Example:\n\n    .. code-block:: yaml\n\n        ps_beacon:\n          beacon.present:\n            - name: ps\n            - save: True\n            - enable: False\n            - services:\n                salt-master: running\n                apache2: stopped\n    '
    ret = {'name': name, 'result': True, 'changes': {}, 'comment': []}
    current_beacons = __salt__['beacons.list'](return_yaml=False, **kwargs)
    beacon_data = [{k: v} for (k, v) in kwargs.items()]
    if name in current_beacons:
        if beacon_data == current_beacons[name]:
            ret['comment'].append('Job {} in correct state'.format(name))
        elif __opts__.get('test'):
            kwargs['test'] = True
            result = __salt__['beacons.modify'](name, beacon_data, **kwargs)
            ret['comment'].append(result['comment'])
            ret['changes'] = result['changes']
        else:
            result = __salt__['beacons.modify'](name, beacon_data, **kwargs)
            if not result['result']:
                ret['result'] = result['result']
                ret['comment'] = result['comment']
                return ret
            elif 'changes' in result:
                ret['comment'].append('Modifying {} in beacons'.format(name))
                ret['changes'] = result['changes']
            else:
                ret['comment'].append(result['comment'])
    elif __opts__.get('test'):
        kwargs['test'] = True
        result = __salt__['beacons.add'](name, beacon_data, **kwargs)
        ret['comment'].append(result['comment'])
    else:
        result = __salt__['beacons.add'](name, beacon_data, **kwargs)
        if not result['result']:
            ret['result'] = result['result']
            ret['comment'] = result['comment']
            return ret
        else:
            ret['comment'].append('Adding {} to beacons'.format(name))
    if save:
        if __opts__.get('test'):
            ret['comment'].append('Beacon {} would be saved'.format(name))
        else:
            __salt__['beacons.save']()
            ret['comment'].append('Beacon {} saved'.format(name))
    ret['comment'] = '\n'.join(ret['comment'])
    return ret

def absent(name, save=False, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Ensure beacon is absent.\n\n    name\n        The name of the beacon that is ensured absent.\n    save\n        True/False, if True the beacons.conf file be updated too. Default is False.\n\n    Example:\n\n    .. code-block:: yaml\n\n        remove_beacon:\n          beacon.absent:\n            - name: ps\n            - save: True\n\n    '
    ret = {'name': name, 'result': True, 'changes': {}, 'comment': []}
    current_beacons = __salt__['beacons.list'](return_yaml=False, **kwargs)
    if name in current_beacons:
        if __opts__.get('test'):
            kwargs['test'] = True
            result = __salt__['beacons.delete'](name, **kwargs)
            ret['comment'].append(result['comment'])
        else:
            result = __salt__['beacons.delete'](name, **kwargs)
            if not result['result']:
                ret['result'] = result['result']
                ret['comment'] = result['comment']
                return ret
            else:
                ret['comment'].append('Removed {} from beacons'.format(name))
    else:
        ret['comment'].append('{} not configured in beacons'.format(name))
    if save:
        if __opts__.get('test'):
            ret['comment'].append('Beacon {} would be saved'.format(name))
        else:
            __salt__['beacons.save']()
            ret['comment'].append('Beacon {} saved'.format(name))
    ret['comment'] = '\n'.join(ret['comment'])
    return ret

def enabled(name, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Enable a beacon.\n\n    name\n        The name of the beacon to enable.\n\n    Example:\n\n    .. code-block:: yaml\n\n        enable_beacon:\n          beacon.enabled:\n            - name: ps\n\n    '
    ret = {'name': name, 'result': True, 'changes': {}, 'comment': []}
    current_beacons = __salt__['beacons.list'](return_yaml=False, **kwargs)
    if name in current_beacons:
        if __opts__.get('test'):
            kwargs['test'] = True
            result = __salt__['beacons.enable_beacon'](name, **kwargs)
            ret['comment'].append(result['comment'])
        else:
            result = __salt__['beacons.enable_beacon'](name, **kwargs)
            if not result['result']:
                ret['result'] = result['result']
                ret['comment'] = result['comment']
                return ret
            else:
                ret['comment'].append('Enabled {} from beacons'.format(name))
    else:
        ret['comment'].append('{} not a configured beacon'.format(name))
    ret['comment'] = '\n'.join(ret['comment'])
    return ret

def disabled(name, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Disable a beacon.\n\n    name\n        The name of the beacon to disable.\n\n    Example:\n\n    .. code-block:: yaml\n\n        disable_beacon:\n          beacon.disabled:\n            - name: psp\n\n    '
    ret = {'name': name, 'result': True, 'changes': {}, 'comment': []}
    current_beacons = __salt__['beacons.list'](return_yaml=False, **kwargs)
    if name in current_beacons:
        if __opts__.get('test'):
            kwargs['test'] = True
            result = __salt__['beacons.disable_beacon'](name, **kwargs)
            ret['comment'].append(result['comment'])
        else:
            result = __salt__['beacons.disable_beacon'](name, **kwargs)
            if not result['result']:
                ret['result'] = result['result']
                ret['comment'] = result['comment']
                return ret
            else:
                ret['comment'].append('Disabled beacon {}.'.format(name))
    else:
        ret['comment'].append('Job {} is not configured.'.format(name))
    ret['comment'] = '\n'.join(ret['comment'])
    return ret