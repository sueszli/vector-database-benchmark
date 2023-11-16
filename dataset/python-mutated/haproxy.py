"""
Watch current connections of haproxy server backends.
Fire an event when over a specified threshold.

.. versionadded:: 2016.11.0
"""
import logging
import salt.utils.beacons
log = logging.getLogger(__name__)
__virtualname__ = 'haproxy'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load the module if haproxyctl module is installed\n    '
    if 'haproxy.get_sessions' in __salt__:
        return __virtualname__
    else:
        err_msg = 'haproxy.get_sessions is missing.'
        log.error('Unable to load %s beacon: %s', __virtualname__, err_msg)
        return (False, err_msg)

def validate(config):
    if False:
        while True:
            i = 10
    '\n    Validate the beacon configuration\n    '
    if not isinstance(config, list):
        return (False, 'Configuration for haproxy beacon must be a list.')
    else:
        config = salt.utils.beacons.list_to_dict(config)
        if 'backends' not in config:
            return (False, 'Configuration for haproxy beacon requires backends.')
        elif not isinstance(config['backends'], dict):
            return (False, 'Backends for haproxy beacon must be a dictionary.')
        else:
            for backend in config['backends']:
                log.debug('config %s', config['backends'][backend])
                if 'servers' not in config['backends'][backend]:
                    return (False, 'Backends for haproxy beacon require servers.')
                else:
                    _servers = config['backends'][backend]['servers']
                    if not isinstance(_servers, list):
                        return (False, 'Servers for haproxy beacon must be a list.')
    return (True, 'Valid beacon configuration')

def beacon(config):
    if False:
        return 10
    '\n    Check if current number of sessions of a server for a specific haproxy backend\n    is over a defined threshold.\n\n    .. code-block:: yaml\n\n        beacons:\n          haproxy:\n            - backends:\n                www-backend:\n                    threshold: 45\n                    servers:\n                      - web1\n                      - web2\n            - interval: 120\n    '
    ret = []
    config = salt.utils.beacons.list_to_dict(config)
    for backend in config.get('backends', ()):
        backend_config = config['backends'][backend]
        threshold = backend_config['threshold']
        for server in backend_config['servers']:
            scur = __salt__['haproxy.get_sessions'](server, backend)
            if scur:
                if int(scur) > int(threshold):
                    _server = {'server': server, 'scur': scur, 'threshold': threshold}
                    log.debug('Emit because %s > %s for %s in %s', scur, threshold, server, backend)
                    ret.append(_server)
    return ret