"""
Send events from Docker events
:Depends:   Docker API >= 1.22
"""
import logging
import traceback
import salt.utils.event
import salt.utils.json
try:
    import docker
    import docker.utils
    HAS_DOCKER_PY = True
except ImportError:
    HAS_DOCKER_PY = False
log = logging.getLogger(__name__)
CLIENT_TIMEOUT = 60
__virtualname__ = 'docker_events'
__deprecated__ = (3009, 'docker', 'https://github.com/saltstack/saltext-docker')

def __virtual__():
    if False:
        return 10
    '\n    Only load if docker libs are present\n    '
    if not HAS_DOCKER_PY:
        return (False, 'Docker_events engine could not be imported')
    return True

def start(docker_url='unix://var/run/docker.sock', timeout=CLIENT_TIMEOUT, tag='salt/engines/docker_events', filters=None):
    if False:
        while True:
            i = 10
    '\n    Scan for Docker events and fire events\n\n    Example Config\n\n    .. code-block:: yaml\n\n        engines:\n          - docker_events:\n              docker_url: unix://var/run/docker.sock\n              filters:\n                event:\n                - start\n                - stop\n                - die\n                - oom\n\n    The config above sets up engines to listen\n    for events from the Docker daemon and publish\n    them to the Salt event bus.\n\n    For filter reference, see https://docs.docker.com/engine/reference/commandline/events/\n    '
    if __opts__.get('__role') == 'master':
        fire_master = salt.utils.event.get_master_event(__opts__, __opts__['sock_dir']).fire_event
    else:
        fire_master = None

    def fire(tag, msg):
        if False:
            for i in range(10):
                print('nop')
        '\n        How to fire the event\n        '
        if fire_master:
            fire_master(msg, tag)
        else:
            __salt__['event.send'](tag, msg)
    try:
        client = docker.APIClient(base_url=docker_url, timeout=timeout)
    except AttributeError:
        client = docker.Client(base_url=docker_url, timeout=timeout)
    try:
        events = client.events(filters=filters)
        for event in events:
            data = salt.utils.json.loads(event.decode(__salt_system_encoding__, errors='replace'))
            if data['Action']:
                fire('{}/{}'.format(tag, data['Action']), data)
            else:
                fire('{}/{}'.format(tag, data['status']), data)
    except Exception:
        traceback.print_exc()