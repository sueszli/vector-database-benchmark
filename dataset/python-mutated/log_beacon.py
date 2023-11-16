"""
Beacon to fire events at specific log messages.

.. versionadded:: 2017.7.0

"""
import logging
import salt.utils.beacons
import salt.utils.files
import salt.utils.platform
try:
    import re
    HAS_REGEX = True
except ImportError:
    HAS_REGEX = False
__virtualname__ = 'log'
LOC_KEY = 'log.loc'
SKEL = {}
SKEL['tag'] = ''
SKEL['match'] = 'no'
SKEL['raw'] = ''
SKEL['error'] = ''
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        print('Hello World!')
    if not salt.utils.platform.is_windows() and HAS_REGEX:
        return __virtualname__
    err_msg = 'Not available for Windows systems or when regex library is missing.'
    log.error('Unable to load %s beacon: %s', __virtualname__, err_msg)
    return (False, err_msg)

def _get_loc():
    if False:
        i = 10
        return i + 15
    '\n    return the active file location\n    '
    if LOC_KEY in __context__:
        return __context__[LOC_KEY]

def validate(config):
    if False:
        print('Hello World!')
    '\n    Validate the beacon configuration\n    '
    if not isinstance(config, list):
        return (False, 'Configuration for log beacon must be a list.')
    config = salt.utils.beacons.list_to_dict(config)
    if 'file' not in config:
        return (False, 'Configuration for log beacon must contain file option.')
    return (True, 'Valid beacon configuration')

def beacon(config):
    if False:
        i = 10
        return i + 15
    '\n    Read the log file and return match whole string\n\n    .. code-block:: yaml\n\n        beacons:\n          log:\n            - file: <path>\n            - tags:\n                <tag>:\n                  regex: <pattern>\n\n    .. note::\n\n        regex matching is based on the `re`_ module\n\n    .. _re: https://docs.python.org/3.6/library/re.html#regular-expression-syntax\n\n    The defined tag is added to the beacon event tag.\n    This is not the tag in the log.\n\n    .. code-block:: yaml\n\n        beacons:\n          log:\n            - file: /var/log/messages #path to log.\n            - tags:\n                goodbye/world: # tag added to beacon event tag.\n                  regex: .*good-bye.* # match good-bye string anywhere in the log entry.\n    '
    config = salt.utils.beacons.list_to_dict(config)
    ret = []
    if 'file' not in config:
        event = SKEL.copy()
        event['tag'] = 'global'
        event['error'] = 'file not defined in config'
        ret.append(event)
        return ret
    with salt.utils.files.fopen(config['file'], 'r') as fp_:
        loc = __context__.get(LOC_KEY, 0)
        if loc == 0:
            fp_.seek(0, 2)
            __context__[LOC_KEY] = fp_.tell()
            return ret
        fp_.seek(0, 2)
        __context__[LOC_KEY] = fp_.tell()
        fp_.seek(loc)
        txt = fp_.read()
        log.info('txt %s', txt)
        d = {}
        for tag in config.get('tags', {}):
            if 'regex' not in config['tags'][tag]:
                continue
            if not config['tags'][tag]['regex']:
                continue
            try:
                d[tag] = re.compile('{}'.format(config['tags'][tag]['regex']))
            except Exception as e:
                event = SKEL.copy()
                event['tag'] = tag
                event['error'] = 'bad regex'
                ret.append(event)
        for line in txt.splitlines():
            for (tag, reg) in d.items():
                try:
                    m = reg.match(line)
                    if m:
                        event = SKEL.copy()
                        event['tag'] = tag
                        event['raw'] = line
                        event['match'] = 'yes'
                        ret.append(event)
                except Exception:
                    event = SKEL.copy()
                    event['tag'] = tag
                    event['error'] = 'bad match'
                    ret.append(event)
    return ret