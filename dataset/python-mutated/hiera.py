"""
Use hiera data as a Pillar source
"""
import logging
import salt.utils.path
import salt.utils.yaml
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only return if hiera is installed\n    '
    return 'hiera' if salt.utils.path.which('hiera') else False

def ext_pillar(minion_id, pillar, conf):
    if False:
        for i in range(10):
            print('nop')
    '\n    Execute hiera and return the data\n    '
    cmd = 'hiera -c {}'.format(conf)
    for (key, val) in __grains__.items():
        if isinstance(val, str):
            cmd += " {}='{}'".format(key, val)
    try:
        data = salt.utils.yaml.safe_load(__salt__['cmd.run'](cmd))
    except Exception:
        log.critical('Hiera YAML data failed to parse from conf %s', conf)
        return {}
    return data