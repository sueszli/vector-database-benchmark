"""
Module for Solaris 10's zonecfg

:maintainer:    Jorge Schrauwen <sjorge@blackdot.be>
:maturity:      new
:platform:      OmniOS,OpenIndiana,SmartOS,OpenSolaris,Solaris 10
:depend:        salt.modules.file

.. versionadded:: 2017.7.0

.. warning::
    Oracle Solaris 11's zonecfg is not supported by this module!
"""
import logging
import re
import salt.utils.args
import salt.utils.data
import salt.utils.decorators
import salt.utils.files
import salt.utils.path
from salt.utils.odict import OrderedDict
log = logging.getLogger(__name__)
__virtualname__ = 'zonecfg'
__func_alias__ = {'import_': 'import'}
_zonecfg_info_resources = ['rctl', 'net', 'fs', 'device', 'dedicated-cpu', 'dataset', 'attr']
_zonecfg_info_resources_calculated = ['capped-cpu', 'capped-memory']
_zonecfg_resource_setters = {'fs': ['dir', 'special', 'raw', 'type', 'options'], 'net': ['address', 'allowed-address', 'global-nic', 'mac-addr', 'physical', 'property', 'vlan-id defrouter'], 'device': ['match', 'property'], 'rctl': ['name', 'value'], 'attr': ['name', 'type', 'value'], 'dataset': ['name'], 'dedicated-cpu': ['ncpus', 'importance'], 'capped-cpu': ['ncpus'], 'capped-memory': ['physical', 'swap', 'locked'], 'admin': ['user', 'auths']}
_zonecfg_resource_default_selectors = {'fs': 'dir', 'net': 'mac-addr', 'device': 'match', 'rctl': 'name', 'attr': 'name', 'dataset': 'name', 'admin': 'user'}

@salt.utils.decorators.memoize
def _is_globalzone():
    if False:
        return 10
    '\n    Check if we are running in the globalzone\n    '
    if not __grains__['kernel'] == 'SunOS':
        return False
    zonename = __salt__['cmd.run_all']('zonename')
    if zonename['retcode']:
        return False
    if zonename['stdout'] == 'global':
        return True
    return False

def __virtual__():
    if False:
        return 10
    '\n    We are available if we are have zonecfg and are the global zone on\n    Solaris 10, OmniOS, OpenIndiana, OpenSolaris, or Smartos.\n    '
    if _is_globalzone() and salt.utils.path.which('zonecfg'):
        if __grains__['os'] in ['OpenSolaris', 'SmartOS', 'OmniOS', 'OpenIndiana']:
            return __virtualname__
        elif __grains__['os'] == 'Oracle Solaris' and int(__grains__['osmajorrelease']) == 10:
            return __virtualname__
    return (False, f'{__virtualname__} module can only be loaded in a solaris globalzone.')

def _clean_message(message):
    if False:
        i = 10
        return i + 15
    'Internal helper to sanitize message output'
    message = message.replace('zonecfg: ', '')
    message = message.splitlines()
    for line in message:
        if line.startswith('On line'):
            message.remove(line)
    return '\n'.join(message)

def _parse_value(value):
    if False:
        i = 10
        return i + 15
    'Internal helper for parsing configuration values into python values'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    elif isinstance(value, str):
        listparser = re.compile('((?:[^,"\']|"[^"]*"|\'[^\']*\')+)')
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            return listparser.split(value[1:-1])[1::2]
        elif value.startswith('(') and value.endswith(')'):
            rval = {}
            for pair in listparser.split(value[1:-1])[1::2]:
                pair = pair.split('=')
                if '"' in pair[1]:
                    pair[1] = pair[1].replace('"', '')
                if pair[1].isdigit():
                    rval[pair[0]] = int(pair[1])
                elif pair[1] == 'true':
                    rval[pair[0]] = True
                elif pair[1] == 'false':
                    rval[pair[0]] = False
                else:
                    rval[pair[0]] = pair[1]
            return rval
        else:
            if '"' in value:
                value = value.replace('"', '')
            if value.isdigit():
                return int(value)
            elif value == 'true':
                return True
            elif value == 'false':
                return False
            else:
                return value
    else:
        return value

def _sanitize_value(value):
    if False:
        for i in range(10):
            print('nop')
    'Internal helper for converting pythonic values to configuration file values'
    if isinstance(value, dict):
        new_value = []
        new_value.append('(')
        for (k, v) in value.items():
            new_value.append(k)
            new_value.append('=')
            new_value.append(v)
            new_value.append(',')
        new_value.append(')')
        return ''.join((str(v) for v in new_value)).replace(',)', ')')
    elif isinstance(value, list):
        new_value = []
        new_value.append('(')
        for item in value:
            if isinstance(item, OrderedDict):
                item = dict(item)
                for (k, v) in item.items():
                    new_value.append(k)
                    new_value.append('=')
                    new_value.append(v)
            else:
                new_value.append(item)
            new_value.append(',')
        new_value.append(')')
        return ''.join((str(v) for v in new_value)).replace(',)', ')')
    else:
        return f'"{value}"' if ' ' in value else value

def _dump_cfg(cfg_file):
    if False:
        return 10
    'Internal helper for debugging cfg files'
    if __salt__['file.file_exists'](cfg_file):
        with salt.utils.files.fopen(cfg_file, 'r') as fp_:
            log.debug('zonecfg - configuration file:\n%s', ''.join(salt.utils.data.decode(fp_.readlines())))

def create(zone, brand, zonepath, force=False):
    if False:
        i = 10
        return i + 15
    "\n    Create an in-memory configuration for the specified zone.\n\n    zone : string\n        name of zone\n    brand : string\n        brand name\n    zonepath : string\n        path of zone\n    force : boolean\n        overwrite configuration\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zonecfg.create deathscythe ipkg /zones/deathscythe\n    "
    ret = {'status': True}
    cfg_file = salt.utils.files.mkstemp()
    with salt.utils.files.fpopen(cfg_file, 'w+', mode=384) as fp_:
        fp_.write('create -b -F\n' if force else 'create -b\n')
        fp_.write(f'set brand={_sanitize_value(brand)}\n')
        fp_.write(f'set zonepath={_sanitize_value(zonepath)}\n')
    if not __salt__['file.directory_exists'](zonepath):
        __salt__['file.makedirs_perms'](zonepath if zonepath[-1] == '/' else f'{zonepath}/', mode='0700')
    _dump_cfg(cfg_file)
    res = __salt__['cmd.run_all']('zonecfg -z {zone} -f {cfg}'.format(zone=zone, cfg=cfg_file))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    if ret['message'] == '':
        del ret['message']
    else:
        ret['message'] = _clean_message(ret['message'])
    if __salt__['file.file_exists'](cfg_file):
        __salt__['file.remove'](cfg_file)
    return ret

def create_from_template(zone, template):
    if False:
        return 10
    "\n    Create an in-memory configuration from a template for the specified zone.\n\n    zone : string\n        name of zone\n    template : string\n        name of template\n\n    .. warning::\n        existing config will be overwritten!\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zonecfg.create_from_template leo tallgeese\n    "
    ret = {'status': True}
    _dump_cfg(template)
    res = __salt__['cmd.run_all']('zonecfg -z {zone} create -t {tmpl} -F'.format(zone=zone, tmpl=template))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    if ret['message'] == '':
        del ret['message']
    else:
        ret['message'] = _clean_message(ret['message'])
    return ret

def delete(zone):
    if False:
        i = 10
        return i + 15
    "\n    Delete the specified configuration from memory and stable storage.\n\n    zone : string\n        name of zone\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zonecfg.delete epyon\n    "
    ret = {'status': True}
    res = __salt__['cmd.run_all']('zonecfg -z {zone} delete -F'.format(zone=zone))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    if ret['message'] == '':
        del ret['message']
    else:
        ret['message'] = _clean_message(ret['message'])
    return ret

def export(zone, path=None):
    if False:
        i = 10
        return i + 15
    "\n    Export the configuration from memory to stable storage.\n\n    zone : string\n        name of zone\n    path : string\n        path of file to export to\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zonecfg.export epyon\n        salt '*' zonecfg.export epyon /zones/epyon.cfg\n    "
    ret = {'status': True}
    res = __salt__['cmd.run_all']('zonecfg -z {zone} export{path}'.format(zone=zone, path=f' -f {path}' if path else ''))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    if ret['message'] == '':
        del ret['message']
    else:
        ret['message'] = _clean_message(ret['message'])
    return ret

def import_(zone, path):
    if False:
        print('Hello World!')
    "\n    Import the configuration to memory from stable storage.\n\n    zone : string\n        name of zone\n    path : string\n        path of file to export to\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zonecfg.import epyon /zones/epyon.cfg\n    "
    ret = {'status': True}
    _dump_cfg(path)
    res = __salt__['cmd.run_all']('zonecfg -z {zone} -f {path}'.format(zone=zone, path=path))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    if ret['message'] == '':
        del ret['message']
    else:
        ret['message'] = _clean_message(ret['message'])
    return ret

def _property(methode, zone, key, value):
    if False:
        print('Hello World!')
    '\n    internal handler for set and clear_property\n\n    methode : string\n        either set, add, or clear\n    zone : string\n        name of zone\n    key : string\n        name of property\n    value : string\n        value of property\n\n    '
    ret = {'status': True}
    cfg_file = None
    if methode not in ['set', 'clear']:
        ret['status'] = False
        ret['message'] = f'unkown methode {methode}!'
    else:
        cfg_file = salt.utils.files.mkstemp()
        with salt.utils.files.fpopen(cfg_file, 'w+', mode=384) as fp_:
            if methode == 'set':
                if isinstance(value, dict) or isinstance(value, list):
                    value = _sanitize_value(value)
                value = str(value).lower() if isinstance(value, bool) else str(value)
                fp_.write(f'{methode} {key}={_sanitize_value(value)}\n')
            elif methode == 'clear':
                fp_.write(f'{methode} {key}\n')
    if cfg_file:
        _dump_cfg(cfg_file)
        res = __salt__['cmd.run_all']('zonecfg -z {zone} -f {path}'.format(zone=zone, path=cfg_file))
        ret['status'] = res['retcode'] == 0
        ret['message'] = res['stdout'] if ret['status'] else res['stderr']
        if ret['message'] == '':
            del ret['message']
        else:
            ret['message'] = _clean_message(ret['message'])
        if __salt__['file.file_exists'](cfg_file):
            __salt__['file.remove'](cfg_file)
    return ret

def set_property(zone, key, value):
    if False:
        return 10
    "\n    Set a property\n\n    zone : string\n        name of zone\n    key : string\n        name of property\n    value : string\n        value of property\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zonecfg.set_property deathscythe cpu-shares 100\n    "
    return _property('set', zone, key, value)

def clear_property(zone, key):
    if False:
        print('Hello World!')
    "\n    Clear a property\n\n    zone : string\n        name of zone\n    key : string\n        name of property\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zonecfg.clear_property deathscythe cpu-shares\n    "
    return _property('clear', zone, key, None)

def _resource(methode, zone, resource_type, resource_selector, **kwargs):
    if False:
        return 10
    '\n    internal resource hanlder\n\n    methode : string\n        add or update\n    zone : string\n        name of zone\n    resource_type : string\n        type of resource\n    resource_selector : string\n        unique resource identifier\n    **kwargs : string|int|...\n        resource properties\n\n    '
    ret = {'status': True}
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    for k in kwargs:
        if isinstance(kwargs[k], dict) or isinstance(kwargs[k], list):
            kwargs[k] = _sanitize_value(kwargs[k])
    if methode not in ['add', 'update']:
        ret['status'] = False
        ret['message'] = f'unknown methode {methode}'
        return ret
    if methode in ['update'] and resource_selector and (resource_selector not in kwargs):
        ret['status'] = False
        ret['message'] = 'resource selector {} not found in parameters'.format(resource_selector)
        return ret
    cfg_file = salt.utils.files.mkstemp()
    with salt.utils.files.fpopen(cfg_file, 'w+', mode=384) as fp_:
        if methode in ['add']:
            fp_.write(f'add {resource_type}\n')
        elif methode in ['update']:
            if resource_selector:
                value = kwargs[resource_selector]
                if isinstance(value, dict) or isinstance(value, list):
                    value = _sanitize_value(value)
                value = str(value).lower() if isinstance(value, bool) else str(value)
                fp_.write('select {} {}={}\n'.format(resource_type, resource_selector, _sanitize_value(value)))
            else:
                fp_.write(f'select {resource_type}\n')
        for (k, v) in kwargs.items():
            if methode in ['update'] and k == resource_selector:
                continue
            if isinstance(v, dict) or isinstance(v, list):
                value = _sanitize_value(value)
            value = str(v).lower() if isinstance(v, bool) else str(v)
            if k in _zonecfg_resource_setters[resource_type]:
                fp_.write(f'set {k}={_sanitize_value(value)}\n')
            else:
                fp_.write(f'add {k} {_sanitize_value(value)}\n')
        fp_.write('end\n')
    if cfg_file:
        _dump_cfg(cfg_file)
        res = __salt__['cmd.run_all']('zonecfg -z {zone} -f {path}'.format(zone=zone, path=cfg_file))
        ret['status'] = res['retcode'] == 0
        ret['message'] = res['stdout'] if ret['status'] else res['stderr']
        if ret['message'] == '':
            del ret['message']
        else:
            ret['message'] = _clean_message(ret['message'])
        if __salt__['file.file_exists'](cfg_file):
            __salt__['file.remove'](cfg_file)
    return ret

def add_resource(zone, resource_type, **kwargs):
    if False:
        print('Hello World!')
    "\n    Add a resource\n\n    zone : string\n        name of zone\n    resource_type : string\n        type of resource\n    kwargs : string|int|...\n        resource properties\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zonecfg.add_resource tallgeese rctl name=zone.max-locked-memory value='(priv=privileged,limit=33554432,action=deny)'\n    "
    return _resource('add', zone, resource_type, None, **kwargs)

def update_resource(zone, resource_type, resource_selector, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Add a resource\n\n    zone : string\n        name of zone\n    resource_type : string\n        type of resource\n    resource_selector : string\n        unique resource identifier\n    kwargs : string|int|...\n        resource properties\n\n    .. note::\n        Set resource_selector to None for resource that do not require one.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zonecfg.update_resource tallgeese rctl name name=zone.max-locked-memory value='(priv=privileged,limit=33554432,action=deny)'\n    "
    return _resource('update', zone, resource_type, resource_selector, **kwargs)

def remove_resource(zone, resource_type, resource_key, resource_value):
    if False:
        while True:
            i = 10
    "\n    Remove a resource\n\n    zone : string\n        name of zone\n    resource_type : string\n        type of resource\n    resource_key : string\n        key for resource selection\n    resource_value : string\n        value for resource selection\n\n    .. note::\n        Set resource_selector to None for resource that do not require one.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zonecfg.remove_resource tallgeese rctl name zone.max-locked-memory\n    "
    ret = {'status': True}
    cfg_file = salt.utils.files.mkstemp()
    with salt.utils.files.fpopen(cfg_file, 'w+', mode=384) as fp_:
        if resource_key:
            fp_.write('remove {} {}={}\n'.format(resource_type, resource_key, _sanitize_value(resource_value)))
        else:
            fp_.write(f'remove {resource_type}\n')
    if cfg_file:
        _dump_cfg(cfg_file)
        res = __salt__['cmd.run_all']('zonecfg -z {zone} -f {path}'.format(zone=zone, path=cfg_file))
        ret['status'] = res['retcode'] == 0
        ret['message'] = res['stdout'] if ret['status'] else res['stderr']
        if ret['message'] == '':
            del ret['message']
        else:
            ret['message'] = _clean_message(ret['message'])
        if __salt__['file.file_exists'](cfg_file):
            __salt__['file.remove'](cfg_file)
    return ret

def info(zone, show_all=False):
    if False:
        print('Hello World!')
    "\n    Display the configuration from memory\n\n    zone : string\n        name of zone\n    show_all : boolean\n        also include calculated values like capped-cpu, cpu-shares, ...\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zonecfg.info tallgeese\n    "
    ret = {}
    res = __salt__['cmd.run_all']('zonecfg -z {zone} info'.format(zone=zone))
    if res['retcode'] == 0:
        resname = None
        resdata = {}
        for line in res['stdout'].split('\n'):
            if ':' not in line:
                continue
            if line.startswith('['):
                if not show_all:
                    continue
                line = line.rstrip()[1:-1]
            key = line.strip().split(':')[0]
            if '[' in key:
                key = key[1:]
            if key in _zonecfg_info_resources_calculated:
                if resname:
                    ret[resname].append(resdata)
                if show_all:
                    resname = key
                    resdata = {}
                    if key not in ret:
                        ret[key] = []
                else:
                    resname = None
                    resdata = {}
            elif key in _zonecfg_info_resources:
                if resname:
                    ret[resname].append(resdata)
                resname = key
                resdata = {}
                if key not in ret:
                    ret[key] = []
            elif line.startswith('\t'):
                if line.strip().startswith('['):
                    if not show_all:
                        continue
                    line = line.strip()[1:-1]
                if key == 'property':
                    if 'property' not in resdata:
                        resdata[key] = {}
                    kv = _parse_value(line.strip()[line.strip().index(':') + 1:])
                    if 'name' in kv and 'value' in kv:
                        resdata[key][kv['name']] = kv['value']
                    else:
                        log.warning('zonecfg.info - not sure how to deal with: %s', kv)
                else:
                    resdata[key] = _parse_value(line.strip()[line.strip().index(':') + 1:])
            else:
                if resname:
                    ret[resname].append(resdata)
                resname = None
                resdata = {}
                if key == 'property':
                    if 'property' not in ret:
                        ret[key] = {}
                    kv = _parse_value(line.strip()[line.strip().index(':') + 1:])
                    if 'name' in kv and 'value' in kv:
                        res[key][kv['name']] = kv['value']
                    else:
                        log.warning('zonecfg.info - not sure how to deal with: %s', kv)
                else:
                    ret[key] = _parse_value(line.strip()[line.strip().index(':') + 1:])
        if resname:
            ret[resname].append(resdata)
    return ret