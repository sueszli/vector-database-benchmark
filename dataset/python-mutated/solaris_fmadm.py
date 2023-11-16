"""
Module for running fmadm and fmdump on Solaris

:maintainer:    Jorge Schrauwen <sjorge@blackdot.be>
:maturity:      new
:platform:      solaris,illumos

.. versionadded:: 2016.3.0
"""
import logging
import salt.utils.decorators as decorators
import salt.utils.path
import salt.utils.platform
from salt.utils.odict import OrderedDict
log = logging.getLogger(__name__)
__func_alias__ = {'list_records': 'list'}
__virtualname__ = 'fmadm'

@decorators.memoize
def _check_fmadm():
    if False:
        i = 10
        return i + 15
    '\n    Looks to see if fmadm is present on the system\n    '
    return salt.utils.path.which('fmadm')

def _check_fmdump():
    if False:
        return 10
    '\n    Looks to see if fmdump is present on the system\n    '
    return salt.utils.path.which('fmdump')

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Provides fmadm only on Solaris\n    '
    if salt.utils.platform.is_sunos() and _check_fmadm() and _check_fmdump():
        return __virtualname__
    return (False, '{} module can only be loaded on Solaris with the fault management installed'.format(__virtualname__))

def _parse_fmdump(output):
    if False:
        return 10
    '\n    Parses fmdump output\n    '
    result = []
    output = output.split('\n')
    header = [field for field in output[0].lower().split(' ') if field]
    del output[0]
    for entry in output:
        entry = [item for item in entry.split(' ') if item]
        entry = [f'{entry[0]} {entry[1]} {entry[2]}'] + entry[3:]
        fault = OrderedDict()
        for field in header:
            fault[field] = entry[header.index(field)]
        result.append(fault)
    return result

def _parse_fmdump_verbose(output):
    if False:
        while True:
            i = 10
    '\n    Parses fmdump verbose output\n    '
    result = []
    output = output.split('\n')
    fault = []
    verbose_fault = {}
    for line in output:
        if line.startswith('TIME'):
            fault.append(line)
            if verbose_fault:
                result.append(verbose_fault)
                verbose_fault = {}
        elif len(fault) == 1:
            fault.append(line)
            verbose_fault = _parse_fmdump('\n'.join(fault))[0]
            fault = []
        elif verbose_fault:
            if 'details' not in verbose_fault:
                verbose_fault['details'] = ''
            if line.strip() == '':
                continue
            verbose_fault['details'] = '{}{}\n'.format(verbose_fault['details'], line)
    if len(verbose_fault) > 0:
        result.append(verbose_fault)
    return result

def _parse_fmadm_config(output):
    if False:
        i = 10
        return i + 15
    '\n    Parsbb fmdump/fmadm output\n    '
    result = []
    output = output.split('\n')
    header = [field for field in output[0].lower().split(' ') if field]
    del output[0]
    for entry in output:
        entry = [item for item in entry.split(' ') if item]
        entry = entry[0:3] + [' '.join(entry[3:])]
        component = OrderedDict()
        for field in header:
            component[field] = entry[header.index(field)]
        result.append(component)
    keyed_result = OrderedDict()
    for component in result:
        keyed_result[component['module']] = component
        del keyed_result[component['module']]['module']
    result = keyed_result
    return result

def _fmadm_action_fmri(action, fmri):
    if False:
        i = 10
        return i + 15
    '\n    Internal function for fmadm.repqired, fmadm.replaced, fmadm.flush\n    '
    ret = {}
    fmadm = _check_fmadm()
    cmd = f'{fmadm} {action} {fmri}'
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    result = {}
    if retcode != 0:
        result['Error'] = res['stderr']
    else:
        result = True
    return result

def _parse_fmadm_faulty(output):
    if False:
        return 10
    '\n    Parse fmadm faulty output\n    '

    def _merge_data(summary, fault):
        if False:
            i = 10
            return i + 15
        result = {}
        uuid = summary['event-id']
        del summary['event-id']
        result[uuid] = OrderedDict()
        result[uuid]['summary'] = summary
        result[uuid]['fault'] = fault
        return result
    result = {}
    summary = []
    summary_data = {}
    fault_data = {}
    data_key = None
    for line in output.split('\n'):
        if line.startswith('-'):
            if summary and summary_data and fault_data:
                result.update(_merge_data(summary_data, fault_data))
                summary = []
                summary_data = {}
                fault_data = {}
                continue
            else:
                continue
        if not summary:
            summary.append(line)
            continue
        if summary and (not summary_data):
            summary.append(line)
            summary_data = _parse_fmdump('\n'.join(summary))[0]
            continue
        if summary and summary_data:
            if line.startswith(' ') and data_key:
                fault_data[data_key] = '{}\n{}'.format(fault_data[data_key], line.strip())
            elif ':' in line:
                line = line.split(':')
                data_key = line[0].strip()
                fault_data[data_key] = ':'.join(line[1:]).strip()
                if data_key == 'Platform':
                    fault_data['Chassis_id'] = fault_data[data_key][fault_data[data_key].index('Chassis_id'):].split(':')[-1].strip()
                    fault_data[data_key] = fault_data[data_key][0:fault_data[data_key].index('Chassis_id')].strip()
    result.update(_merge_data(summary_data, fault_data))
    return result

def list_records(after=None, before=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Display fault management logs\n\n    after : string\n        filter events after time, see man fmdump for format\n\n    before : string\n        filter events before time, see man fmdump for format\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' fmadm.list\n    "
    ret = {}
    fmdump = _check_fmdump()
    cmd = '{cmd}{after}{before}'.format(cmd=fmdump, after=f' -t {after}' if after else '', before=f' -T {before}' if before else '')
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    result = {}
    if retcode != 0:
        result['Error'] = 'error executing fmdump'
    else:
        result = _parse_fmdump(res['stdout'])
    return result

def show(uuid):
    if False:
        i = 10
        return i + 15
    "\n    Display log details\n\n    uuid: string\n        uuid of fault\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' fmadm.show 11b4070f-4358-62fa-9e1e-998f485977e1\n    "
    ret = {}
    fmdump = _check_fmdump()
    cmd = f'{fmdump} -u {uuid} -V'
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    result = {}
    if retcode != 0:
        result['Error'] = 'error executing fmdump'
    else:
        result = _parse_fmdump_verbose(res['stdout'])
    return result

def config():
    if False:
        return 10
    "\n    Display fault manager configuration\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' fmadm.config\n    "
    ret = {}
    fmadm = _check_fmadm()
    cmd = f'{fmadm} config'
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    result = {}
    if retcode != 0:
        result['Error'] = 'error executing fmadm config'
    else:
        result = _parse_fmadm_config(res['stdout'])
    return result

def load(path):
    if False:
        for i in range(10):
            print('nop')
    "\n    Load specified fault manager module\n\n    path: string\n        path of fault manager module\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' fmadm.load /module/path\n    "
    ret = {}
    fmadm = _check_fmadm()
    cmd = f'{fmadm} load {path}'
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    result = {}
    if retcode != 0:
        result['Error'] = res['stderr']
    else:
        result = True
    return result

def unload(module):
    if False:
        i = 10
        return i + 15
    "\n    Unload specified fault manager module\n\n    module: string\n        module to unload\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' fmadm.unload software-response\n    "
    ret = {}
    fmadm = _check_fmadm()
    cmd = f'{fmadm} unload {module}'
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    result = {}
    if retcode != 0:
        result['Error'] = res['stderr']
    else:
        result = True
    return result

def reset(module, serd=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Reset module or sub-component\n\n    module: string\n        module to unload\n    serd : string\n        serd sub module\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' fmadm.reset software-response\n    "
    ret = {}
    fmadm = _check_fmadm()
    cmd = '{cmd} reset {serd}{module}'.format(cmd=fmadm, serd=f'-s {serd} ' if serd else '', module=module)
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    result = {}
    if retcode != 0:
        result['Error'] = res['stderr']
    else:
        result = True
    return result

def flush(fmri):
    if False:
        i = 10
        return i + 15
    "\n    Flush cached state for resource\n\n    fmri: string\n        fmri\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' fmadm.flush fmri\n    "
    return _fmadm_action_fmri('flush', fmri)

def repaired(fmri):
    if False:
        return 10
    "\n    Notify fault manager that resource has been repaired\n\n    fmri: string\n        fmri\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' fmadm.repaired fmri\n    "
    return _fmadm_action_fmri('repaired', fmri)

def replaced(fmri):
    if False:
        i = 10
        return i + 15
    "\n    Notify fault manager that resource has been replaced\n\n    fmri: string\n        fmri\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' fmadm.repaired fmri\n    "
    return _fmadm_action_fmri('replaced', fmri)

def acquit(fmri):
    if False:
        while True:
            i = 10
    "\n    Acquit resource or acquit case\n\n    fmri: string\n        fmri or uuid\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' fmadm.acquit fmri | uuid\n    "
    return _fmadm_action_fmri('acquit', fmri)

def faulty():
    if False:
        while True:
            i = 10
    "\n    Display list of faulty resources\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' fmadm.faulty\n    "
    fmadm = _check_fmadm()
    cmd = '{cmd} faulty'.format(cmd=fmadm)
    res = __salt__['cmd.run_all'](cmd)
    result = {}
    if res['stdout'] == '':
        result = False
    else:
        result = _parse_fmadm_faulty(res['stdout'])
    return result

def healthy():
    if False:
        print('Hello World!')
    "\n    Return whether fmadm is reporting faults\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' fmadm.healthy\n    "
    return False if faulty() else True