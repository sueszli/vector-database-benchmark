"""
Runner for SmartOS minions control vmadm
"""
import salt.client
from salt.exceptions import SaltClientError
from salt.utils.odict import OrderedDict
__func_alias__ = {'list_vms': 'list'}
__virtualname__ = 'vmadm'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Provides vmadm runner\n    '
    return __virtualname__

def _action(action='get', search=None, one=True, force=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Multi action helper for start, stop, get, ...\n    '
    vms = {}
    matched_vms = []
    with salt.client.get_local_client(__opts__['conf_file']) as client:
        try:
            vmadm_args = {}
            vmadm_args['order'] = 'uuid,alias,hostname,state'
            if '=' in search:
                vmadm_args['search'] = search
            for cn in client.cmd_iter('G@virtual:physical and G@os:smartos', 'vmadm.list', kwarg=vmadm_args, tgt_type='compound'):
                if not cn:
                    continue
                node = next(iter(cn.keys()))
                if not isinstance(cn[node], dict) or 'ret' not in cn[node] or (not isinstance(cn[node]['ret'], dict)):
                    continue
                for vm in cn[node]['ret']:
                    vmcfg = cn[node]['ret'][vm]
                    vmcfg['node'] = node
                    vms[vm] = vmcfg
        except SaltClientError as client_error:
            pass
        if len(vms) == 0:
            return {'Error': 'No vms found.'}
        if '=' not in search:
            loop_pass = 0
            while loop_pass < 3:
                if loop_pass == 0:
                    field = 'uuid'
                elif loop_pass == 1:
                    field = 'hostname'
                else:
                    field = 'alias'
                for vm in vms:
                    if field == 'uuid' and vm == search:
                        matched_vms.append(vm)
                        break
                    elif field in vms[vm] and vms[vm][field] == search:
                        matched_vms.append(vm)
                if len(matched_vms) > 0:
                    break
                else:
                    loop_pass += 1
        else:
            for vm in vms:
                matched_vms.append(vm)
        if len(matched_vms) == 0:
            return {'Error': 'No vms matched.'}
        if one and len(matched_vms) > 1:
            return {'Error': f'Matched {len(matched_vms)} vms, only one allowed!', 'Matches': matched_vms}
        ret = {}
        if action in ['start', 'stop', 'reboot', 'get']:
            for vm in matched_vms:
                vmadm_args = {'key': 'uuid', 'vm': vm}
                try:
                    for vmadm_res in client.cmd_iter(vms[vm]['node'], f'vmadm.{action}', kwarg=vmadm_args):
                        if not vmadm_res:
                            continue
                        if vms[vm]['node'] in vmadm_res:
                            ret[vm] = vmadm_res[vms[vm]['node']]['ret']
                except SaltClientError as client_error:
                    ret[vm] = False
        elif action in ['is_running']:
            ret = True
            for vm in matched_vms:
                if vms[vm]['state'] != 'running':
                    ret = False
                    break
        return ret

def nodes(verbose=False):
    if False:
        while True:
            i = 10
    '\n    List all compute nodes\n\n    verbose : boolean\n        print additional information about the node\n        e.g. platform version, hvm capable, ...\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run vmadm.nodes\n        salt-run vmadm.nodes verbose=True\n    '
    ret = {} if verbose else []
    with salt.client.get_local_client(__opts__['conf_file']) as client:
        try:
            for cn in client.cmd_iter('G@virtual:physical and G@os:smartos', 'grains.items', tgt_type='compound'):
                if not cn:
                    continue
                node = next(iter(cn.keys()))
                if not isinstance(cn[node], dict) or 'ret' not in cn[node] or (not isinstance(cn[node]['ret'], dict)):
                    continue
                if verbose:
                    ret[node] = {}
                    ret[node]['version'] = {}
                    ret[node]['version']['platform'] = cn[node]['ret']['osrelease']
                    if 'computenode_sdc_version' in cn[node]['ret']:
                        ret[node]['version']['sdc'] = cn[node]['ret']['computenode_sdc_version']
                    ret[node]['vms'] = {}
                    if 'computenode_vm_capable' in cn[node]['ret'] and cn[node]['ret']['computenode_vm_capable'] and ('computenode_vm_hw_virt' in cn[node]['ret']):
                        ret[node]['vms']['hw_cap'] = cn[node]['ret']['computenode_vm_hw_virt']
                    else:
                        ret[node]['vms']['hw_cap'] = False
                    if 'computenode_vms_running' in cn[node]['ret']:
                        ret[node]['vms']['running'] = cn[node]['ret']['computenode_vms_running']
                else:
                    ret.append(node)
        except SaltClientError as client_error:
            return f'{client_error}'
        if not verbose:
            ret.sort()
        return ret

def list_vms(search=None, verbose=False):
    if False:
        print('Hello World!')
    "\n    List all vms\n\n    search : string\n        filter vms, see the execution module\n    verbose : boolean\n        print additional information about the vm\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run vmadm.list\n        salt-run vmadm.list search='type=KVM'\n        salt-run vmadm.list verbose=True\n    "
    ret = OrderedDict() if verbose else []
    with salt.client.get_local_client(__opts__['conf_file']) as client:
        try:
            vmadm_args = {}
            vmadm_args['order'] = 'uuid,alias,hostname,state,type,cpu_cap,vcpus,ram'
            if search:
                vmadm_args['search'] = search
            for cn in client.cmd_iter('G@virtual:physical and G@os:smartos', 'vmadm.list', kwarg=vmadm_args, tgt_type='compound'):
                if not cn:
                    continue
                node = next(iter(cn.keys()))
                if not isinstance(cn[node], dict) or 'ret' not in cn[node] or (not isinstance(cn[node]['ret'], dict)):
                    continue
                for vm in cn[node]['ret']:
                    vmcfg = cn[node]['ret'][vm]
                    if verbose:
                        ret[vm] = OrderedDict()
                        ret[vm]['hostname'] = vmcfg['hostname']
                        ret[vm]['alias'] = vmcfg['alias']
                        ret[vm]['computenode'] = node
                        ret[vm]['state'] = vmcfg['state']
                        ret[vm]['resources'] = OrderedDict()
                        ret[vm]['resources']['memory'] = vmcfg['ram']
                        if vmcfg['type'] == 'KVM':
                            ret[vm]['resources']['cpu'] = '{:.2f}'.format(int(vmcfg['vcpus']))
                        elif vmcfg['cpu_cap'] != '':
                            ret[vm]['resources']['cpu'] = '{:.2f}'.format(int(vmcfg['cpu_cap']) / 100)
                    else:
                        ret.append(vm)
        except SaltClientError as client_error:
            return f'{client_error}'
        if not verbose:
            ret = sorted(ret)
        return ret

def start(search, one=True):
    if False:
        while True:
            i = 10
    "\n    Start one or more vms\n\n    search : string\n        filter vms, see the execution module.\n    one : boolean\n        start only one vm\n\n    .. note::\n        If the search parameter does not contain an equal (=) symbol it will be\n        assumed it will be tried as uuid, hostname, and alias.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run vmadm.start 91244bba-1146-e4ec-c07e-e825e0223aa9\n        salt-run vmadm.start search='alias=jiska'\n        salt-run vmadm.start search='type=KVM' one=False\n    "
    return _action('start', search, one)

def stop(search, one=True):
    if False:
        i = 10
        return i + 15
    "\n    Stop one or more vms\n\n    search : string\n        filter vms, see the execution module.\n    one : boolean\n        stop only one vm\n\n    .. note::\n        If the search parameter does not contain an equal (=) symbol it will be\n        assumed it will be tried as uuid, hostname, and alias.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run vmadm.stop 91244bba-1146-e4ec-c07e-e825e0223aa9\n        salt-run vmadm.stop search='alias=jody'\n        salt-run vmadm.stop search='type=KVM' one=False\n    "
    return _action('stop', search, one)

def reboot(search, one=True, force=False):
    if False:
        return 10
    "\n    Reboot one or more vms\n\n    search : string\n        filter vms, see the execution module.\n    one : boolean\n        reboot only one vm\n    force : boolean\n        force reboot, faster but no graceful shutdown\n\n    .. note::\n        If the search parameter does not contain an equal (=) symbol it will be\n        assumed it will be tried as uuid, hostname, and alias.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run vmadm.reboot 91244bba-1146-e4ec-c07e-e825e0223aa9\n        salt-run vmadm.reboot search='alias=marije'\n        salt-run vmadm.reboot search='type=KVM' one=False\n    "
    return _action('reboot', search, one, force)

def get(search, one=True):
    if False:
        return 10
    "\n    Return information for vms\n\n    search : string\n        filter vms, see the execution module.\n    one : boolean\n        return only one vm\n\n    .. note::\n        If the search parameter does not contain an equal (=) symbol it will be\n        assumed it will be tried as uuid, hostname, and alias.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run vmadm.get 91244bba-1146-e4ec-c07e-e825e0223aa9\n        salt-run vmadm.get search='alias=saskia'\n    "
    return _action('get', search, one)

def is_running(search):
    if False:
        return 10
    "\n    Return true if vm is running\n\n    search : string\n        filter vms, see the execution module.\n\n    .. note::\n        If the search parameter does not contain an equal (=) symbol it will be\n        assumed it will be tried as uuid, hostname, and alias.\n\n    .. note::\n        If multiple vms are matched, the result will be true of ALL vms are running\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run vmadm.is_running 91244bba-1146-e4ec-c07e-e825e0223aa9\n        salt-run vmadm.is_running search='alias=julia'\n    "
    return _action('is_running', search, False)