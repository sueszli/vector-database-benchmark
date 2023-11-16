"""
This module (mostly) uses the XenAPI to manage Xen virtual machines.

Big fat warning: the XenAPI used in this file is the one bundled with
Xen Source, NOT XenServer nor Xen Cloud Platform. As a matter of fact it
*will* fail under those platforms. From what I've read, little work is needed
to adapt this code to XS/XCP, mostly playing with XenAPI version, but as
XCP is not taking precedence on Xen Source on many platforms, please keep
compatibility in mind.

Useful documentation:

. http://downloads.xen.org/Wiki/XenAPI/xenapi-1.0.6.pdf
. http://docs.vmd.citrix.com/XenServer/6.0.0/1.0/en_gb/api/
. https://github.com/xapi-project/xen-api/tree/master/scripts/examples/python
. http://xenbits.xen.org/gitweb/?p=xen.git;a=tree;f=tools/python/xen/xm;hb=HEAD
"""
import contextlib
import os
import sys
import salt.modules.cmdmod
import salt.utils.files
import salt.utils.path
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError
try:
    import importlib
    HAS_IMPORTLIB = True
except ImportError:
    HAS_IMPORTLIB = False
__virtualname__ = 'virt'

def _check_xenapi():
    if False:
        print('Hello World!')
    if __grains__['os'] == 'Debian':
        debian_xen_version = '/usr/lib/xen-common/bin/xen-version'
        if os.path.isfile(debian_xen_version):
            xenversion = salt.modules.cmdmod._run_quiet(debian_xen_version)
            xapipath = '/usr/lib/xen-{}/lib/python'.format(xenversion)
            if os.path.isdir(xapipath):
                sys.path.append(xapipath)
    try:
        if HAS_IMPORTLIB:
            return importlib.import_module('xen.xm.XenAPI')
        return __import__('xen.xm.XenAPI').xm.XenAPI
    except (ImportError, AttributeError):
        return False

def __virtual__():
    if False:
        return 10
    if _check_xenapi() is not False:
        return __virtualname__
    return (False, 'Module xapi: xenapi check failed')

@contextlib.contextmanager
def _get_xapi_session():
    if False:
        for i in range(10):
            print('nop')
    '\n    Get a session to XenAPI. By default, use the local UNIX socket.\n    '
    _xenapi = _check_xenapi()
    xapi_uri = __salt__['config.option']('xapi.uri')
    xapi_login = __salt__['config.option']('xapi.login')
    xapi_password = __salt__['config.option']('xapi.password')
    if not xapi_uri:
        xapi_uri = 'httpu:///var/run/xend/xen-api.sock'
    if not xapi_login:
        xapi_login = ''
    if not xapi_password:
        xapi_password = ''
    try:
        session = _xenapi.Session(xapi_uri)
        session.xenapi.login_with_password(xapi_login, xapi_password)
        yield session.xenapi
    except Exception:
        raise CommandExecutionError('Failed to connect to XenAPI socket.')
    finally:
        session.xenapi.session.logout()

def _get_xtool():
    if False:
        return 10
    '\n    Internal, returns xl or xm command line path\n    '
    for xtool in ['xl', 'xm']:
        path = salt.utils.path.which(xtool)
        if path is not None:
            return path

def _get_all(xapi, rectype):
    if False:
        print('Hello World!')
    '\n    Internal, returns all members of rectype\n    '
    return getattr(xapi, rectype).get_all()

def _get_label_uuid(xapi, rectype, label):
    if False:
        i = 10
        return i + 15
    "\n    Internal, returns label's uuid\n    "
    try:
        return getattr(xapi, rectype).get_by_name_label(label)[0]
    except Exception:
        return False

def _get_record(xapi, rectype, uuid):
    if False:
        print('Hello World!')
    '\n    Internal, returns a full record for uuid\n    '
    return getattr(xapi, rectype).get_record(uuid)

def _get_record_by_label(xapi, rectype, label):
    if False:
        for i in range(10):
            print('nop')
    '\n    Internal, returns a full record for uuid\n    '
    uuid = _get_label_uuid(xapi, rectype, label)
    if uuid is False:
        return False
    return getattr(xapi, rectype).get_record(uuid)

def _get_metrics_record(xapi, rectype, record):
    if False:
        print('Hello World!')
    '\n    Internal, returns metrics record for a rectype\n    '
    metrics_id = record['metrics']
    return getattr(xapi, '{}_metrics'.format(rectype)).get_record(metrics_id)

def _get_val(record, keys):
    if False:
        for i in range(10):
            print('nop')
    '\n    Internal, get value from record\n    '
    data = record
    for key in keys:
        if key in data:
            data = data[key]
        else:
            return None
    return data

def list_domains():
    if False:
        print('Hello World!')
    "\n    Return a list of virtual machine names on the minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.list_domains\n    "
    with _get_xapi_session() as xapi:
        hosts = xapi.VM.get_all()
        ret = []
        for _host in hosts:
            if xapi.VM.get_record(_host)['is_control_domain'] is False:
                ret.append(xapi.VM.get_name_label(_host))
        return ret

def vm_info(vm_=None):
    if False:
        print('Hello World!')
    "\n    Return detailed information about the vms.\n\n    If you pass a VM name in as an argument then it will return info\n    for just the named VM, otherwise it will return all VMs.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.vm_info\n    "
    with _get_xapi_session() as xapi:

        def _info(vm_):
            if False:
                return 10
            vm_rec = _get_record_by_label(xapi, 'VM', vm_)
            if vm_rec is False:
                return False
            vm_metrics_rec = _get_metrics_record(xapi, 'VM', vm_rec)
            return {'cpu': vm_metrics_rec['VCPUs_number'], 'maxCPU': _get_val(vm_rec, ['VCPUs_max']), 'cputime': vm_metrics_rec['VCPUs_utilisation'], 'disks': get_disks(vm_), 'nics': get_nics(vm_), 'maxMem': int(_get_val(vm_rec, ['memory_dynamic_max'])), 'mem': int(vm_metrics_rec['memory_actual']), 'state': _get_val(vm_rec, ['power_state'])}
        info = {}
        if vm_:
            ret = _info(vm_)
            if ret is not None:
                info[vm_] = ret
        else:
            for vm_ in list_domains():
                ret = _info(vm_)
                if ret is not None:
                    info[vm_] = _info(vm_)
        return info

def vm_state(vm_=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return list of all the vms and their state.\n\n    If you pass a VM name in as an argument then it will return info\n    for just the named VM, otherwise it will return all VMs.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.vm_state <vm name>\n    "
    with _get_xapi_session() as xapi:
        info = {}
        if vm_:
            info[vm_] = _get_record_by_label(xapi, 'VM', vm_)['power_state']
            return info
        for vm_ in list_domains():
            info[vm_] = _get_record_by_label(xapi, 'VM', vm_)['power_state']
        return info

def node_info():
    if False:
        return 10
    "\n    Return a dict with information about this node\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.node_info\n    "
    with _get_xapi_session() as xapi:
        host_rec = _get_record(xapi, 'host', _get_all(xapi, 'host')[0])
        host_cpu_rec = _get_record(xapi, 'host_cpu', host_rec['host_CPUs'][0])
        host_metrics_rec = _get_metrics_record(xapi, 'host', host_rec)

        def getCpuMhz():
            if False:
                i = 10
                return i + 15
            cpu_speeds = [int(host_cpu_rec['speed']) for host_cpu_it in host_cpu_rec if 'speed' in host_cpu_it]
            if cpu_speeds:
                return sum(cpu_speeds) / len(cpu_speeds)
            else:
                return 0

        def getCpuFeatures():
            if False:
                i = 10
                return i + 15
            if host_cpu_rec:
                return host_cpu_rec['features']

        def getFreeCpuCount():
            if False:
                i = 10
                return i + 15
            cnt = 0
            for host_cpu_it in host_cpu_rec:
                if not host_cpu_rec['cpu_pool']:
                    cnt += 1
            return cnt
        info = {'cpucores': _get_val(host_rec, ['cpu_configuration', 'nr_cpus']), 'cpufeatures': getCpuFeatures(), 'cpumhz': getCpuMhz(), 'cpuarch': _get_val(host_rec, ['software_version', 'machine']), 'cputhreads': _get_val(host_rec, ['cpu_configuration', 'threads_per_core']), 'phymemory': int(host_metrics_rec['memory_total']) / 1024 / 1024, 'cores_per_sockets': _get_val(host_rec, ['cpu_configuration', 'cores_per_socket']), 'free_cpus': getFreeCpuCount(), 'free_memory': int(host_metrics_rec['memory_free']) / 1024 / 1024, 'xen_major': _get_val(host_rec, ['software_version', 'xen_major']), 'xen_minor': _get_val(host_rec, ['software_version', 'xen_minor']), 'xen_extra': _get_val(host_rec, ['software_version', 'xen_extra']), 'xen_caps': ' '.join(_get_val(host_rec, ['capabilities'])), 'xen_scheduler': _get_val(host_rec, ['sched_policy']), 'xen_pagesize': _get_val(host_rec, ['other_config', 'xen_pagesize']), 'platform_params': _get_val(host_rec, ['other_config', 'platform_params']), 'xen_commandline': _get_val(host_rec, ['other_config', 'xen_commandline']), 'xen_changeset': _get_val(host_rec, ['software_version', 'xen_changeset']), 'cc_compiler': _get_val(host_rec, ['software_version', 'cc_compiler']), 'cc_compile_by': _get_val(host_rec, ['software_version', 'cc_compile_by']), 'cc_compile_domain': _get_val(host_rec, ['software_version', 'cc_compile_domain']), 'cc_compile_date': _get_val(host_rec, ['software_version', 'cc_compile_date']), 'xend_config_format': _get_val(host_rec, ['software_version', 'xend_config_format'])}
        return info

def get_nics(vm_):
    if False:
        return 10
    "\n    Return info about the network interfaces of a named vm\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.get_nics <vm name>\n    "
    with _get_xapi_session() as xapi:
        nic = {}
        vm_rec = _get_record_by_label(xapi, 'VM', vm_)
        if vm_rec is False:
            return False
        for vif in vm_rec['VIFs']:
            vif_rec = _get_record(xapi, 'VIF', vif)
            nic[vif_rec['MAC']] = {'mac': vif_rec['MAC'], 'device': vif_rec['device'], 'mtu': vif_rec['MTU']}
        return nic

def get_macs(vm_):
    if False:
        return 10
    "\n    Return a list off MAC addresses from the named vm\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.get_macs <vm name>\n    "
    macs = []
    nics = get_nics(vm_)
    if nics is None:
        return None
    for nic in nics:
        macs.append(nic)
    return macs

def get_disks(vm_):
    if False:
        print('Hello World!')
    "\n    Return the disks of a named vm\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.get_disks <vm name>\n    "
    with _get_xapi_session() as xapi:
        disk = {}
        vm_uuid = _get_label_uuid(xapi, 'VM', vm_)
        if vm_uuid is False:
            return False
        for vbd in xapi.VM.get_VBDs(vm_uuid):
            dev = xapi.VBD.get_device(vbd)
            if not dev:
                continue
            prop = xapi.VBD.get_runtime_properties(vbd)
            disk[dev] = {'backend': prop['backend'], 'type': prop['device-type'], 'protocol': prop['protocol']}
        return disk

def setmem(vm_, memory):
    if False:
        for i in range(10):
            print('nop')
    "\n    Changes the amount of memory allocated to VM.\n\n    Memory is to be specified in MB\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.setmem myvm 768\n    "
    with _get_xapi_session() as xapi:
        mem_target = int(memory) * 1024 * 1024
        vm_uuid = _get_label_uuid(xapi, 'VM', vm_)
        if vm_uuid is False:
            return False
        try:
            xapi.VM.set_memory_dynamic_max_live(vm_uuid, mem_target)
            xapi.VM.set_memory_dynamic_min_live(vm_uuid, mem_target)
            return True
        except Exception:
            return False

def setvcpus(vm_, vcpus):
    if False:
        for i in range(10):
            print('nop')
    "\n    Changes the amount of vcpus allocated to VM.\n\n    vcpus is an int representing the number to be assigned\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.setvcpus myvm 2\n    "
    with _get_xapi_session() as xapi:
        vm_uuid = _get_label_uuid(xapi, 'VM', vm_)
        if vm_uuid is False:
            return False
        try:
            xapi.VM.set_VCPUs_number_live(vm_uuid, vcpus)
            return True
        except Exception:
            return False

def vcpu_pin(vm_, vcpu, cpus):
    if False:
        i = 10
        return i + 15
    "\n    Set which CPUs a VCPU can use.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'foo' virt.vcpu_pin domU-id 2 1\n        salt 'foo' virt.vcpu_pin domU-id 2 2-6\n    "
    with _get_xapi_session() as xapi:
        vm_uuid = _get_label_uuid(xapi, 'VM', vm_)
        if vm_uuid is False:
            return False

        def cpu_make_map(cpulist):
            if False:
                return 10
            cpus = []
            for c in cpulist.split(','):
                if c == '':
                    continue
                if '-' in c:
                    (x, y) = c.split('-')
                    for i in range(int(x), int(y) + 1):
                        cpus.append(int(i))
                elif c[0] == '^':
                    cpus = [x for x in cpus if x != int(c[1:])]
                else:
                    cpus.append(int(c))
            cpus.sort()
            return ','.join(map(str, cpus))
        if cpus == 'all':
            cpumap = cpu_make_map('0-63')
        else:
            cpumap = cpu_make_map('{}'.format(cpus))
        try:
            xapi.VM.add_to_VCPUs_params_live(vm_uuid, 'cpumap{}'.format(vcpu), cpumap)
            return True
        except Exception:
            return __salt__['cmd.run']('{} vcpu-pin {} {} {}'.format(_get_xtool(), vm_, vcpu, cpus), python_shell=False)

def freemem():
    if False:
        print('Hello World!')
    "\n    Return an int representing the amount of memory that has not been given\n    to virtual machines on this node\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.freemem\n    "
    return node_info()['free_memory']

def freecpu():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return an int representing the number of unallocated cpus on this\n    hypervisor\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.freecpu\n    "
    return node_info()['free_cpus']

def full_info():
    if False:
        i = 10
        return i + 15
    "\n    Return the node_info, vm_info and freemem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.full_info\n    "
    return {'node_info': node_info(), 'vm_info': vm_info()}

def shutdown(vm_):
    if False:
        print('Hello World!')
    "\n    Send a soft shutdown signal to the named vm\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.shutdown <vm name>\n    "
    with _get_xapi_session() as xapi:
        vm_uuid = _get_label_uuid(xapi, 'VM', vm_)
        if vm_uuid is False:
            return False
        try:
            xapi.VM.clean_shutdown(vm_uuid)
            return True
        except Exception:
            return False

def pause(vm_):
    if False:
        return 10
    "\n    Pause the named vm\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.pause <vm name>\n    "
    with _get_xapi_session() as xapi:
        vm_uuid = _get_label_uuid(xapi, 'VM', vm_)
        if vm_uuid is False:
            return False
        try:
            xapi.VM.pause(vm_uuid)
            return True
        except Exception:
            return False

def resume(vm_):
    if False:
        i = 10
        return i + 15
    "\n    Resume the named vm\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.resume <vm name>\n    "
    with _get_xapi_session() as xapi:
        vm_uuid = _get_label_uuid(xapi, 'VM', vm_)
        if vm_uuid is False:
            return False
        try:
            xapi.VM.unpause(vm_uuid)
            return True
        except Exception:
            return False

def start(config_):
    if False:
        i = 10
        return i + 15
    "\n    Start a defined domain\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.start <path to Xen cfg file>\n    "
    return __salt__['cmd.run']('{} create {}'.format(_get_xtool(), config_), python_shell=False)

def reboot(vm_):
    if False:
        for i in range(10):
            print('nop')
    "\n    Reboot a domain via ACPI request\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.reboot <vm name>\n    "
    with _get_xapi_session() as xapi:
        vm_uuid = _get_label_uuid(xapi, 'VM', vm_)
        if vm_uuid is False:
            return False
        try:
            xapi.VM.clean_reboot(vm_uuid)
            return True
        except Exception:
            return False

def reset(vm_):
    if False:
        while True:
            i = 10
    "\n    Reset a VM by emulating the reset button on a physical machine\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.reset <vm name>\n    "
    with _get_xapi_session() as xapi:
        vm_uuid = _get_label_uuid(xapi, 'VM', vm_)
        if vm_uuid is False:
            return False
        try:
            xapi.VM.hard_reboot(vm_uuid)
            return True
        except Exception:
            return False

def migrate(vm_, target, live=1, port=0, node=-1, ssl=None, change_home_server=0):
    if False:
        for i in range(10):
            print('nop')
    "\n    Migrates the virtual machine to another hypervisor\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.migrate <vm name> <target hypervisor> [live] [port] [node] [ssl] [change_home_server]\n\n    Optional values:\n\n    live\n        Use live migration\n    port\n        Use a specified port\n    node\n        Use specified NUMA node on target\n    ssl\n        use ssl connection for migration\n    change_home_server\n        change home server for managed domains\n    "
    with _get_xapi_session() as xapi:
        vm_uuid = _get_label_uuid(xapi, 'VM', vm_)
        if vm_uuid is False:
            return False
        other_config = {'port': port, 'node': node, 'ssl': ssl, 'change_home_server': change_home_server}
        try:
            xapi.VM.migrate(vm_uuid, target, bool(live), other_config)
            return True
        except Exception:
            return False

def stop(vm_):
    if False:
        for i in range(10):
            print('nop')
    "\n    Hard power down the virtual machine, this is equivalent to pulling the\n    power\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.stop <vm name>\n    "
    with _get_xapi_session() as xapi:
        vm_uuid = _get_label_uuid(xapi, 'VM', vm_)
        if vm_uuid is False:
            return False
        try:
            xapi.VM.hard_shutdown(vm_uuid)
            return True
        except Exception:
            return False

def is_hyper():
    if False:
        i = 10
        return i + 15
    "\n    Returns a bool whether or not this node is a hypervisor of any kind\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.is_hyper\n    "
    try:
        if __grains__['virtual_subtype'] != 'Xen Dom0':
            return False
    except KeyError:
        return False
    try:
        with salt.utils.files.fopen('/proc/modules') as fp_:
            if 'xen_' not in salt.utils.stringutils.to_unicode(fp_.read()):
                return False
    except OSError:
        return False
    return 'xenstore' in __salt__['cmd.run'](__grains__['ps'])

def vm_cputime(vm_=None):
    if False:
        i = 10
        return i + 15
    "\n    Return cputime used by the vms on this hyper in a\n    list of dicts:\n\n    .. code-block:: python\n\n        [\n            'your-vm': {\n                'cputime' <int>\n                'cputime_percent' <int>\n                },\n            ...\n            ]\n\n    If you pass a VM name in as an argument then it will return info\n    for just the named VM, otherwise it will return all VMs.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.vm_cputime\n    "
    with _get_xapi_session() as xapi:

        def _info(vm_):
            if False:
                print('Hello World!')
            host_rec = _get_record_by_label(xapi, 'VM', vm_)
            host_cpus = len(host_rec['host_CPUs'])
            if host_rec is False:
                return False
            host_metrics = _get_metrics_record(xapi, 'VM', host_rec)
            vcpus = int(host_metrics['VCPUs_number'])
            cputime = int(host_metrics['VCPUs_utilisation']['0'])
            cputime_percent = 0
            if cputime:
                cputime_percent = 1e-07 * cputime / host_cpus / vcpus
            return {'cputime': int(cputime), 'cputime_percent': int('{:.0f}'.format(cputime_percent))}
        info = {}
        if vm_:
            info[vm_] = _info(vm_)
            return info
        for vm_ in list_domains():
            info[vm_] = _info(vm_)
        return info

def vm_netstats(vm_=None):
    if False:
        return 10
    "\n    Return combined network counters used by the vms on this hyper in a\n    list of dicts:\n\n    .. code-block:: python\n\n        [\n            'your-vm': {\n                'io_read_kbs'           : 0,\n                'io_total_read_kbs'     : 0,\n                'io_total_write_kbs'    : 0,\n                'io_write_kbs'          : 0\n                },\n            ...\n            ]\n\n    If you pass a VM name in as an argument then it will return info\n    for just the named VM, otherwise it will return all VMs.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.vm_netstats\n    "
    with _get_xapi_session() as xapi:

        def _info(vm_):
            if False:
                while True:
                    i = 10
            ret = {}
            vm_rec = _get_record_by_label(xapi, 'VM', vm_)
            if vm_rec is False:
                return False
            for vif in vm_rec['VIFs']:
                vif_rec = _get_record(xapi, 'VIF', vif)
                ret[vif_rec['device']] = _get_metrics_record(xapi, 'VIF', vif_rec)
                del ret[vif_rec['device']]['last_updated']
            return ret
        info = {}
        if vm_:
            info[vm_] = _info(vm_)
        else:
            for vm_ in list_domains():
                info[vm_] = _info(vm_)
        return info

def vm_diskstats(vm_=None):
    if False:
        i = 10
        return i + 15
    "\n    Return disk usage counters used by the vms on this hyper in a\n    list of dicts:\n\n    .. code-block:: python\n\n        [\n            'your-vm': {\n                'io_read_kbs'   : 0,\n                'io_write_kbs'  : 0\n                },\n            ...\n            ]\n\n    If you pass a VM name in as an argument then it will return info\n    for just the named VM, otherwise it will return all VMs.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.vm_diskstats\n    "
    with _get_xapi_session() as xapi:

        def _info(vm_):
            if False:
                return 10
            ret = {}
            vm_uuid = _get_label_uuid(xapi, 'VM', vm_)
            if vm_uuid is False:
                return False
            for vbd in xapi.VM.get_VBDs(vm_uuid):
                vbd_rec = _get_record(xapi, 'VBD', vbd)
                ret[vbd_rec['device']] = _get_metrics_record(xapi, 'VBD', vbd_rec)
                del ret[vbd_rec['device']]['last_updated']
            return ret
        info = {}
        if vm_:
            info[vm_] = _info(vm_)
        else:
            for vm_ in list_domains():
                info[vm_] = _info(vm_)
        return info