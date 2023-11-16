"""
virst compatibility module for managing VMs on SmartOS
"""
import logging
import salt.utils.path
import salt.utils.platform
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__virtualname__ = 'virt'

def __virtual__():
    if False:
        return 10
    '\n    Provides virt on SmartOS\n    '
    if salt.utils.platform.is_smartos_globalzone() and salt.utils.path.which('vmadm'):
        return __virtualname__
    return (False, '{} module can only be loaded on SmartOS compute nodes'.format(__virtualname__))

def init(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Initialize a new VM\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.init image_uuid='...' alias='...' [...]\n    "
    return __salt__['vmadm.create'](**kwargs)

def list_domains():
    if False:
        while True:
            i = 10
    "\n    Return a list of virtual machine names on the minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.list_domains\n    "
    data = __salt__['vmadm.list'](keyed=True)
    vms = ['UUID                                  TYPE  RAM      STATE             ALIAS']
    for vm in data:
        vms.append('{vmuuid}{vmtype}{vmram}{vmstate}{vmalias}'.format(vmuuid=vm.ljust(38), vmtype=data[vm]['type'].ljust(6), vmram=data[vm]['ram'].ljust(9), vmstate=data[vm]['state'].ljust(18), vmalias=data[vm]['alias']))
    return vms

def list_active_vms():
    if False:
        print('Hello World!')
    "\n    Return a list of uuids for active virtual machine on the minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.list_active_vms\n    "
    return __salt__['vmadm.list'](search="state='running'", order='uuid')

def list_inactive_vms():
    if False:
        print('Hello World!')
    "\n    Return a list of uuids for inactive virtual machine on the minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.list_inactive_vms\n    "
    return __salt__['vmadm.list'](search="state='stopped'", order='uuid')

def vm_info(domain):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a dict with information about the specified VM on this CN\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.vm_info <domain>\n    "
    return __salt__['vmadm.get'](domain)

def start(domain):
    if False:
        for i in range(10):
            print('nop')
    "\n    Start a defined domain\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.start <domain>\n    "
    if domain in list_active_vms():
        raise CommandExecutionError('The specified vm is already running')
    __salt__['vmadm.start'](domain)
    return domain in list_active_vms()

def shutdown(domain):
    if False:
        while True:
            i = 10
    "\n    Send a soft shutdown signal to the named vm\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.shutdown <domain>\n    "
    if domain in list_inactive_vms():
        raise CommandExecutionError('The specified vm is already stopped')
    __salt__['vmadm.stop'](domain)
    return domain in list_inactive_vms()

def reboot(domain):
    if False:
        return 10
    "\n    Reboot a domain via ACPI request\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.reboot <domain>\n    "
    if domain in list_inactive_vms():
        raise CommandExecutionError('The specified vm is stopped')
    __salt__['vmadm.reboot'](domain)
    return domain in list_active_vms()

def stop(domain):
    if False:
        while True:
            i = 10
    "\n    Hard power down the virtual machine, this is equivalent to powering off the hardware.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.destroy <domain>\n    "
    if domain in list_inactive_vms():
        raise CommandExecutionError('The specified vm is stopped')
    return __salt__['vmadm.delete'](domain)

def vm_virt_type(domain):
    if False:
        i = 10
        return i + 15
    "\n    Return VM virtualization type : OS or KVM\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.vm_virt_type <domain>\n    "
    ret = __salt__['vmadm.lookup'](search='uuid={uuid}'.format(uuid=domain), order='type')
    if not ret:
        raise CommandExecutionError("We can't determine the type of this VM")
    return ret[0]['type']

def setmem(domain, memory):
    if False:
        print('Hello World!')
    "\n    Change the amount of memory allocated to VM.\n    <memory> is to be specified in MB.\n\n    Note for KVM : this would require a restart of the VM.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.setmem <domain> 512\n    "
    vmtype = vm_virt_type(domain)
    if vmtype == 'OS':
        return __salt__['vmadm.update'](vm=domain, max_physical_memory=memory)
    elif vmtype == 'LX':
        return __salt__['vmadm.update'](vm=domain, max_physical_memory=memory)
    elif vmtype == 'KVM':
        log.warning('Changes will be applied after the VM restart.')
        return __salt__['vmadm.update'](vm=domain, ram=memory)
    else:
        raise CommandExecutionError('Unknown VM type')
    return False

def get_macs(domain):
    if False:
        while True:
            i = 10
    "\n    Return a list off MAC addresses from the named VM\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' virt.get_macs <domain>\n    "
    macs = []
    ret = __salt__['vmadm.lookup'](search='uuid={uuid}'.format(uuid=domain), order='nics')
    if not ret:
        raise CommandExecutionError("We can't find the MAC address of this VM")
    else:
        for nic in ret[0]['nics']:
            macs.append(nic['mac'])
        return macs