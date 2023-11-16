"""
Module for Solaris 10's zoneadm

:maintainer:    Jorge Schrauwen <sjorge@blackdot.be>
:maturity:      new
:platform:      OmniOS,OpenIndiana,SmartOS,OpenSolaris,Solaris 10

.. versionadded:: 2017.7.0

.. warning::
    Oracle Solaris 11's zoneadm is not supported by this module!
"""
import logging
import salt.utils.decorators
import salt.utils.path
log = logging.getLogger(__name__)
__virtualname__ = 'zoneadm'
__func_alias__ = {'list_zones': 'list'}

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

def _is_uuid(zone):
    if False:
        i = 10
        return i + 15
    '\n    Check if zone is actually a UUID\n    '
    return len(zone) == 36 and zone.index('-') == 8

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    We are available if we are have zoneadm and are the global zone on\n    Solaris 10, OmniOS, OpenIndiana, OpenSolaris, or Smartos.\n    '
    if _is_globalzone() and salt.utils.path.which('zoneadm'):
        if __grains__['os'] in ['OpenSolaris', 'SmartOS', 'OmniOS', 'OpenIndiana']:
            return __virtualname__
        elif __grains__['os'] == 'Oracle Solaris' and int(__grains__['osmajorrelease']) == 10:
            return __virtualname__
    return (False, f'{__virtualname__} module can only be loaded in a solaris globalzone.')

def list_zones(verbose=True, installed=False, configured=False, hide_global=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    List all zones\n\n    verbose : boolean\n        display additional zone information\n    installed : boolean\n        include installed zones in output\n    configured : boolean\n        include configured zones in output\n    hide_global : boolean\n        do not include global zone\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zoneadm.list\n    "
    zones = {}
    header = 'zoneid:zonename:state:zonepath:uuid:brand:ip-type'.split(':')
    zone_data = __salt__['cmd.run_all']('zoneadm list -p -c')
    if zone_data['retcode'] == 0:
        for zone in zone_data['stdout'].splitlines():
            zone = zone.split(':')
            zone_t = {}
            for (num, val) in enumerate(header):
                zone_t[val] = zone[num]
            if hide_global and zone_t['zonename'] == 'global':
                continue
            if not installed and zone_t['state'] == 'installed':
                continue
            if not configured and zone_t['state'] == 'configured':
                continue
            zones[zone_t['zonename']] = zone_t
            del zones[zone_t['zonename']]['zonename']
    return zones if verbose else sorted(zones.keys())

def boot(zone, single=False, altinit=None, smf_options=None):
    if False:
        print('Hello World!')
    "\n    Boot (or activate) the specified zone.\n\n    zone : string\n        name or uuid of the zone\n    single : boolean\n        boots only to milestone svc:/milestone/single-user:default.\n    altinit : string\n        valid path to an alternative executable to be the primordial process.\n    smf_options : string\n        include two categories of options to control booting behavior of\n        the service management facility: recovery options and messages options.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zoneadm.boot clementine\n        salt '*' zoneadm.boot maeve single=True\n        salt '*' zoneadm.boot teddy single=True smf_options=verbose\n    "
    ret = {'status': True}
    boot_options = ''
    if single:
        boot_options = f'-s {boot_options}'
    if altinit:
        boot_options = f'-i {altinit} {boot_options}'
    if smf_options:
        boot_options = f'-m {smf_options} {boot_options}'
    if boot_options != '':
        boot_options = f' -- {boot_options.strip()}'
    res = __salt__['cmd.run_all']('zoneadm {zone} boot{boot_opts}'.format(zone=f'-u {zone}' if _is_uuid(zone) else f'-z {zone}', boot_opts=boot_options))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    ret['message'] = ret['message'].replace('zoneadm: ', '')
    if ret['message'] == '':
        del ret['message']
    return ret

def reboot(zone, single=False, altinit=None, smf_options=None):
    if False:
        print('Hello World!')
    "\n    Restart the zone. This is equivalent to a halt boot sequence.\n\n    zone : string\n        name or uuid of the zone\n    single : boolean\n        boots only to milestone svc:/milestone/single-user:default.\n    altinit : string\n        valid path to an alternative executable to be the primordial process.\n    smf_options : string\n        include two categories of options to control booting behavior of\n        the service management facility: recovery options and messages options.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zoneadm.reboot dolores\n        salt '*' zoneadm.reboot teddy single=True\n    "
    ret = {'status': True}
    boot_options = ''
    if single:
        boot_options = f'-s {boot_options}'
    if altinit:
        boot_options = f'-i {altinit} {boot_options}'
    if smf_options:
        boot_options = f'-m {smf_options} {boot_options}'
    if boot_options != '':
        boot_options = f' -- {boot_options.strip()}'
    res = __salt__['cmd.run_all']('zoneadm {zone} reboot{boot_opts}'.format(zone=f'-u {zone}' if _is_uuid(zone) else f'-z {zone}', boot_opts=boot_options))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    ret['message'] = ret['message'].replace('zoneadm: ', '')
    if ret['message'] == '':
        del ret['message']
    return ret

def halt(zone):
    if False:
        for i in range(10):
            print('nop')
    "\n    Halt the specified zone.\n\n    zone : string\n        name or uuid of the zone\n\n    .. note::\n        To cleanly shutdown the zone use the shutdown function.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zoneadm.halt hector\n    "
    ret = {'status': True}
    res = __salt__['cmd.run_all']('zoneadm {zone} halt'.format(zone=f'-u {zone}' if _is_uuid(zone) else f'-z {zone}'))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    ret['message'] = ret['message'].replace('zoneadm: ', '')
    if ret['message'] == '':
        del ret['message']
    return ret

def shutdown(zone, reboot=False, single=False, altinit=None, smf_options=None):
    if False:
        return 10
    "\n    Gracefully shutdown the specified zone.\n\n    zone : string\n        name or uuid of the zone\n    reboot : boolean\n        reboot zone after shutdown (equivalent of shutdown -i6 -g0 -y)\n    single : boolean\n        boots only to milestone svc:/milestone/single-user:default.\n    altinit : string\n        valid path to an alternative executable to be the primordial process.\n    smf_options : string\n        include two categories of options to control booting behavior of\n        the service management facility: recovery options and messages options.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zoneadm.shutdown peter\n        salt '*' zoneadm.shutdown armistice reboot=True\n    "
    ret = {'status': True}
    boot_options = ''
    if single:
        boot_options = f'-s {boot_options}'
    if altinit:
        boot_options = f'-i {altinit} {boot_options}'
    if smf_options:
        boot_options = f'-m {smf_options} {boot_options}'
    if boot_options != '':
        boot_options = f' -- {boot_options.strip()}'
    res = __salt__['cmd.run_all']('zoneadm {zone} shutdown{reboot}{boot_opts}'.format(zone=f'-u {zone}' if _is_uuid(zone) else f'-z {zone}', reboot=' -r' if reboot else '', boot_opts=boot_options))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    ret['message'] = ret['message'].replace('zoneadm: ', '')
    if ret['message'] == '':
        del ret['message']
    return ret

def detach(zone):
    if False:
        i = 10
        return i + 15
    "\n    Detach the specified zone.\n\n    zone : string\n        name or uuid of the zone\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zoneadm.detach kissy\n    "
    ret = {'status': True}
    res = __salt__['cmd.run_all']('zoneadm {zone} detach'.format(zone=f'-u {zone}' if _is_uuid(zone) else f'-z {zone}'))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    ret['message'] = ret['message'].replace('zoneadm: ', '')
    if ret['message'] == '':
        del ret['message']
    return ret

def attach(zone, force=False, brand_opts=None):
    if False:
        print('Hello World!')
    '\n    Attach the specified zone.\n\n    zone : string\n        name of the zone\n    force : boolean\n        force the zone into the "installed" state with no validation\n    brand_opts : string\n        brand specific options to pass\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' zoneadm.attach lawrence\n        salt \'*\' zoneadm.attach lawrence True\n    '
    ret = {'status': True}
    res = __salt__['cmd.run_all']('zoneadm -z {zone} attach{force}{brand_opts}'.format(zone=zone, force=' -F' if force else '', brand_opts=f' {brand_opts}' if brand_opts else ''))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    ret['message'] = ret['message'].replace('zoneadm: ', '')
    if ret['message'] == '':
        del ret['message']
    return ret

def ready(zone):
    if False:
        print('Hello World!')
    "\n    Prepares a zone for running applications.\n\n    zone : string\n        name or uuid of the zone\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zoneadm.ready clementine\n    "
    ret = {'status': True}
    res = __salt__['cmd.run_all']('zoneadm {zone} ready'.format(zone=f'-u {zone}' if _is_uuid(zone) else f'-z {zone}'))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    ret['message'] = ret['message'].replace('zoneadm: ', '')
    if ret['message'] == '':
        del ret['message']
    return ret

def verify(zone):
    if False:
        return 10
    "\n    Check to make sure the configuration of the specified\n    zone can safely be installed on the machine.\n\n    zone : string\n        name of the zone\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zoneadm.verify dolores\n    "
    ret = {'status': True}
    res = __salt__['cmd.run_all']('zoneadm -z {zone} verify'.format(zone=zone))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    ret['message'] = ret['message'].replace('zoneadm: ', '')
    if ret['message'] == '':
        del ret['message']
    return ret

def move(zone, zonepath):
    if False:
        i = 10
        return i + 15
    "\n    Move zone to new zonepath.\n\n    zone : string\n        name or uuid of the zone\n    zonepath : string\n        new zonepath\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zoneadm.move meave /sweetwater/meave\n    "
    ret = {'status': True}
    res = __salt__['cmd.run_all']('zoneadm {zone} move {path}'.format(zone=f'-u {zone}' if _is_uuid(zone) else f'-z {zone}', path=zonepath))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    ret['message'] = ret['message'].replace('zoneadm: ', '')
    if ret['message'] == '':
        del ret['message']
    return ret

def uninstall(zone):
    if False:
        while True:
            i = 10
    "\n    Uninstall the specified zone from the system.\n\n    zone : string\n        name or uuid of the zone\n\n    .. warning::\n        The -F flag is always used to avoid the prompts when uninstalling.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zoneadm.uninstall teddy\n    "
    ret = {'status': True}
    res = __salt__['cmd.run_all']('zoneadm {zone} uninstall -F'.format(zone=f'-u {zone}' if _is_uuid(zone) else f'-z {zone}'))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    ret['message'] = ret['message'].replace('zoneadm: ', '')
    if ret['message'] == '':
        del ret['message']
    return ret

def install(zone, nodataset=False, brand_opts=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Install the specified zone from the system.\n\n    zone : string\n        name of the zone\n    nodataset : boolean\n        do not create a ZFS file system\n    brand_opts : string\n        brand specific options to pass\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zoneadm.install dolores\n        salt '*' zoneadm.install teddy True\n    "
    ret = {'status': True}
    res = __salt__['cmd.run_all']('zoneadm -z {zone} install{nodataset}{brand_opts}'.format(zone=zone, nodataset=' -x nodataset' if nodataset else '', brand_opts=f' {brand_opts}' if brand_opts else ''))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    ret['message'] = ret['message'].replace('zoneadm: ', '')
    if ret['message'] == '':
        del ret['message']
    return ret

def clone(zone, source, snapshot=None):
    if False:
        i = 10
        return i + 15
    "\n    Install a zone by copying an existing installed zone.\n\n    zone : string\n        name of the zone\n    source : string\n        zone to clone from\n    snapshot : string\n        optional name of snapshot to use as source\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zoneadm.clone clementine dolores\n    "
    ret = {'status': True}
    res = __salt__['cmd.run_all']('zoneadm -z {zone} clone {snapshot}{source}'.format(zone=zone, source=source, snapshot=f'-s {snapshot} ' if snapshot else ''))
    ret['status'] = res['retcode'] == 0
    ret['message'] = res['stdout'] if ret['status'] else res['stderr']
    ret['message'] = ret['message'].replace('zoneadm: ', '')
    if ret['message'] == '':
        del ret['message']
    return ret