"""
Manage and query udev info

.. versionadded:: 2015.8.0

"""
import logging
import salt.modules.cmdmod
import salt.utils.path
from salt.exceptions import CommandExecutionError
__salt__ = {'cmd.run_all': salt.modules.cmdmod.run_all}
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only work when udevadm is installed.\n    '
    return salt.utils.path.which_bin(['udevadm']) is not None

def _parse_udevadm_info(udev_info):
    if False:
        i = 10
        return i + 15
    '\n    Parse the info returned by udevadm command.\n    '
    devices = []
    dev = {}
    for line in (line.strip() for line in udev_info.splitlines()):
        if line:
            line = line.split(':', 1)
            if len(line) != 2:
                continue
            (query, data) = line
            if query == 'E':
                if query not in dev:
                    dev[query] = {}
                (key, val) = data.strip().split('=', 1)
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                dev[query][key] = val
            else:
                if query not in dev:
                    dev[query] = []
                dev[query].append(data.strip())
        elif dev:
            devices.append(_normalize_info(dev))
            dev = {}
    if dev:
        _normalize_info(dev)
        devices.append(_normalize_info(dev))
    return devices

def _normalize_info(dev):
    if False:
        return 10
    '\n    Replace list with only one element to the value of the element.\n\n    :param dev:\n    :return:\n    '
    for (sect, val) in dev.items():
        if len(val) == 1:
            dev[sect] = val[0]
    return dev

def info(dev):
    if False:
        print('Hello World!')
    "\n    Extract all info delivered by udevadm\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' udev.info /dev/sda\n        salt '*' udev.info /sys/class/net/eth0\n    "
    if 'sys' in dev:
        qtype = 'path'
    else:
        qtype = 'name'
    cmd = 'udevadm info --export --query=all --{}={}'.format(qtype, dev)
    udev_result = __salt__['cmd.run_all'](cmd, output_loglevel='quiet')
    if udev_result['retcode'] != 0:
        raise CommandExecutionError(udev_result['stderr'])
    return _parse_udevadm_info(udev_result['stdout'])[0]

def env(dev):
    if False:
        i = 10
        return i + 15
    "\n    Return all environment variables udev has for dev\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' udev.env /dev/sda\n        salt '*' udev.env /sys/class/net/eth0\n    "
    return info(dev).get('E', None)

def name(dev):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the actual dev name(s?) according to udev for dev\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' udev.dev /dev/sda\n        salt '*' udev.dev /sys/class/net/eth0\n    "
    return info(dev).get('N', None)

def path(dev):
    if False:
        i = 10
        return i + 15
    "\n    Return the physical device path(s?) according to udev for dev\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' udev.path /dev/sda\n        salt '*' udev.path /sys/class/net/eth0\n    "
    return info(dev).get('P', None)

def links(dev):
    if False:
        return 10
    "\n    Return all udev-created device symlinks\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' udev.links /dev/sda\n        salt '*' udev.links /sys/class/net/eth0\n    "
    return info(dev).get('S', None)

def exportdb():
    if False:
        return 10
    "\n    Return all the udev database\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' udev.exportdb\n    "
    cmd = 'udevadm info --export-db'
    udev_result = __salt__['cmd.run_all'](cmd, output_loglevel='quiet')
    if udev_result['retcode']:
        raise CommandExecutionError(udev_result['stderr'])
    return _parse_udevadm_info(udev_result['stdout'])