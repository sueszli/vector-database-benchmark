"""
Beacon to monitor disk usage.

.. versionadded:: 2015.5.0

:depends: python-psutil
"""
import logging
import re
import salt.utils.beacons
import salt.utils.platform
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
log = logging.getLogger(__name__)
__virtualname__ = 'diskusage'

def __virtual__():
    if False:
        i = 10
        return i + 15
    if HAS_PSUTIL is False:
        err_msg = 'psutil library is missing.'
        log.error('Unable to load %s beacon: %s', __virtualname__, err_msg)
        return (False, err_msg)
    else:
        return __virtualname__

def validate(config):
    if False:
        return 10
    '\n    Validate the beacon configuration\n    '
    if not isinstance(config, list):
        return (False, 'Configuration for diskusage beacon must be a list.')
    return (True, 'Valid beacon configuration')

def beacon(config):
    if False:
        while True:
            i = 10
    '\n    Monitor the disk usage of the minion\n\n    Specify thresholds for each disk and only emit a beacon if any of them are\n    exceeded.\n\n    .. code-block:: yaml\n\n        beacons:\n          diskusage:\n            - /: 63%\n            - /mnt/nfs: 50%\n\n    Windows drives must be quoted to avoid yaml syntax errors\n\n    .. code-block:: yaml\n\n        beacons:\n          diskusage:\n            -  interval: 120\n            - \'c:\\\\\': 90%\n            - \'d:\\\\\': 50%\n\n    Regular expressions can be used as mount points.\n\n    .. code-block:: yaml\n\n        beacons:\n          diskusage:\n            - \'^\\/(?!home).*$\': 90%\n            - \'^[a-zA-Z]:\\\\$\': 50%\n\n    The first one will match all mounted disks beginning with "/", except /home\n    The second one will match disks from A:\\ to Z:\\ on a Windows system\n\n    Note that if a regular expression are evaluated after static mount points,\n    which means that if a regular expression matches another defined mount point,\n    it will override the previously defined threshold.\n\n    '
    whitelist = []
    config = salt.utils.beacons.remove_hidden_options(config, whitelist)
    parts = psutil.disk_partitions(all=True)
    ret = []
    for mounts in config:
        mount = next(iter(mounts))
        mount_re = mount
        if not mount.endswith('$'):
            mount_re = '{}$'.format(mount)
        if salt.utils.platform.is_windows():
            mount_re = re.sub(':\\\\\\$', ':\\\\\\\\', mount_re)
            mount_re = re.sub(':\\\\\\\\\\$', ':\\\\\\\\', mount_re)
            mount_re = mount_re.upper()
        for part in parts:
            if re.match(mount_re, part.mountpoint):
                _mount = part.mountpoint
                try:
                    _current_usage = psutil.disk_usage(_mount)
                except OSError:
                    log.warning('%s is not a valid mount point.', _mount)
                    continue
                current_usage = _current_usage.percent
                monitor_usage = mounts[mount]
                if isinstance(monitor_usage, str) and '%' in monitor_usage:
                    monitor_usage = re.sub('%', '', monitor_usage)
                monitor_usage = float(monitor_usage)
                if current_usage >= monitor_usage:
                    ret.append({'diskusage': current_usage, 'mount': _mount})
    return ret