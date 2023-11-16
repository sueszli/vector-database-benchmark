"""
Grains for Fibre Channel WWN's. On Windows this runs a PowerShell command that
queries WMI to get the Fibre Channel WWN's available.

.. versionadded:: 2018.3.0

To enable these grains set ``fibre_channel_grains: True`` in the minion config.

.. code-block:: yaml

    fibre_channel_grains: True
"""
import glob
import logging
import salt.modules.cmdmod
import salt.utils.files
import salt.utils.platform
__virtualname__ = 'fibre_channel'
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    if __opts__.get('fibre_channel_grains', False) is False:
        return False
    else:
        return __virtualname__

def _linux_wwns():
    if False:
        print('Hello World!')
    '\n    Return Fibre Channel port WWNs from a Linux host.\n    '
    ret = []
    for fc_file in glob.glob('/sys/class/fc_host/*/port_name'):
        with salt.utils.files.fopen(fc_file, 'r') as _wwn:
            content = _wwn.read()
            for line in content.splitlines():
                ret.append(line.rstrip()[2:])
    return ret

def _windows_wwns():
    if False:
        print('Hello World!')
    '\n    Return Fibre Channel port WWNs from a Windows host.\n    '
    ps_cmd = 'Get-WmiObject -ErrorAction Stop -class MSFC_FibrePortHBAAttributes -namespace "root\\WMI" | Select -Expandproperty Attributes | %{($_.PortWWN | % {"{0:x2}" -f $_}) -join ""}'
    ret = []
    cmd_ret = salt.modules.cmdmod.powershell(ps_cmd)
    for line in cmd_ret:
        ret.append(line.rstrip())
    return ret

def fibre_channel_wwns():
    if False:
        while True:
            i = 10
    '\n    Return list of fiber channel HBA WWNs\n    '
    grains = {'fc_wwn': False}
    if salt.utils.platform.is_linux():
        grains['fc_wwn'] = _linux_wwns()
    elif salt.utils.platform.is_windows():
        grains['fc_wwn'] = _windows_wwns()
    return grains