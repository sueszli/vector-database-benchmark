"""
Execution module to upgrade Cisco NX-OS Switches.

.. versionadded:: 3001

This module supports execution using a Proxy Minion or Native Minion:
    1) Proxy Minion: Connect over SSH or NX-API HTTP(S).
       See :mod:`salt.proxy.nxos <salt.proxy.nxos>` for proxy minion setup details.
    2) Native Minion: Connect over NX-API Unix Domain Socket (UDS).
       Install the minion inside the GuestShell running on the NX-OS device.

:maturity:   new
:platform:   nxos
:codeauthor: Michael G Wiebe

.. note::

    To use this module over remote NX-API the feature must be enabled on the
    NX-OS device by executing ``feature nxapi`` in configuration mode.

    This is not required for NX-API over UDS.

    Configuration example:

    .. code-block:: bash

        switch# conf t
        switch(config)# feature nxapi

    To check that NX-API is properly enabled, execute ``show nxapi``.

    Output example:

    .. code-block:: bash

        switch# show nxapi
        nxapi enabled
        HTTPS Listen on port 443
"""
import ast
import logging
import re
import time
from salt.exceptions import CommandExecutionError, NxosError
__virtualname__ = 'nxos'
__virtual_aliases__ = ('nxos_upgrade',)
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        print('Hello World!')
    return __virtualname__

def check_upgrade_impact(system_image, kickstart_image=None, issu=True, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Display upgrade impact information without actually upgrading the device.\n\n    system_image (Mandatory Option)\n        Path on bootflash: to system image upgrade file.\n\n    kickstart_image\n        Path on bootflash: to kickstart image upgrade file.\n        (Not required if using combined system/kickstart image file)\n        Default: None\n\n    issu\n        In Service Software Upgrade (non-disruptive). When True,\n        the upgrade will abort if issu is not possible.\n        When False: Force (disruptive) Upgrade/Downgrade.\n        Default: True\n\n    timeout\n        Timeout in seconds for long running 'install all' impact command.\n        Default: 900\n\n    error_pattern\n        Use the option to pass in a regular expression to search for in the\n        output of the 'install all impact' command that indicates an error\n        has occurred.  This option is only used when proxy minion connection\n        type is ssh and otherwise ignored.\n\n    .. code-block:: bash\n\n        salt 'n9k' nxos.check_upgrade_impact system_image=nxos.9.2.1.bin\n        salt 'n7k' nxos.check_upgrade_impact system_image=n7000-s2-dk9.8.1.1.bin \\\n            kickstart_image=n7000-s2-kickstart.8.1.1.bin issu=False\n    "
    if not isinstance(issu, bool):
        return 'Input Error: The [issu] parameter must be either True or False'
    si = system_image
    ki = kickstart_image
    dev = 'bootflash'
    cmd = 'terminal dont-ask ; show install all impact'
    if ki is not None:
        cmd = cmd + ' kickstart {0}:{1} system {0}:{2}'.format(dev, ki, si)
    else:
        cmd = cmd + ' nxos {}:{}'.format(dev, si)
    if issu and ki is None:
        cmd = cmd + ' non-disruptive'
    log.info("Check upgrade impact using command: '%s'", cmd)
    kwargs.update({'timeout': kwargs.get('timeout', 900)})
    error_pattern_list = ['Another install procedure may be in progress', 'Pre-upgrade check failed']
    kwargs.update({'error_pattern': error_pattern_list})
    try:
        impact_check = __salt__['nxos.sendline'](cmd, **kwargs)
    except CommandExecutionError as e:
        impact_check = ast.literal_eval(e.message)
    return _parse_upgrade_data(impact_check)

def upgrade(system_image, kickstart_image=None, issu=True, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Upgrade NX-OS switch.\n\n    system_image (Mandatory Option)\n        Path on bootflash: to system image upgrade file.\n\n    kickstart_image\n        Path on bootflash: to kickstart image upgrade file.\n        (Not required if using combined system/kickstart image file)\n        Default: None\n\n    issu\n        Set this option to True when an In Service Software Upgrade or\n        non-disruptive upgrade is required. The upgrade will abort if issu is\n        not possible.\n        Default: True\n\n    timeout\n        Timeout in seconds for long running 'install all' upgrade command.\n        Default: 900\n\n    error_pattern\n        Use the option to pass in a regular expression to search for in the\n        output of the 'install all upgrade command that indicates an error\n        has occurred.  This option is only used when proxy minion connection\n        type is ssh and otherwise ignored.\n\n    .. code-block:: bash\n\n        salt 'n9k' nxos.upgrade system_image=nxos.9.2.1.bin\n        salt 'n7k' nxos.upgrade system_image=n7000-s2-dk9.8.1.1.bin \\\n            kickstart_image=n7000-s2-kickstart.8.1.1.bin issu=False\n    "
    if not isinstance(issu, bool):
        return 'Input Error: The [issu] parameter must be either True or False'
    impact = None
    upgrade = None
    maxtry = 60
    for attempt in range(1, maxtry):
        if impact is None:
            log.info('Gathering impact data')
            impact = check_upgrade_impact(system_image, kickstart_image, issu, **kwargs)
            if impact['installing']:
                log.info('Another show impact in progress... wait and retry')
                time.sleep(30)
                continue
            if impact['invalid_command']:
                impact = False
                continue
            log.info('Impact data gathered:\n%s', impact)
            if impact['error_data']:
                return impact
            if issu and (not impact['upgrade_non_disruptive']):
                impact['error_data'] = impact['upgrade_data']
                return impact
            if not impact['succeeded'] and (not impact['module_data']):
                impact['error_data'] = impact['upgrade_data']
                return impact
            if not impact['upgrade_required']:
                impact['succeeded'] = True
                return impact
        upgrade = _upgrade(system_image, kickstart_image, issu, **kwargs)
        if upgrade['installing']:
            log.info('Another install is in progress... wait and retry')
            time.sleep(30)
            continue
        if upgrade['invalid_command']:
            log_msg = 'The [issu] option was set to False for this upgrade.'
            log_msg = log_msg + ' Attempt was made to ugrade using the force'
            log_msg = log_msg + ' keyword which is not supported in this'
            log_msg = log_msg + ' image.  Set [issu=True] and re-try.'
            upgrade['upgrade_data'] = log_msg
            break
        break
    if upgrade['backend_processing_error']:
        impact['upgrade_in_progress'] = True
        return impact
    return upgrade

def _upgrade(system_image, kickstart_image, issu, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Helper method that does the heavy lifting for upgrades.\n    '
    si = system_image
    ki = kickstart_image
    dev = 'bootflash'
    cmd = 'terminal dont-ask ; install all'
    if ki is None:
        logmsg = 'Upgrading device using combined system/kickstart image.'
        logmsg += '\nSystem Image: {}'.format(si)
        cmd = cmd + ' nxos {}:{}'.format(dev, si)
        if issu:
            cmd = cmd + ' non-disruptive'
    else:
        logmsg = 'Upgrading device using separate system/kickstart images.'
        logmsg += '\nSystem Image: {}'.format(si)
        logmsg += '\nKickstart Image: {}'.format(ki)
        if not issu:
            log.info('Attempting upgrade using force option')
            cmd = cmd + ' force'
        cmd = cmd + ' kickstart {0}:{1} system {0}:{2}'.format(dev, ki, si)
    if issu:
        logmsg += '\nIn Service Software Upgrade/Downgrade (non-disruptive) requested.'
    else:
        logmsg += '\nDisruptive Upgrade/Downgrade requested.'
    log.info(logmsg)
    log.info("Begin upgrade using command: '%s'", cmd)
    kwargs.update({'timeout': kwargs.get('timeout', 900)})
    error_pattern_list = ['Another install procedure may be in progress']
    kwargs.update({'error_pattern': error_pattern_list})
    try:
        upgrade_result = __salt__['nxos.sendline'](cmd, **kwargs)
    except CommandExecutionError as e:
        upgrade_result = ast.literal_eval(e.message)
    except NxosError as e:
        if re.search('Code: 500', e.message):
            upgrade_result = e.message
        else:
            upgrade_result = ast.literal_eval(e.message)
    return _parse_upgrade_data(upgrade_result)

def _parse_upgrade_data(data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper method to parse upgrade data from the NX-OS device.\n    '
    upgrade_result = {}
    upgrade_result['upgrade_data'] = None
    upgrade_result['succeeded'] = False
    upgrade_result['upgrade_required'] = False
    upgrade_result['upgrade_non_disruptive'] = False
    upgrade_result['upgrade_in_progress'] = False
    upgrade_result['installing'] = False
    upgrade_result['module_data'] = {}
    upgrade_result['error_data'] = None
    upgrade_result['backend_processing_error'] = False
    upgrade_result['invalid_command'] = False
    if isinstance(data, str) and re.search('Code: 500', data):
        log.info('Detected backend processing error')
        upgrade_result['error_data'] = data
        upgrade_result['backend_processing_error'] = True
        return upgrade_result
    if isinstance(data, dict):
        if 'code' in data and data['code'] == '400':
            log.info('Detected client error')
            upgrade_result['error_data'] = data['cli_error']
            if re.search('install.*may be in progress', data['cli_error']):
                log.info('Detected install in progress...')
                upgrade_result['installing'] = True
            if re.search('Invalid command', data['cli_error']):
                log.info('Detected invalid command...')
                upgrade_result['invalid_command'] = True
        else:
            log.info('Probable backend processing error')
            upgrade_result['backend_processing_error'] = True
        return upgrade_result
    if isinstance(data, list) and len(data) == 2:
        data = data[1]
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    log.info('Parsing NX-OS upgrade data')
    upgrade_result['upgrade_data'] = data
    for line in data.split('\n'):
        log.info('Processing line: (%s)', line)
        if re.search('non-disruptive', line):
            log.info('Found non-disruptive line')
            upgrade_result['upgrade_non_disruptive'] = True
        mo = re.search('(\\d+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(yes|no)', line)
        if mo:
            log.info('Matched Module Running/New Version Upg-Req Line')
            bk = 'module_data'
            g1 = mo.group(1)
            g2 = mo.group(2)
            g3 = mo.group(3)
            g4 = mo.group(4)
            g5 = mo.group(5)
            mk = 'module {}:image {}'.format(g1, g2)
            upgrade_result[bk][mk] = {}
            upgrade_result[bk][mk]['running_version'] = g3
            upgrade_result[bk][mk]['new_version'] = g4
            if g5 == 'yes':
                upgrade_result['upgrade_required'] = True
                upgrade_result[bk][mk]['upgrade_required'] = True
            continue
        if re.search('Install has been successful', line):
            log.info('Install successful line')
            upgrade_result['succeeded'] = True
            continue
        if re.search('Finishing the upgrade, switch will reboot in', line):
            log.info('Finishing upgrade line')
            upgrade_result['upgrade_in_progress'] = True
            continue
        if re.search('Switch will be reloaded for disruptive upgrade', line):
            log.info('Switch will be reloaded line')
            upgrade_result['upgrade_in_progress'] = True
            continue
        if re.search('Switching over onto standby', line):
            log.info('Switching over onto standby line')
            upgrade_result['upgrade_in_progress'] = True
            continue
    return upgrade_result