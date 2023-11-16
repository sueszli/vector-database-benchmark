"""

    Copyright (C) 2013-2014 Team-XBMC
    Copyright (C) 2014-2019 Team Kodi

    This file is part of service.xbmc.versioncheck

    SPDX-License-Identifier: GPL-3.0-or-later
    See LICENSES/GPL-3.0-or-later.txt for more information.

"""
import sys
from .common import log
from .handler import Handler
try:
    from subprocess import check_output
except ImportError:
    check_output = None
    log('ImportError: subprocess')

class ShellHandlerApt(Handler):
    """ Apt shell handler
    """

    def __init__(self, use_sudo=False):
        if False:
            print('Hello World!')
        Handler.__init__(self)
        self.sudo = use_sudo
        self._update = False
        (installed, _) = self._check_versions('kodi')
        if not installed:
            log('No installed package found, exiting')
            sys.exit(0)
        self._update = True

    def _check_versions(self, package):
        if False:
            i = 10
            return i + 15
        ' Check apt package versions\n\n        :param package: package to check\n        :type package: str\n        :return: installed version, candidate version\n        :rtype: str, str / False, False\n        '
        _cmd = 'apt-cache policy ' + package
        if self.update and (not self._update_cache()):
            return (False, False)
        try:
            result = check_output([_cmd], shell=True).split('\n')
        except Exception as error:
            log('ShellHandlerApt: exception while executing shell command %s: %s' % (_cmd, error))
            return (False, False)
        if result[0].replace(':', '') == package:
            installed = result[1].split()[1]
            candidate = result[2].split()[1]
            if installed == '(none)':
                installed = False
            if candidate == '(none)':
                candidate = False
            return (installed, candidate)
        log('ShellHandlerApt: error during version check')
        return (False, False)

    def _update_cache(self):
        if False:
            return 10
        ' Update apt cache\n\n        :return: success of updating apt cache\n        :rtype: bool\n        '
        _cmd = 'apt-get update'
        try:
            if self.sudo:
                _ = check_output("echo '%s' | sudo -S %s" % (self._get_password(), _cmd), shell=True)
            else:
                _ = check_output(_cmd.split())
        except Exception as error:
            log('Exception while executing shell command %s: %s' % (_cmd, error))
            return False
        return True

    def upgrade_package(self, package):
        if False:
            print('Hello World!')
        ' Upgrade apt package\n\n        :param package: package to upgrade\n        :type package: str\n        :return: success of apt package upgrade\n        :rtype: bool\n        '
        _cmd = 'apt-get install -y ' + package
        try:
            if self.sudo:
                _ = check_output("echo '%s' | sudo -S %s" % (self._get_password(), _cmd), shell=True)
            else:
                _ = check_output(_cmd.split())
            log('Upgrade successful')
        except Exception as error:
            log('Exception while executing shell command %s: %s' % (_cmd, error))
            return False
        return True

    def upgrade_system(self):
        if False:
            while True:
                i = 10
        ' Upgrade system\n\n        :return: success of system upgrade\n        :rtype: bool\n        '
        _cmd = 'apt-get upgrade -y'
        try:
            log('Upgrading system')
            if self.sudo:
                _ = check_output("echo '%s' | sudo -S %s" % (self._get_password(), _cmd), shell=True)
            else:
                _ = check_output(_cmd.split())
        except Exception as error:
            log('Exception while executing shell command %s: %s' % (_cmd, error))
            return False
        return True