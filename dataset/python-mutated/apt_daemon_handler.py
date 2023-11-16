"""

    Copyright (C) 2013-2014 Team-XBMC
    Copyright (C) 2014-2019 Team Kodi

    This file is part of service.xbmc.versioncheck

    SPDX-License-Identifier: GPL-3.0-or-later
    See LICENSES/GPL-3.0-or-later.txt for more information.

"""
from .common import log
from .handler import Handler
try:
    import apt
    from aptdaemon import client
    from aptdaemon import errors
except ImportError:
    apt = None
    client = None
    errors = None
    log('ImportError: apt, aptdaemon')

class AptDaemonHandler(Handler):
    """ Apt daemon handler
    """

    def __init__(self):
        if False:
            print('Hello World!')
        Handler.__init__(self)
        self.apt_client = client.AptClient()

    def _check_versions(self, package):
        if False:
            for i in range(10):
                print('nop')
        ' Check apt package versions\n\n        :param package: package to check\n        :type package: str\n        :return: installed version, candidate version\n        :rtype: str, str / False, False\n        '
        if self.update and (not self._update_cache()):
            return (False, False)
        try:
            trans = self.apt_client.upgrade_packages([package])
            trans.simulate(reply_handler=self._apt_trans_started, error_handler=self._apt_error_handler)
            pkg = trans.packages[4][0]
            if pkg == package:
                cache = apt.Cache()
                cache.open(None)
                cache.upgrade()
                if cache[pkg].installed:
                    return (cache[pkg].installed.version, cache[pkg].candidate.version)
            return (False, False)
        except Exception as error:
            log('Exception while checking versions: %s' % error)
            return (False, False)

    def _update_cache(self):
        if False:
            return 10
        ' Update apt client cache\n\n        :return: success of updating apt cache\n        :rtype: bool\n        '
        try:
            return self.apt_client.update_cache(wait=True) == 'exit-success'
        except errors.NotAuthorizedError:
            log('You are not allowed to update the cache')
            return False

    def upgrade_package(self, package):
        if False:
            return 10
        ' Upgrade apt package\n\n        :param package: package to upgrade\n        :type package: str\n        :return: success of apt package upgrade\n        :rtype: bool\n        '
        try:
            log('Installing new version')
            if self.apt_client.upgrade_packages([package], wait=True) == 'exit-success':
                log('Upgrade successful')
                return True
        except Exception as error:
            log('Exception during upgrade: %s' % error)
        return False

    def upgrade_system(self):
        if False:
            while True:
                i = 10
        ' Upgrade system\n\n        :return: success of system upgrade\n        :rtype: bool\n        '
        try:
            log('Upgrading system')
            if self.apt_client.upgrade_system(wait=True) == 'exit-success':
                return True
        except Exception as error:
            log('Exception during system upgrade: %s' % error)
        return False

    def _apt_trans_started(self):
        if False:
            while True:
                i = 10
        ' Apt transfer reply handler\n        '

    @staticmethod
    def _apt_error_handler(error):
        if False:
            return 10
        ' Apt transfer error handler\n\n        :param error: apt error message\n        :type error: str\n        '
        log('Apt Error %s' % error)