import gevent
import apt
import apt.cache
from apt.progress.base import AcquireProgress
from jadi import component
from aj.plugins.packages.api import PackageManager, Package

@component(PackageManager)
class APTPackageManager(PackageManager):
    """
    Manager to handle apt packages.
    """
    id = 'apt'
    name = 'APT'

    def __init__(self, context):
        if False:
            return 10
        PackageManager.__init__(self, context)

    def __make_package(self, apt_package):
        if False:
            return 10
        '\n        Convert apt package object in package object.\n\n        :param apt_package: Apt package object from apt module\n        :type apt_package: Apt package object from apt module\n        :return: Package object\n        :rtype: Package object\n        '
        p = Package(self)
        p.id = apt_package.fullname if hasattr(apt_package, 'fullname') else apt_package.name
        p.name = p.id
        v = apt_package.versions[-1]
        p.version = v.version
        p.description = v.summary
        p.is_installed = apt_package.installed is not None
        if p.is_installed:
            p.installed_version = apt_package.installed.version
            p.is_upgradeable = p.installed_version != p.version
        return p

    def list(self, query=None):
        if False:
            print('Hello World!')
        '\n        Generator for all packages.\n\n        :param query: Search string\n        :type query: string\n        :return: Package object\n        :rtype:Package object\n        '
        cache = apt.Cache()
        for _id in cache.keys():
            yield self.__make_package(cache[_id])

    def get_package(self, _id):
        if False:
            i = 10
            return i + 15
        '\n        Get package informations.\n\n        :param _id: Package name\n        :type _id: string\n        :return: Package object\n        :rtype: Package object\n        '
        cache = apt.Cache()
        return self.__make_package(cache[_id])

    def update_lists(self, progress_callback):
        if False:
            while True:
                i = 10
        '\n        Refresh list of packages.\n\n        :param progress_callback: Callback function to follow progress\n        :type progress_callback: function\n        '

        class Progress(AcquireProgress):

            def fetch(self, item):
                if False:
                    for i in range(10):
                        print('nop')
                progress = int(100 * self.current_items / self.total_items)
                message = f'{progress}%% {item.shortdesc}'
                progress_callback(message=message, done=self.current_items, total=self.total_items)

            def stop(self):
                if False:
                    i = 10
                    return i + 15
                self.done = True
        cache = apt.Cache()
        ack = Progress()
        try:
            cache.update(fetch_progress=ack)
        except apt.cache.FetchFailedException:
            pass
        while not hasattr(ack, 'done'):
            gevent.sleep(1)

    def get_apply_cmd(self, selection):
        if False:
            for i in range(10):
                print('nop')
        '\n        Prepare command to apply.\n\n        :param selection: Dict of packages an actions\n        :type selection: dict\n        :return: Command for terminal use\n        :rtype: string\n        '
        cmd = 'apt-get install '
        for sel in selection:
            cmd += sel['package']['id'] + {'remove': '-', 'install': '+', 'upgrade': '+'}[sel['operation']] + ' '
        return cmd