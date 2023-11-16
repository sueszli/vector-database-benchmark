import yum
from jadi import component
from aj.plugins.packages.api import PackageManager, Package

@component(PackageManager)
class YUMPackageManager(PackageManager):
    """
    Manager to handle rpm packages.
    """
    id = 'yum'
    name = 'YUM'

    @classmethod
    def __verify__(cls):
        if False:
            return 10
        try:
            yum.YumBase().doGenericSetup(cache=1)
            return True
        except Exception as e:
            return False

    def __init__(self, context):
        if False:
            print('Hello World!')
        PackageManager.__init__(self, context)
        self.yum = yum.YumBase()
        self.yum.doGenericSetup(cache=1)

    def __make_package(self, pkg):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert yum package object in package object.\n\n        :param apt_package: Yum package object from apt module\n        :type apt_package: Yum package object from apt module\n        :return: Package object\n        :rtype: Package object\n        '
        pkg_installed = (self.yum.rpmdb.searchNames(names=[pkg.name]) or [None])[0]
        p = Package(self)
        p.id = f'{pkg.name}.{pkg.arch}'
        p.name = pkg.name
        p.version = pkg.version
        p.description = pkg.arch
        p.is_installed = pkg_installed is not None
        if p.is_installed:
            p.installed_version = pkg_installed.version
            p.is_upgradeable = p.installed_version != p.version
        return p

    def list(self, query=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generator for all packages.\n\n        :param query: Search string\n        :type query: string\n        :return: Package object\n        :rtype:Package object\n        '
        for pkg in self.yum.pkgSack.returnPackages():
            yield self.__make_package(pkg)

    def get_package(self, _id):
        if False:
            while True:
                i = 10
        '\n        Get package informations.\n\n        :param _id: Package name\n        :type _id: string\n        :return: Package object\n        :rtype: Package object\n        '
        pkg = (self.yum.searchNames(names=[_id]) or [None])[0]
        return self.__make_package(pkg)

    def update_lists(self, progress_callback):
        if False:
            print('Hello World!')
        '\n        Refresh list of packages.\n\n        :param progress_callback: Callback function to follow progress\n        :type progress_callback: function\n        '

        class Progress:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.size = 0
                self.done = 0
                self.name = None

            def end(self, amount_read, now=None):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def start(self, filename=None, url=None, basename=None, size=None, now=None, text=None):
                if False:
                    for i in range(10):
                        print('nop')
                self.size = size
                self.done = 0
                self.name = url

            def update(self, amount_read, now=None):
                if False:
                    for i in range(10):
                        print('nop')
                self.done = amount_read
                progress = int(100 * self.done / self.size)
                message = f'{progress}%% {self.name}'
                progress_callback(message=message, done=self.done, total=self.size)
        progress_callback(message='Preparing')
        y = yum.YumBase()
        y.repos.setProgressBar(Progress())
        y.cleanMetadata()
        y.repos.doSetup()
        y.repos.populateSack()

    def get_apply_cmd(self, selection):
        if False:
            print('Hello World!')
        '\n        Prepare command to apply.\n\n        :param selection: Dict of packages an actions\n        :type selection: dict\n        :return: Command for terminal use\n        :rtype: string\n        '
        to_install = [sel['package']['id'] for sel in selection if sel['operation'] in ['install', 'upgrade']]
        to_remove = [sel['package']['id'] for sel in selection if sel['operation'] == 'remove']
        cmd = ''
        if len(to_install) > 0:
            cmd += 'yum install ' + ' '.join(to_install)
            if len(to_remove) > 0:
                cmd += ' && '
        if len(to_remove) > 0:
            cmd += 'yum remove ' + ' '.join(to_remove)
        return cmd