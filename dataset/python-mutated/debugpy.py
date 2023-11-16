from typing import List
from localstack.packages import InstallTarget, Package, PackageInstaller
from localstack.utils.run import run

class DebugPyPackage(Package):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('DebugPy', 'latest')

    def get_versions(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        return ['latest']

    def _get_installer(self, version: str) -> PackageInstaller:
        if False:
            return 10
        return DebugPyPackageInstaller('debugpy', version)

class DebugPyPackageInstaller(PackageInstaller):

    def is_installed(self) -> bool:
        if False:
            return 10
        try:
            import debugpy
            assert debugpy
            return True
        except ModuleNotFoundError:
            return False

    def _get_install_marker_path(self, install_dir: str) -> str:
        if False:
            return 10
        return install_dir

    def _install(self, target: InstallTarget) -> None:
        if False:
            return 10
        cmd = 'pip install debugpy'
        run(cmd)
debugpy_package = DebugPyPackage()