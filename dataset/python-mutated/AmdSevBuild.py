import os
import sys
import subprocess
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from PlatformBuildLib import SettingsManager
from PlatformBuildLib import PlatformBuilder

class CommonPlatform:
    """ Common settings for this platform.  Define static data here and use
        for the different parts of stuart
    """
    PackagesSupported = ('OvmfPkg',)
    ArchSupported = ('X64',)
    TargetsSupported = ('DEBUG', 'RELEASE', 'NOOPT')
    Scopes = ('ovmf', 'edk2-build')
    WorkspaceRoot = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

    @classmethod
    def GetDscName(cls, ArchCsv: str) -> str:
        if False:
            return 10
        ' return the DSC given the architectures requested.\n\n        ArchCsv: csv string containing all architectures to build\n        '
        return 'AmdSev/AmdSevX64.dsc'
import PlatformBuildLib
PlatformBuildLib.CommonPlatform = CommonPlatform
subprocess.run(['touch', 'OvmfPkg/AmdSev/Grub/grub.efi'])
subprocess.run(['ls', '-l', '--sort=time', 'OvmfPkg/AmdSev/Grub'])