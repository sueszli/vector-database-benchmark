import os
import sys
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
            print('Hello World!')
        ' return the DSC given the architectures requested.\n\n        ArchCsv: csv string containing all architectures to build\n        '
        return 'IntelTdx/IntelTdxX64.dsc'
import PlatformBuildLib
PlatformBuildLib.CommonPlatform = CommonPlatform