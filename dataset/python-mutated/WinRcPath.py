import os
from edk2toolext.environment.plugintypes.uefi_build_plugin import IUefiBuildPlugin
import edk2toollib.windows.locate_tools as locate_tools
from edk2toolext.environment import shell_environment
from edk2toolext.environment import version_aggregator

class WinRcPath(IUefiBuildPlugin):

    def do_post_build(self, thebuilder):
        if False:
            return 10
        return 0

    def do_pre_build(self, thebuilder):
        if False:
            i = 10
            return i + 15
        path = locate_tools.FindToolInWinSdk('rc.exe')
        if path is None:
            thebuilder.logging.warning('Failed to find rc.exe')
        else:
            p = os.path.abspath(os.path.dirname(path))
            shell_environment.GetEnvironment().set_shell_var('WINSDK_PATH_FOR_RC_EXE', p)
            version_aggregator.GetVersionAggregator().ReportVersion('WINSDK_PATH_FOR_RC_EXE', p, version_aggregator.VersionTypes.INFO)
        return 0