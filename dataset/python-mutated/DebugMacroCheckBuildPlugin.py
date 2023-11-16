import logging
import os
import pathlib
import sys
import yaml
plugin_file = pathlib.Path(__file__)
sys.path.append(str(plugin_file.parent.parent))
import DebugMacroCheck
from edk2toolext import edk2_logging
from edk2toolext.environment.plugintypes.uefi_build_plugin import IUefiBuildPlugin
from edk2toolext.environment.uefi_build import UefiBuilder
from edk2toollib.uefi.edk2.path_utilities import Edk2Path
from pathlib import Path

class DebugMacroCheckBuildPlugin(IUefiBuildPlugin):

    def do_pre_build(self, builder: UefiBuilder) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Debug Macro Check pre-build functionality.\n\n        The plugin is invoked in pre-build since it can operate independently\n        of build tools and to notify the user of any errors earlier in the\n        build process to reduce feedback time.\n\n        Args:\n            builder (UefiBuilder): A UEFI builder object for this build.\n\n        Returns:\n            int: The number of debug macro errors found. Zero indicates the\n            check either did not run or no errors were found.\n        '
        env_disable = builder.env.GetValue('DISABLE_DEBUG_MACRO_CHECK')
        if env_disable:
            return 0
        build_target = builder.env.GetValue('TARGET').lower()
        if 'no-target' in build_target:
            return 0
        edk2 = builder.edk2path
        package = edk2.GetContainingPackage(builder.edk2path.GetAbsolutePathOnThisSystemFromEdk2RelativePath(builder.env.GetValue('ACTIVE_PLATFORM')))
        package_path = Path(edk2.GetAbsolutePathOnThisSystemFromEdk2RelativePath(package))
        handler_level_context = []
        for h in logging.getLogger().handlers:
            if h.level < logging.INFO:
                handler_level_context.append((h, h.level))
                h.setLevel(logging.INFO)
        edk2_logging.log_progress('Checking DEBUG Macros')
        sub_data = {}
        package_config_file = Path(os.path.join(package_path, package + '.ci.yaml'))
        if package_config_file.is_file():
            with open(package_config_file, 'r') as cf:
                package_config_file_data = yaml.safe_load(cf)
                if 'DebugMacroCheck' in package_config_file_data and 'StringSubstitutions' in package_config_file_data['DebugMacroCheck']:
                    logging.info(f'Loading substitution data in {str(package_config_file)}')
                    sub_data |= package_config_file_data['DebugMacroCheck']['StringSubstitutions']
        sub_file = builder.env.GetValue('DEBUG_MACRO_CHECK_SUB_FILE')
        if sub_file:
            logging.info(f'Loading substitution file {sub_file}')
            with open(sub_file, 'r') as sf:
                sub_data |= yaml.safe_load(sf)
        try:
            error_count = DebugMacroCheck.check_macros_in_directory(package_path, ignore_git_submodules=False, show_progress_bar=False, **sub_data)
        finally:
            for (h, l) in handler_level_context:
                h.setLevel(l)
        return error_count