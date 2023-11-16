from edk2toolext.environment.uefi_build import UefiBuilder
from edk2toollib.utility_functions import GetHostInfo
from argparse import ArgumentParser, Namespace
from typing import Tuple

def add_command_line_option(parser: ArgumentParser) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Adds the CodeQL command to the platform command line options.\n\n    Args:\n        parser (ArgumentParser): The argument parser used in this build.\n\n    '
    parser.add_argument('--codeql', dest='codeql', action='store_true', default=False, help='Optional - Produces CodeQL results from the build. See BaseTools/Plugin/CodeQL/Readme.md for more info.')

def get_scopes(codeql_enabled: bool) -> Tuple[str]:
    if False:
        i = 10
        return i + 15
    'Returns the active CodeQL scopes for this build.\n\n    Args:\n        codeql_enabled (bool): Whether CodeQL is enabled.\n\n    Returns:\n        Tuple[str]: A tuple of strings containing scopes that enable the\n                    CodeQL plugin.\n    '
    active_scopes = ()
    if codeql_enabled:
        if GetHostInfo().os == 'Linux':
            active_scopes += ('codeql-linux-ext-dep',)
        else:
            active_scopes += ('codeql-windows-ext-dep',)
        active_scopes += ('codeql-build', 'codeql-analyze')
    return active_scopes

def is_codeql_enabled_on_command_line(args: Namespace) -> bool:
    if False:
        while True:
            i = 10
    'Returns whether CodeQL was enabled on the command line.\n\n    Args:\n        args (Namespace): Object holding a string representation of command\n                          line arguments.\n\n    Returns:\n        bool: True if CodeQL is enabled on the command line. Otherwise, false.\n    '
    return args.codeql

def set_audit_only_mode(uefi_builder: UefiBuilder) -> None:
    if False:
        return 10
    'Configures the CodeQL plugin to run in audit only mode.\n\n    Args:\n        uefi_builder (UefiBuilder): The UefiBuilder object for this platform\n                                    build.\n\n    '
    uefi_builder.env.SetValue('STUART_CODEQL_AUDIT_ONLY', 'true', 'Platform Defined')