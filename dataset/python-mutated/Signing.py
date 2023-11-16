""" Signing of executables.

"""
from nuitka.Options import getMacOSSigningIdentity, shallUseSigningForNotarization
from nuitka.Tracing import postprocessing_logger
from .Execution import executeToolChecked
from .FileOperations import withMadeWritableFileMode
_macos_codesign_usage = "The 'codesign' is used to make signatures on macOS and required to be found."

def _filterCodesignErrorOutput(stderr):
    if False:
        i = 10
        return i + 15
    stderr = b'\n'.join((line for line in stderr.splitlines() if line if b'replacing existing signature' not in line))
    if b'errSecInternalComponent' in stderr:
        postprocessing_logger.sysexit("Access to the specified codesign certificate was not allowed. Please 'allow all items' or when compiling with GUI available, enable prompting for the certificate in KeyChain Access application for this certificate.")
    return (None, stderr)

def addMacOSCodeSignature(filenames):
    if False:
        return 10
    'Add the code signature to filenames.\n\n    Args:\n        filenames - The filenames to be signed.\n\n    Returns:\n        None\n\n    Notes:\n        This is macOS specific.\n    '
    identity = getMacOSSigningIdentity()
    command = ['codesign', '-s', identity, '--force', '--deep', '--preserve-metadata=entitlements']
    if shallUseSigningForNotarization():
        command.append('--options=runtime')
    assert type(filenames) is not str
    command.extend(filenames)
    with withMadeWritableFileMode(filenames):
        executeToolChecked(logger=postprocessing_logger, command=command, absence_message=_macos_codesign_usage, stderr_filter=_filterCodesignErrorOutput)