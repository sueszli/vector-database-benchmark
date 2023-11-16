"""Metadata generation logic for source distributions.
"""
import os
from pip._vendor.pyproject_hooks import BuildBackendHookCaller
from pip._internal.build_env import BuildEnvironment
from pip._internal.exceptions import InstallationSubprocessError, MetadataGenerationFailed
from pip._internal.utils.subprocess import runner_with_spinner_message
from pip._internal.utils.temp_dir import TempDirectory

def generate_editable_metadata(build_env: BuildEnvironment, backend: BuildBackendHookCaller, details: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Generate metadata using mechanisms described in PEP 660.\n\n    Returns the generated metadata directory.\n    '
    metadata_tmpdir = TempDirectory(kind='modern-metadata', globally_managed=True)
    metadata_dir = metadata_tmpdir.path
    with build_env:
        runner = runner_with_spinner_message('Preparing editable metadata (pyproject.toml)')
        with backend.subprocess_runner(runner):
            try:
                distinfo_dir = backend.prepare_metadata_for_build_editable(metadata_dir)
            except InstallationSubprocessError as error:
                raise MetadataGenerationFailed(package_details=details) from error
    return os.path.join(metadata_dir, distinfo_dir)