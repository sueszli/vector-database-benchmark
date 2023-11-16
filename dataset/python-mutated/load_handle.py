import os
from typing import Mapping, cast
from click import UsageError
import dagster._check as check
from dagster._cli.workspace.cli_target import get_working_directory_from_kwargs
from dagster._core.definitions.reconstruct import ReconstructableRepository

def _cli_load_invariant(condition, msg=None):
    if False:
        return 10
    msg = msg or 'Invalid set of CLI arguments for loading repository/job. See --help for details.'
    if not condition:
        raise UsageError(msg)

def recon_repo_for_cli_args(kwargs: Mapping[str, str]):
    if False:
        for i in range(10):
            print('nop')
    'Builds a ReconstructableRepository for CLI arguments, which can be any of the combinations\n    for repo loading above.\n    '
    check.mapping_param(kwargs, 'kwargs')
    _cli_load_invariant(kwargs.get('pipeline_name') is None)
    if kwargs.get('workspace'):
        check.not_implemented('Workspace not supported yet in this cli command')
    elif kwargs.get('module_name') and kwargs.get('fn_name'):
        _cli_load_invariant(kwargs.get('repository_yaml') is None)
        _cli_load_invariant(kwargs.get('python_file') is None)
        return ReconstructableRepository.for_module(kwargs['module_name'], kwargs['fn_name'], get_working_directory_from_kwargs(kwargs))
    elif kwargs.get('python_file') and kwargs.get('fn_name'):
        _cli_load_invariant(kwargs.get('repository_yaml') is None)
        _cli_load_invariant(kwargs.get('module_name') is None)
        return ReconstructableRepository.for_file(os.path.abspath(cast(str, kwargs['python_file'])), kwargs['fn_name'], get_working_directory_from_kwargs(kwargs))
    else:
        _cli_load_invariant(False)