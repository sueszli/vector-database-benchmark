"""Detect archspec name."""
import os
from .. import CondaVirtualPackage, hookimpl

@hookimpl
def conda_virtual_packages():
    if False:
        while True:
            i = 10
    from ...core.index import get_archspec_name
    archspec_name = get_archspec_name()
    archspec_name = os.getenv('CONDA_OVERRIDE_ARCHSPEC', archspec_name)
    if archspec_name:
        yield CondaVirtualPackage('archspec', '1', archspec_name)