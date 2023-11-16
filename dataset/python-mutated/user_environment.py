import os
import sys
from contextlib import contextmanager
from typing import Callable
from llnl.util.lang import nullcontext
import spack.build_environment
import spack.config
import spack.error
import spack.spec
import spack.util.environment as environment
import spack.util.prefix as prefix
from spack import traverse
from spack.context import Context
spack_loaded_hashes_var = 'SPACK_LOADED_HASHES'

def prefix_inspections(platform):
    if False:
        while True:
            i = 10
    'Get list of prefix inspections for platform\n\n    Arguments:\n        platform (str): the name of the platform to consider. The platform\n            determines what environment variables Spack will use for some\n            inspections.\n\n    Returns:\n        A dictionary mapping subdirectory names to lists of environment\n            variables to modify with that directory if it exists.\n    '
    inspections = spack.config.get('modules:prefix_inspections')
    if isinstance(inspections, dict):
        return inspections
    inspections = {'bin': ['PATH'], 'man': ['MANPATH'], 'share/man': ['MANPATH'], 'share/aclocal': ['ACLOCAL_PATH'], 'lib/pkgconfig': ['PKG_CONFIG_PATH'], 'lib64/pkgconfig': ['PKG_CONFIG_PATH'], 'share/pkgconfig': ['PKG_CONFIG_PATH'], '': ['CMAKE_PREFIX_PATH']}
    if platform == 'darwin':
        inspections['lib'] = ['DYLD_FALLBACK_LIBRARY_PATH']
        inspections['lib64'] = ['DYLD_FALLBACK_LIBRARY_PATH']
    return inspections

def unconditional_environment_modifications(view):
    if False:
        return 10
    'List of environment (shell) modifications to be processed for view.\n\n    This list does not depend on the specs in this environment'
    env = environment.EnvironmentModifications()
    for (subdir, vars) in prefix_inspections(sys.platform).items():
        full_subdir = os.path.join(view.root, subdir)
        for var in vars:
            env.prepend_path(var, full_subdir)
    return env

@contextmanager
def projected_prefix(*specs: spack.spec.Spec, projection: Callable[[spack.spec.Spec], str]):
    if False:
        return 10
    "Temporarily replace every Spec's prefix with projection(s)"
    prefixes = dict()
    for s in traverse.traverse_nodes(specs, key=lambda s: s.dag_hash()):
        if s.external:
            continue
        prefixes[s.dag_hash()] = s.prefix
        s.prefix = prefix.Prefix(projection(s))
    yield
    for s in traverse.traverse_nodes(specs, key=lambda s: s.dag_hash()):
        s.prefix = prefixes.get(s.dag_hash(), s.prefix)

def environment_modifications_for_specs(*specs: spack.spec.Spec, view=None, set_package_py_globals: bool=True):
    if False:
        i = 10
        return i + 15
    'List of environment (shell) modifications to be processed for spec.\n\n    This list is specific to the location of the spec or its projection in\n    the view.\n\n    Args:\n        specs: spec(s) for which to list the environment modifications\n        view: view associated with the spec passed as first argument\n        set_package_py_globals: whether or not to set the global variables in the\n            package.py files (this may be problematic when using buildcaches that have\n            been built on a different but compatible OS)\n    '
    env = environment.EnvironmentModifications()
    topo_ordered = traverse.traverse_nodes(specs, root=True, deptype=('run', 'link'), order='topo')
    if view:
        maybe_projected = projected_prefix(*specs, projection=view.get_projection_for_spec)
    else:
        maybe_projected = nullcontext()
    with maybe_projected:
        for s in reversed(list(topo_ordered)):
            static = environment.inspect_path(s.prefix, prefix_inspections(s.platform), exclude=environment.is_system_path)
            env.extend(static)
        setup_context = spack.build_environment.SetupContext(*specs, context=Context.RUN)
        if set_package_py_globals:
            setup_context.set_all_package_py_globals()
        dynamic = setup_context.get_env_modifications()
        env.extend(dynamic)
    return env