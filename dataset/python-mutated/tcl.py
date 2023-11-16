"""This module implements the classes necessary to generate Tcl
non-hierarchical modules.
"""
import os.path
from typing import Dict, Optional, Tuple
import spack.config
import spack.spec
import spack.tengine as tengine
from .common import BaseConfiguration, BaseContext, BaseFileLayout, BaseModuleFileWriter

def configuration(module_set_name: str) -> dict:
    if False:
        return 10
    return spack.config.get(f'modules:{module_set_name}:tcl', {})
configuration_registry: Dict[Tuple[str, str, bool], BaseConfiguration] = {}

def make_configuration(spec: spack.spec.Spec, module_set_name: str, explicit: Optional[bool]=None) -> BaseConfiguration:
    if False:
        i = 10
        return i + 15
    'Returns the tcl configuration for spec'
    explicit = bool(spec._installed_explicitly()) if explicit is None else explicit
    key = (spec.dag_hash(), module_set_name, explicit)
    try:
        return configuration_registry[key]
    except KeyError:
        return configuration_registry.setdefault(key, TclConfiguration(spec, module_set_name, explicit))

def make_layout(spec: spack.spec.Spec, module_set_name: str, explicit: Optional[bool]=None) -> BaseFileLayout:
    if False:
        print('Hello World!')
    'Returns the layout information for spec'
    return TclFileLayout(make_configuration(spec, module_set_name, explicit))

def make_context(spec: spack.spec.Spec, module_set_name: str, explicit: Optional[bool]=None) -> BaseContext:
    if False:
        print('Hello World!')
    'Returns the context information for spec'
    return TclContext(make_configuration(spec, module_set_name, explicit))

class TclConfiguration(BaseConfiguration):
    """Configuration class for tcl module files."""

class TclFileLayout(BaseFileLayout):
    """File layout for tcl module files."""

    @property
    def modulerc(self):
        if False:
            while True:
                i = 10
        'Returns the modulerc file associated with current module file'
        return os.path.join(os.path.dirname(self.filename), '.modulerc')

class TclContext(BaseContext):
    """Context class for tcl module files."""

    @tengine.context_property
    def prerequisites(self):
        if False:
            print('Hello World!')
        'List of modules that needs to be loaded automatically.'
        return self._create_module_list_of('specs_to_prereq')

class TclModulefileWriter(BaseModuleFileWriter):
    """Writer class for tcl module files."""
    default_template = 'modules/modulefile.tcl'
    modulerc_header = ['#%Module4.7']
    hide_cmd_format = 'module-hide --soft --hidden-loaded %s'