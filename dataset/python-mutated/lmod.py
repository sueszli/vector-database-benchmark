import collections
import itertools
import os.path
from typing import Dict, List, Optional, Tuple
import llnl.util.filesystem as fs
import llnl.util.lang as lang
import spack.compilers
import spack.config
import spack.error
import spack.repo
import spack.spec
import spack.tengine as tengine
import spack.util.environment
from .common import BaseConfiguration, BaseContext, BaseFileLayout, BaseModuleFileWriter

def configuration(module_set_name: str) -> dict:
    if False:
        i = 10
        return i + 15
    return spack.config.get(f'modules:{module_set_name}:lmod', {})
configuration_registry: Dict[Tuple[str, str, bool], BaseConfiguration] = {}

def make_configuration(spec: spack.spec.Spec, module_set_name: str, explicit: Optional[bool]=None) -> BaseConfiguration:
    if False:
        while True:
            i = 10
    'Returns the lmod configuration for spec'
    explicit = bool(spec._installed_explicitly()) if explicit is None else explicit
    key = (spec.dag_hash(), module_set_name, explicit)
    try:
        return configuration_registry[key]
    except KeyError:
        return configuration_registry.setdefault(key, LmodConfiguration(spec, module_set_name, explicit))

def make_layout(spec: spack.spec.Spec, module_set_name: str, explicit: Optional[bool]=None) -> BaseFileLayout:
    if False:
        return 10
    'Returns the layout information for spec'
    return LmodFileLayout(make_configuration(spec, module_set_name, explicit))

def make_context(spec: spack.spec.Spec, module_set_name: str, explicit: Optional[bool]=None) -> BaseContext:
    if False:
        while True:
            i = 10
    'Returns the context information for spec'
    return LmodContext(make_configuration(spec, module_set_name, explicit))

def guess_core_compilers(name, store=False) -> List[spack.spec.CompilerSpec]:
    if False:
        print('Hello World!')
    'Guesses the list of core compilers installed in the system.\n\n    Args:\n        store (bool): if True writes the core compilers to the\n            modules.yaml configuration file\n\n    Returns:\n        List of found core compilers\n    '
    core_compilers = []
    for compiler in spack.compilers.all_compilers():
        try:
            is_system_compiler = any((os.path.dirname(getattr(compiler, x, '')) in spack.util.environment.SYSTEM_DIRS for x in ('cc', 'cxx', 'f77', 'fc')))
            if is_system_compiler:
                core_compilers.append(compiler.spec)
        except (KeyError, TypeError, AttributeError):
            continue
    if store and core_compilers:
        modules_cfg = spack.config.get('modules:' + name, {}, scope=spack.config.default_modify_scope())
        modules_cfg.setdefault('lmod', {})['core_compilers'] = [str(x) for x in core_compilers]
        spack.config.set('modules:' + name, modules_cfg, scope=spack.config.default_modify_scope())
    return core_compilers

class LmodConfiguration(BaseConfiguration):
    """Configuration class for lmod module files."""
    default_projections = {'all': '{name}/{version}'}

    @property
    def core_compilers(self) -> List[spack.spec.CompilerSpec]:
        if False:
            i = 10
            return i + 15
        'Returns the list of "Core" compilers\n\n        Raises:\n            CoreCompilersNotFoundError: if the key was not\n                specified in the configuration file or the sequence\n                is empty\n        '
        compilers = [spack.spec.CompilerSpec(c) for c in configuration(self.name).get('core_compilers', [])]
        if not compilers:
            compilers = guess_core_compilers(self.name, store=True)
        if not compilers:
            msg = 'the key "core_compilers" must be set in modules.yaml'
            raise CoreCompilersNotFoundError(msg)
        return compilers

    @property
    def core_specs(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the list of "Core" specs'
        return configuration(self.name).get('core_specs', [])

    @property
    def filter_hierarchy_specs(self):
        if False:
            print('Hello World!')
        'Returns the dict of specs with modified hierarchies'
        return configuration(self.name).get('filter_hierarchy_specs', {})

    @property
    @lang.memoized
    def hierarchy_tokens(self):
        if False:
            for i in range(10):
                print('nop')
        "Returns the list of tokens that are part of the modulefile\n        hierarchy. 'compiler' is always present.\n        "
        tokens = configuration(self.name).get('hierarchy', [])
        not_virtual = [t for t in tokens if t != 'compiler' and (not spack.repo.PATH.is_virtual(t))]
        if not_virtual:
            msg = "Non-virtual specs in 'hierarchy' list for lmod: {0}\n"
            msg += "Please check the 'modules.yaml' configuration files"
            msg = msg.format(', '.join(not_virtual))
            raise NonVirtualInHierarchyError(msg)
        tokens.append('compiler')
        tokens = list(lang.dedupe(tokens))
        return tokens

    @property
    @lang.memoized
    def requires(self):
        if False:
            for i in range(10):
                print('nop')
        "Returns a dictionary mapping all the requirements of this spec\n        to the actual provider. 'compiler' is always present among the\n        requirements.\n        "
        if any((self.spec.satisfies(core_spec) for core_spec in self.core_specs)):
            return {'compiler': self.core_compilers[0]}
        hierarchy_filter_list = []
        for (spec, filter_list) in self.filter_hierarchy_specs.items():
            if self.spec.satisfies(spec):
                hierarchy_filter_list = filter_list
                break
        requirements = {'compiler': self.spec.compiler}
        for x in self.hierarchy_tokens:
            if x in hierarchy_filter_list:
                continue
            if x in self.spec and (not self.spec.package.provides(x)):
                requirements[x] = self.spec[x]
        return requirements

    @property
    def provides(self):
        if False:
            print('Hello World!')
        'Returns a dictionary mapping all the services provided by this\n        spec to the spec itself.\n        '
        provides = {}
        if self.spec.name in spack.compilers.supported_compilers():
            provides['compiler'] = spack.spec.CompilerSpec(self.spec.format('{name}{@versions}'))
        elif self.spec.name in spack.compilers.package_name_to_compiler_name:
            cname = spack.compilers.package_name_to_compiler_name[self.spec.name]
            provides['compiler'] = spack.spec.CompilerSpec(cname, self.spec.versions)
        for x in self.hierarchy_tokens:
            if self.spec.package.provides(x):
                provides[x] = self.spec[x]
        return provides

    @property
    def available(self):
        if False:
            while True:
                i = 10
        'Returns a dictionary of the services that are currently\n        available.\n        '
        available = {}
        available.update(self.requires)
        available.update(self.provides)
        return available

    @property
    @lang.memoized
    def missing(self):
        if False:
            print('Hello World!')
        'Returns the list of tokens that are not available.'
        return [x for x in self.hierarchy_tokens if x not in self.available]

    @property
    def hidden(self):
        if False:
            while True:
                i = 10
        if any((self.spec.package.provides(x) for x in self.hierarchy_tokens)):
            return False
        return super().hidden

class LmodFileLayout(BaseFileLayout):
    """File layout for lmod module files."""
    extension = 'lua'

    @property
    def arch_dirname(self):
        if False:
            print('Hello World!')
        'Returns the root folder for THIS architecture'
        arch_folder_conf = spack.config.get('modules:%s:arch_folder' % self.conf.name, True)
        if arch_folder_conf:
            arch_folder = '-'.join([str(self.spec.platform), str(self.spec.os), str(self.spec.target.family)])
            return os.path.join(self.dirname(), arch_folder)
        return self.dirname()

    @property
    def filename(self):
        if False:
            i = 10
            return i + 15
        'Returns the filename for the current module file'
        requires = self.conf.requires
        hierarchy = self.conf.hierarchy_tokens
        path_parts = lambda x: self.token_to_path(x, requires[x])
        parts = [path_parts(x) for x in hierarchy if x in requires]
        hierarchy_name = os.path.join(*parts)
        return os.path.join(self.arch_dirname, hierarchy_name, f'{self.use_name}.{self.extension}')

    @property
    def modulerc(self):
        if False:
            i = 10
            return i + 15
        'Returns the modulerc file associated with current module file'
        return os.path.join(os.path.dirname(self.filename), f'.modulerc.{self.extension}')

    def token_to_path(self, name, value):
        if False:
            i = 10
            return i + 15
        'Transforms a hierarchy token into the corresponding path part.\n\n        Args:\n            name (str): name of the service in the hierarchy\n            value: actual provider of the service\n\n        Returns:\n            str: part of the path associated with the service\n        '

        def path_part_fmt(token):
            if False:
                return 10
            return fs.polite_path([f'{token.name}', f'{token.version}'])
        core_compilers = self.conf.core_compilers
        if name == 'compiler' and any((spack.spec.CompilerSpec(value).satisfies(c) for c in core_compilers)):
            return 'Core'
        if name == 'compiler':
            return path_part_fmt(token=value)
        return f'{path_part_fmt(token=value)}-{value.dag_hash(length=7)}'

    @property
    def available_path_parts(self):
        if False:
            while True:
                i = 10
        'List of path parts that are currently available. Needed to\n        construct the file name.\n        '
        available = self.conf.available
        hierarchy = self.conf.hierarchy_tokens
        return [self.token_to_path(x, available[x]) for x in hierarchy if x in available]

    @property
    @lang.memoized
    def unlocked_paths(self):
        if False:
            while True:
                i = 10
        "Returns a dictionary mapping conditions to a list of unlocked\n        paths.\n\n        The paths that are unconditionally unlocked are under the\n        key 'None'. The other keys represent the list of services you need\n        loaded to unlock the corresponding paths.\n        "
        unlocked = collections.defaultdict(list)
        requires_key = list(self.conf.requires)
        provides_key = list(self.conf.provides)
        if 'compiler' in provides_key:
            requires_key.remove('compiler')
        combinations = []
        for ii in range(len(provides_key)):
            combinations += itertools.combinations(provides_key, ii + 1)
        to_be_processed = [x + tuple(requires_key) for x in combinations]
        available_combination = []
        for item in to_be_processed:
            hierarchy = self.conf.hierarchy_tokens
            available = self.conf.available
            ac = [x for x in hierarchy if x in item]
            available_combination.append(tuple(ac))
            parts = [self.token_to_path(x, available[x]) for x in ac]
            unlocked[None].append(tuple([self.arch_dirname] + parts))
        unlocked[None] = list(lang.dedupe(unlocked[None]))
        missing = self.conf.missing
        missing_combinations = []
        for ii in range(len(missing)):
            missing_combinations += itertools.combinations(missing, ii + 1)
        for m in missing_combinations:
            to_be_processed = [m + x for x in available_combination]
            for item in to_be_processed:
                hierarchy = self.conf.hierarchy_tokens
                available = self.conf.available
                token2path = lambda x: self.token_to_path(x, available[x])
                parts = []
                for x in hierarchy:
                    if x not in item:
                        continue
                    value = token2path(x) if x in available else x
                    parts.append(value)
                unlocked[m].append(tuple([self.arch_dirname] + parts))
            unlocked[m] = list(lang.dedupe(unlocked[m]))
        return unlocked

class LmodContext(BaseContext):
    """Context class for lmod module files."""

    @tengine.context_property
    def has_modulepath_modifications(self):
        if False:
            for i in range(10):
                print('nop')
        'True if this module modifies MODULEPATH, False otherwise.'
        return bool(self.conf.provides)

    @tengine.context_property
    def has_conditional_modifications(self):
        if False:
            while True:
                i = 10
        'True if this module modifies MODULEPATH conditionally to the\n        presence of other services in the environment, False otherwise.\n        '
        provides = self.conf.provides
        provide_compiler_only = 'compiler' in provides and len(provides) == 1
        has_modifications = self.has_modulepath_modifications
        return has_modifications and (not provide_compiler_only)

    @tengine.context_property
    def name_part(self):
        if False:
            return 10
        'Name of this provider.'
        return self.spec.name

    @tengine.context_property
    def version_part(self):
        if False:
            i = 10
            return i + 15
        'Version of this provider.'
        s = self.spec
        return '-'.join([str(s.version), s.dag_hash(length=7)])

    @tengine.context_property
    def provides(self):
        if False:
            i = 10
            return i + 15
        'Returns the dictionary of provided services.'
        return self.conf.provides

    @tengine.context_property
    def missing(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of missing services.'
        return self.conf.missing

    @tengine.context_property
    @lang.memoized
    def unlocked_paths(self):
        if False:
            return 10
        'Returns the list of paths that are unlocked unconditionally.'
        layout = make_layout(self.spec, self.conf.name)
        return [os.path.join(*parts) for parts in layout.unlocked_paths[None]]

    @tengine.context_property
    def conditionally_unlocked_paths(self):
        if False:
            print('Hello World!')
        'Returns the list of paths that are unlocked conditionally.\n        Each item in the list is a tuple with the structure (condition, path).\n        '
        layout = make_layout(self.spec, self.conf.name)
        value = []
        conditional_paths = layout.unlocked_paths
        conditional_paths.pop(None)
        for (services_needed, list_of_path_parts) in conditional_paths.items():
            condition = ' and '.join([x + '_name' for x in services_needed])
            for parts in list_of_path_parts:

                def manipulate_path(token):
                    if False:
                        for i in range(10):
                            print('nop')
                    if token in self.conf.hierarchy_tokens:
                        return '{0}_name, {0}_version'.format(token)
                    return '"' + token + '"'
                path = ', '.join([manipulate_path(x) for x in parts])
                value.append((condition, path))
        return value

class LmodModulefileWriter(BaseModuleFileWriter):
    """Writer class for lmod module files."""
    default_template = 'modules/modulefile.lua'
    modulerc_header = []
    hide_cmd_format = 'hide_version("%s")'

class CoreCompilersNotFoundError(spack.error.SpackError, KeyError):
    """Error raised if the key 'core_compilers' has not been specified
    in the configuration file.
    """

class NonVirtualInHierarchyError(spack.error.SpackError, TypeError):
    """Error raised if non-virtual specs are used as hierarchy tokens in
    the lmod section of 'modules.yaml'.
    """