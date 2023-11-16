"""
Functions here are used to take abstract specs and make them concrete.
For example, if a spec asks for a version between 1.8 and 1.9, these
functions might take will take the most recent 1.9 version of the
package available.  Or, if the user didn't specify a compiler for a
spec, then this will assign a compiler to the spec based on defaults
or user preferences.

TODO: make this customizable and allow users to configure
      concretization  policies.
"""
import functools
import platform
import tempfile
from contextlib import contextmanager
from itertools import chain
from typing import Union
import archspec.cpu
import llnl.util.lang
import llnl.util.tty as tty
import spack.abi
import spack.compilers
import spack.config
import spack.environment
import spack.error
import spack.platforms
import spack.repo
import spack.spec
import spack.target
import spack.tengine
import spack.util.path
import spack.variant as vt
from spack.package_prefs import PackagePrefs, is_spec_buildable, spec_externals
from spack.version import ClosedOpenRange, VersionList, ver
_abi: Union[spack.abi.ABI, llnl.util.lang.Singleton] = llnl.util.lang.Singleton(lambda : spack.abi.ABI())

@functools.total_ordering
class reverse_order:
    """Helper for creating key functions.

    This is a wrapper that inverts the sense of the natural
    comparisons on the object.
    """

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self.value = value

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return other.value == self.value

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        return other.value < self.value

class Concretizer:
    """You can subclass this class to override some of the default
    concretization strategies, or you can override all of them.
    """
    check_for_compiler_existence = None

    def __init__(self, abstract_spec=None):
        if False:
            return 10
        if Concretizer.check_for_compiler_existence is None:
            Concretizer.check_for_compiler_existence = not spack.config.get('config:install_missing_compilers', False)
        self.abstract_spec = abstract_spec
        self._adjust_target_answer_generator = None

    def concretize_develop(self, spec):
        if False:
            return 10
        '\n        Add ``dev_path=*`` variant to packages built from local source.\n        '
        env = spack.environment.active_environment()
        dev_info = env.dev_specs.get(spec.name, {}) if env else {}
        if not dev_info:
            return False
        path = spack.util.path.canonicalize_path(dev_info['path'], default_wd=env.path)
        if 'dev_path' in spec.variants:
            assert spec.variants['dev_path'].value == path
            changed = False
        else:
            spec.variants.setdefault('dev_path', vt.SingleValuedVariant('dev_path', path))
            changed = True
        changed |= spec.constrain(dev_info['spec'])
        return changed

    def _valid_virtuals_and_externals(self, spec):
        if False:
            print('Hello World!')
        'Returns a list of candidate virtual dep providers and external\n        packages that coiuld be used to concretize a spec.\n\n        Preferred specs come first in the list.\n        '
        candidates = [spec]
        pref_key = lambda spec: 0
        if spec.virtual:
            candidates = spack.repo.PATH.providers_for(spec)
            if not candidates:
                raise spack.error.UnsatisfiableProviderSpecError(candidates[0], spec)
            spec_w_prefs = find_spec(spec, lambda p: PackagePrefs.has_preferred_providers(p.name, spec.name), spec)
            pref_key = PackagePrefs(spec_w_prefs.name, 'providers', spec.name)
        usable = []
        for cspec in candidates:
            if is_spec_buildable(cspec):
                usable.append(cspec)
            externals = spec_externals(cspec)
            for ext in externals:
                if ext.intersects(spec):
                    usable.append(ext)
        if not usable:
            raise NoBuildError(spec)
        return sorted(usable, key=lambda spec: (not spec.external, pref_key(spec), spec.name, reverse_order(spec.versions), spec))

    def choose_virtual_or_external(self, spec: spack.spec.Spec):
        if False:
            for i in range(10):
                print('nop')
        'Given a list of candidate virtual and external packages, try to\n        find one that is most ABI compatible.\n        '
        candidates = self._valid_virtuals_and_externals(spec)
        if not candidates:
            return candidates
        abi_exemplar = find_spec(spec, lambda x: x.compiler)
        if abi_exemplar is None:
            abi_exemplar = spec.root
        return sorted(candidates, reverse=True, key=lambda spec: (_abi.compatible(spec, abi_exemplar, loose=True), _abi.compatible(spec, abi_exemplar)))

    def concretize_version(self, spec):
        if False:
            i = 10
            return i + 15
        "If the spec is already concrete, return.  Otherwise take\n        the preferred version from spackconfig, and default to the package's\n        version if there are no available versions.\n\n        TODO: In many cases we probably want to look for installed\n              versions of each package and use an installed version\n              if we can link to it.  The policy implemented here will\n              tend to rebuild a lot of stuff becasue it will prefer\n              a compiler in the spec to any compiler already-\n              installed things were built with.  There is likely\n              some better policy that finds some middle ground\n              between these two extremes.\n        "
        if spec.versions.concrete:
            return False
        pkg_versions = spec.package_class.versions
        usable = [v for v in pkg_versions if any((v.intersects(sv) for sv in spec.versions))]
        yaml_prefs = PackagePrefs(spec.name, 'version')
        keyfn = lambda v: (-yaml_prefs(v), pkg_versions.get(v).get('preferred', False), not v.isdevelop(), v)
        usable.sort(key=keyfn, reverse=True)
        if usable:
            spec.versions = ver([usable[0]])
        elif not spec.versions or spec.versions == VersionList([':']):
            raise NoValidVersionError(spec)
        else:
            last = spec.versions[-1]
            if isinstance(last, ClosedOpenRange):
                range_as_version = VersionList([last]).concrete_range_as_version
                if range_as_version:
                    spec.versions = ver([range_as_version])
                else:
                    raise NoValidVersionError(spec)
            else:
                spec.versions = ver([last])
        return True

    def concretize_architecture(self, spec):
        if False:
            while True:
                i = 10
        'If the spec is empty provide the defaults of the platform. If the\n        architecture is not a string type, then check if either the platform,\n        target or operating system are concretized. If any of the fields are\n        changed then return True. If everything is concretized (i.e the\n        architecture attribute is a namedtuple of classes) then return False.\n        If the target is a string type, then convert the string into a\n        concretized architecture. If it has no architecture and the root of the\n        DAG has an architecture, then use the root otherwise use the defaults\n        on the platform.\n        '
        if spec.architecture is None:
            spec.architecture = spack.spec.ArchSpec()
        if spec.architecture.concrete:
            return False
        if spec.architecture.platform:
            new_plat = spack.platforms.by_name(spec.architecture.platform)
        else:
            platform_spec = find_spec(spec, lambda x: x.architecture and x.architecture.platform)
            if platform_spec:
                new_plat = spack.platforms.by_name(platform_spec.architecture.platform)
            else:
                new_plat = spack.platforms.host()
        if spec.architecture.os:
            new_os = spec.architecture.os
        else:
            new_os_spec = find_spec(spec, lambda x: x.architecture and x.architecture.platform == str(new_plat) and x.architecture.os)
            if new_os_spec:
                new_os = new_os_spec.architecture.os
            else:
                new_os = new_plat.operating_system('default_os')
        curr_target = None
        if spec.architecture.target:
            curr_target = spec.architecture.target
        if spec.architecture.target and spec.architecture.target_concrete:
            new_target = spec.architecture.target
        else:
            new_target_spec = find_spec(spec, lambda x: x.architecture and x.architecture.platform == str(new_plat) and x.architecture.target and (x.architecture.target != curr_target))
            if new_target_spec:
                if curr_target:
                    new_target_arch = spack.spec.ArchSpec((None, None, new_target_spec.architecture.target))
                    curr_target_arch = spack.spec.ArchSpec((None, None, curr_target))
                    curr_target_arch.constrain(new_target_arch)
                    new_target = curr_target_arch.target
                else:
                    new_target = new_target_spec.architecture.target
            else:
                if PackagePrefs.has_preferred_targets(spec.name):
                    new_target = self.target_from_package_preferences(spec)
                else:
                    new_target = new_plat.target('default_target')
                if curr_target:
                    new_target_arch = spack.spec.ArchSpec((None, None, str(new_target)))
                    curr_target_arch = spack.spec.ArchSpec((None, None, str(curr_target)))
                    if not new_target_arch.intersects(curr_target_arch):
                        valid_target_ranges = str(curr_target).split(',')
                        for target_range in valid_target_ranges:
                            (t_min, t_sep, t_max) = target_range.partition(':')
                            if not t_sep:
                                new_target = t_min
                                break
                            elif t_max:
                                new_target = t_max
                                break
                            elif t_min:
                                new_target = t_min
                                break
        arch_spec = (str(new_plat), str(new_os), str(new_target))
        new_arch = spack.spec.ArchSpec(arch_spec)
        spec_changed = new_arch != spec.architecture
        spec.architecture = new_arch
        return spec_changed

    def target_from_package_preferences(self, spec):
        if False:
            for i in range(10):
                print('nop')
        "Returns the preferred target from the package preferences if\n        there's any.\n\n        Args:\n            spec: abstract spec to be concretized\n        "
        target_prefs = PackagePrefs(spec.name, 'target')
        target_specs = [spack.spec.Spec('target=%s' % tname) for tname in archspec.cpu.TARGETS]

        def tspec_filter(s):
            if False:
                return 10
            target = archspec.cpu.TARGETS[str(s.architecture.target)]
            arch_family_name = target.family.name
            return arch_family_name == platform.machine()
        target_specs = list(filter(tspec_filter, target_specs))
        target_specs.sort(key=target_prefs)
        new_target = target_specs[0].architecture.target
        return new_target

    def concretize_variants(self, spec):
        if False:
            return 10
        'If the spec already has variants filled in, return.  Otherwise, add\n        the user preferences from packages.yaml or the default variants from\n        the package specification.\n        '
        changed = False
        preferred_variants = PackagePrefs.preferred_variants(spec.name)
        pkg_cls = spec.package_class
        for (name, entry) in pkg_cls.variants.items():
            (variant, when) = entry
            var = spec.variants.get(name, None)
            if var and '*' in var:
                spec.variants.pop(name)
            if name not in spec.variants and any((spec.satisfies(w) for w in when)):
                changed = True
                if name in preferred_variants:
                    spec.variants[name] = preferred_variants.get(name)
                else:
                    spec.variants[name] = variant.make_default()
            if name in spec.variants and (not any((spec.satisfies(w) for w in when))):
                raise vt.InvalidVariantForSpecError(name, when, spec)
        return changed

    def concretize_compiler(self, spec):
        if False:
            return 10
        "If the spec already has a compiler, we're done.  If not, then take\n        the compiler used for the nearest ancestor with a compiler\n        spec and use that.  If the ancestor's compiler is not\n        concrete, then used the preferred compiler as specified in\n        spackconfig.\n\n        Intuition: Use the spackconfig default if no package that depends on\n        this one has a strict compiler requirement.  Otherwise, try to\n        build with the compiler that will be used by libraries that\n        link to this one, to maximize compatibility.\n        "
        if not spec.architecture.concrete:
            return True

        def _proper_compiler_style(cspec, aspec):
            if False:
                for i in range(10):
                    print('nop')
            compilers = spack.compilers.compilers_for_spec(cspec, arch_spec=aspec)
            if cspec.concrete and compilers and (cspec.version not in [c.version for c in compilers]):
                return []
            return compilers
        if spec.compiler and spec.compiler.concrete:
            if self.check_for_compiler_existence and (not _proper_compiler_style(spec.compiler, spec.architecture)):
                _compiler_concretization_failure(spec.compiler, spec.architecture)
            return False
        other_spec = spec if spec.compiler else find_spec(spec, lambda x: x.compiler, spec.root)
        other_compiler = other_spec.compiler
        assert other_spec
        if other_compiler and other_compiler.concrete:
            if self.check_for_compiler_existence and (not _proper_compiler_style(other_compiler, spec.architecture)):
                _compiler_concretization_failure(other_compiler, spec.architecture)
            spec.compiler = other_compiler
            return True
        if other_compiler:
            compiler_list = spack.compilers.find_specs_by_arch(other_compiler, spec.architecture)
            if not compiler_list:
                if not self.check_for_compiler_existence:
                    cpkg_spec = spack.compilers.pkg_spec_for_compiler(other_compiler)
                    self.concretize_version(cpkg_spec)
                    spec.compiler = spack.spec.CompilerSpec(other_compiler.name, cpkg_spec.versions)
                    return True
                else:
                    raise UnavailableCompilerVersionError(other_compiler, spec.architecture)
        else:
            compiler_list = spack.compilers.all_compiler_specs()
            if not compiler_list:
                raise spack.compilers.NoCompilersError()
        compiler_list = sorted(compiler_list, key=lambda x: (x.name, x.version), reverse=True)
        ppk = PackagePrefs(other_spec.name, 'compiler')
        matches = sorted(compiler_list, key=ppk)
        try:
            spec.compiler = next((c for c in matches if _proper_compiler_style(c, spec.architecture))).copy()
        except StopIteration:
            _compiler_concretization_failure(other_compiler, spec.architecture)
        assert spec.compiler.concrete
        return True

    def concretize_compiler_flags(self, spec):
        if False:
            while True:
                i = 10
        '\n        The compiler flags are updated to match those of the spec whose\n        compiler is used, defaulting to no compiler flags in the spec.\n        Default specs set at the compiler level will still be added later.\n        '
        if not spec.architecture.concrete:
            return True
        compiler_match = lambda other: spec.compiler == other.compiler and spec.architecture == other.architecture
        ret = False
        for flag in spack.spec.FlagMap.valid_compiler_flags():
            if flag not in spec.compiler_flags:
                spec.compiler_flags[flag] = list()
            try:
                nearest = next((p for p in spec.traverse(direction='parents') if compiler_match(p) and p is not spec and (flag in p.compiler_flags)))
                nearest_flags = nearest.compiler_flags.get(flag, [])
                flags = spec.compiler_flags.get(flag, [])
                if set(nearest_flags) - set(flags):
                    spec.compiler_flags[flag] = list(llnl.util.lang.dedupe(nearest_flags + flags))
                    ret = True
            except StopIteration:
                pass
        try:
            compiler = spack.compilers.compiler_for_spec(spec.compiler, spec.architecture)
        except spack.compilers.NoCompilerForSpecError:
            if self.check_for_compiler_existence:
                raise
            return ret
        for flag in compiler.flags:
            config_flags = compiler.flags.get(flag, [])
            flags = spec.compiler_flags.get(flag, [])
            spec.compiler_flags[flag] = list(llnl.util.lang.dedupe(config_flags + flags))
            if set(config_flags) - set(flags):
                ret = True
        return ret

    def adjust_target(self, spec):
        if False:
            return 10
        'Adjusts the target microarchitecture if the compiler is too old\n        to support the default one.\n\n        Args:\n            spec: spec to be concretized\n\n        Returns:\n            True if spec was modified, False otherwise\n        '
        if not (spec.architecture and spec.architecture.concrete):
            return True

        def _make_only_one_call(spec):
            if False:
                i = 10
                return i + 15
            yield self._adjust_target(spec)
            while True:
                yield False
        if self._adjust_target_answer_generator is None:
            self._adjust_target_answer_generator = _make_only_one_call(spec)
        return next(self._adjust_target_answer_generator)

    def _adjust_target(self, spec):
        if False:
            for i in range(10):
                print('nop')
        'Assumes that the architecture and the compiler have been\n        set already and checks if the current target microarchitecture\n        is the default and can be optimized by the compiler.\n\n        If not, downgrades the microarchitecture until a suitable one\n        is found. If none can be found raise an error.\n\n        Args:\n            spec: spec to be concretized\n\n        Returns:\n            True if any modification happened, False otherwise\n        '
        import archspec.cpu
        current_target = spec.architecture.target
        current_platform = spack.platforms.by_name(spec.architecture.platform)
        default_target = current_platform.target('default_target')
        if PackagePrefs.has_preferred_targets(spec.name):
            default_target = self.target_from_package_preferences(spec)
        if current_target != default_target or (self.abstract_spec and self.abstract_spec.architecture and self.abstract_spec.architecture.concrete):
            return False
        try:
            current_target.optimization_flags(spec.compiler)
        except archspec.cpu.UnsupportedMicroarchitecture:
            microarchitecture = current_target.microarchitecture
            for ancestor in microarchitecture.ancestors:
                candidate = None
                try:
                    candidate = spack.target.Target(ancestor)
                    candidate.optimization_flags(spec.compiler)
                except archspec.cpu.UnsupportedMicroarchitecture:
                    continue
                if candidate is not None:
                    msg = '{0.name}@{0.version} cannot build optimized binaries for "{1}". Using best target possible: "{2}"'
                    msg = msg.format(spec.compiler, current_target, candidate)
                    tty.warn(msg)
                    spec.architecture.target = candidate
                    return True
            else:
                raise
        return False

@contextmanager
def disable_compiler_existence_check():
    if False:
        i = 10
        return i + 15
    saved = Concretizer.check_for_compiler_existence
    Concretizer.check_for_compiler_existence = False
    yield
    Concretizer.check_for_compiler_existence = saved

@contextmanager
def enable_compiler_existence_check():
    if False:
        for i in range(10):
            print('nop')
    saved = Concretizer.check_for_compiler_existence
    Concretizer.check_for_compiler_existence = True
    yield
    Concretizer.check_for_compiler_existence = saved

def find_spec(spec, condition, default=None):
    if False:
        i = 10
        return i + 15
    'Searches the dag from spec in an intelligent order and looks\n    for a spec that matches a condition'
    deptype = ('build', 'link')
    dagiter = chain(spec.traverse(direction='parents', deptype=deptype, root=False), spec.traverse(direction='children', deptype=deptype, root=False))
    visited = set()
    for relative in dagiter:
        if condition(relative):
            return relative
        visited.add(id(relative))
    for relative in spec.root.traverse(deptype='all'):
        if relative is spec:
            continue
        if id(relative) in visited:
            continue
        if condition(relative):
            return relative
    if condition(spec):
        return spec
    return default

def _compiler_concretization_failure(compiler_spec, arch):
    if False:
        while True:
            i = 10
    if not spack.compilers.compilers_for_arch(arch):
        available_os_targets = set(((c.operating_system, c.target) for c in spack.compilers.all_compilers()))
        raise NoCompilersForArchError(arch, available_os_targets)
    else:
        raise UnavailableCompilerVersionError(compiler_spec, arch)

def concretize_specs_together(*abstract_specs, **kwargs):
    if False:
        i = 10
        return i + 15
    'Given a number of specs as input, tries to concretize them together.\n\n    Args:\n        tests (bool or list or set): False to run no tests, True to test\n            all packages, or a list of package names to run tests for some\n        *abstract_specs: abstract specs to be concretized, given either\n            as Specs or strings\n\n    Returns:\n        List of concretized specs\n    '
    if spack.config.get('config:concretizer', 'clingo') == 'original':
        return _concretize_specs_together_original(*abstract_specs, **kwargs)
    return _concretize_specs_together_new(*abstract_specs, **kwargs)

def _concretize_specs_together_new(*abstract_specs, **kwargs):
    if False:
        while True:
            i = 10
    import spack.solver.asp
    allow_deprecated = spack.config.get('config:deprecated', False)
    solver = spack.solver.asp.Solver()
    result = solver.solve(abstract_specs, tests=kwargs.get('tests', False), allow_deprecated=allow_deprecated)
    result.raise_if_unsat()
    return [s.copy() for s in result.specs]

def _concretize_specs_together_original(*abstract_specs, **kwargs):
    if False:
        return 10
    abstract_specs = [spack.spec.Spec(s) for s in abstract_specs]
    tmpdir = tempfile.mkdtemp()
    builder = spack.repo.MockRepositoryBuilder(tmpdir)
    split_specs = [dep.copy(deps=False) for spec1 in abstract_specs for dep in spec1.traverse(root=True)]
    builder.add_package('concretizationroot', dependencies=[(str(x), None, None) for x in split_specs])
    with spack.repo.use_repositories(builder.root, override=False):
        concretization_root = spack.spec.Spec('concretizationroot')
        concretization_root.concretize(tests=kwargs.get('tests', False))
        concrete_specs = [concretization_root[spec.name].copy() for spec in abstract_specs]
    return concrete_specs

class NoCompilersForArchError(spack.error.SpackError):

    def __init__(self, arch, available_os_targets):
        if False:
            return 10
        err_msg = 'No compilers found for operating system %s and target %s.\nIf previous installations have succeeded, the operating system may have been updated.' % (arch.os, arch.target)
        available_os_target_strs = list()
        for (operating_system, t) in available_os_targets:
            os_target_str = '%s-%s' % (operating_system, t) if t else operating_system
            available_os_target_strs.append(os_target_str)
        err_msg += '\nCompilers are defined for the following operating systems and targets:\n\t' + '\n\t'.join(available_os_target_strs)
        super().__init__(err_msg, "Run 'spack compiler find' to add compilers.")

class UnavailableCompilerVersionError(spack.error.SpackError):
    """Raised when there is no available compiler that satisfies a
    compiler spec."""

    def __init__(self, compiler_spec, arch=None):
        if False:
            print('Hello World!')
        err_msg = 'No compilers with spec {0} found'.format(compiler_spec)
        if arch:
            err_msg += ' for operating system {0} and target {1}.'.format(arch.os, arch.target)
        super().__init__(err_msg, "Run 'spack compiler find' to add compilers or 'spack compilers' to see which compilers are already recognized by spack.")

class NoValidVersionError(spack.error.SpackError):
    """Raised when there is no way to have a concrete version for a
    particular spec."""

    def __init__(self, spec):
        if False:
            while True:
                i = 10
        super().__init__("There are no valid versions for %s that match '%s'" % (spec.name, spec.versions))

class InsufficientArchitectureInfoError(spack.error.SpackError):
    """Raised when details on architecture cannot be collected from the
    system"""

    def __init__(self, spec, archs):
        if False:
            while True:
                i = 10
        super().__init__("Cannot determine necessary architecture information for '%s': %s" % (spec.name, str(archs)))

class NoBuildError(spack.error.SpecError):
    """Raised when a package is configured with the buildable option False, but
    no satisfactory external versions can be found
    """

    def __init__(self, spec):
        if False:
            while True:
                i = 10
        msg = "The spec\n    '%s'\n    is configured as not buildable, and no matching external installs were found"
        super().__init__(msg % spec)