"""This is where most of the action happens in Spack.

The spack package class structure is based strongly on Homebrew
(http://brew.sh/), mainly because Homebrew makes it very easy to create
packages.
"""
import base64
import collections
import copy
import functools
import glob
import hashlib
import inspect
import io
import os
import re
import shutil
import sys
import textwrap
import time
import traceback
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar
import llnl.util.filesystem as fsys
import llnl.util.tty as tty
from llnl.util.lang import classproperty, memoized
from llnl.util.link_tree import LinkTree
import spack.compilers
import spack.config
import spack.deptypes as dt
import spack.directives
import spack.directory_layout
import spack.environment
import spack.error
import spack.fetch_strategy as fs
import spack.hooks
import spack.mirror
import spack.mixins
import spack.multimethod
import spack.patch
import spack.paths
import spack.repo
import spack.spec
import spack.store
import spack.url
import spack.util.environment
import spack.util.path
import spack.util.web
from spack.filesystem_view import YamlFilesystemView
from spack.install_test import PackageTest, TestFailure, TestStatus, TestSuite, cache_extra_test_sources, install_test_root
from spack.installer import InstallError, PackageInstaller
from spack.stage import DIYStage, ResourceStage, Stage, StageComposite, compute_stage_name
from spack.util.executable import ProcessError, which
from spack.util.package_hash import package_hash
from spack.version import GitVersion, StandardVersion, Version
FLAG_HANDLER_RETURN_TYPE = Tuple[Optional[Iterable[str]], Optional[Iterable[str]], Optional[Iterable[str]]]
FLAG_HANDLER_TYPE = Callable[[str, Iterable[str]], FLAG_HANDLER_RETURN_TYPE]
'Allowed URL schemes for spack packages.'
_ALLOWED_URL_SCHEMES = ['http', 'https', 'ftp', 'file', 'git']
_spack_build_logfile = 'spack-build-out.txt'
_spack_build_envfile = 'spack-build-env.txt'
_spack_build_envmodsfile = 'spack-build-env-mods.txt'
_spack_configure_argsfile = 'spack-configure-args.txt'
spack_times_log = 'install_times.json'

def deprecated_version(pkg, version):
    if False:
        for i in range(10):
            print('nop')
    'Return True if the version is deprecated, False otherwise.\n\n    Arguments:\n        pkg (PackageBase): The package whose version is to be checked.\n        version (str or spack.version.StandardVersion): The version being checked\n    '
    if not isinstance(version, StandardVersion):
        version = Version(version)
    for (k, v) in pkg.versions.items():
        if version == k and v.get('deprecated', False):
            return True
    return False

def preferred_version(pkg):
    if False:
        while True:
            i = 10
    '\n    Returns a sorted list of the preferred versions of the package.\n\n    Arguments:\n        pkg (PackageBase): The package whose versions are to be assessed.\n    '
    key_fn = lambda v: (pkg.versions[v].get('preferred', False), not v.isdevelop(), v)
    return max(pkg.versions, key=key_fn)

class WindowsRPath:
    """Collection of functionality surrounding Windows RPATH specific features

    This is essentially meaningless for all other platforms
    due to their use of RPATH. All methods within this class are no-ops on
    non Windows. Packages can customize and manipulate this class as
    they would a genuine RPATH, i.e. adding directories that contain
    runtime library dependencies"""

    def win_add_library_dependent(self):
        if False:
            print('Hello World!')
        "Return extra set of directories that require linking for package\n\n        This method should be overridden by packages that produce\n        binaries/libraries/python extension modules/etc that are installed into\n        directories outside a package's `bin`, `lib`, and `lib64` directories,\n        but still require linking against one of the packages dependencies, or\n        other components of the package itself. No-op otherwise.\n\n        Returns:\n            List of additional directories that require linking\n        "
        return []

    def win_add_rpath(self):
        if False:
            for i in range(10):
                print('nop')
        'Return extra set of rpaths for package\n\n        This method should be overridden by packages needing to\n        include additional paths to be searched by rpath. No-op otherwise\n\n        Returns:\n            List of additional rpaths\n        '
        return []

    def windows_establish_runtime_linkage(self):
        if False:
            for i in range(10):
                print('nop')
        'Establish RPATH on Windows\n\n        Performs symlinking to incorporate rpath dependencies to Windows runtime search paths\n        '
        if sys.platform == 'win32':
            self.win_rpath.add_library_dependent(*self.win_add_library_dependent())
            self.win_rpath.add_rpath(*self.win_add_rpath())
            self.win_rpath.establish_link()
detectable_packages = collections.defaultdict(list)

class DetectablePackageMeta(type):
    """Check if a package is detectable and add default implementations
    for the detection function.
    """
    TAG = 'detectable'

    def __init__(cls, name, bases, attr_dict):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(cls, 'executables') and hasattr(cls, 'libraries'):
            msg = "a package can have either an 'executables' or 'libraries' attribute"
            raise spack.error.SpackError(f"{msg} [package '{name}' defines both]")
        if hasattr(cls, 'executables') or hasattr(cls, 'libraries'):
            if hasattr(cls, 'tags'):
                getattr(cls, 'tags').append(DetectablePackageMeta.TAG)
            else:
                setattr(cls, 'tags', [DetectablePackageMeta.TAG])

            @classmethod
            def platform_executables(cls):
                if False:
                    return 10

                def to_windows_exe(exe):
                    if False:
                        return 10
                    if exe.endswith('$'):
                        exe = exe.replace('$', '%s$' % spack.util.path.win_exe_ext())
                    else:
                        exe += spack.util.path.win_exe_ext()
                    return exe
                plat_exe = []
                if hasattr(cls, 'executables'):
                    for exe in cls.executables:
                        if sys.platform == 'win32':
                            exe = to_windows_exe(exe)
                        plat_exe.append(exe)
                return plat_exe

            @classmethod
            def determine_spec_details(cls, prefix, objs_in_prefix):
                if False:
                    return 10
                'Allow ``spack external find ...`` to locate installations.\n\n                Args:\n                    prefix (str): the directory containing the executables\n                                  or libraries\n                    objs_in_prefix (set): the executables or libraries that\n                                          match the regex\n\n                Returns:\n                    The list of detected specs for this package\n                '
                objs_by_version = collections.defaultdict(list)
                filter_fn = getattr(cls, 'filter_detected_exes', lambda x, exes: exes)
                objs_in_prefix = filter_fn(prefix, objs_in_prefix)
                for obj in objs_in_prefix:
                    try:
                        version_str = cls.determine_version(obj)
                        if version_str:
                            objs_by_version[version_str].append(obj)
                    except Exception as e:
                        msg = 'An error occurred when trying to detect the version of "{0}" [{1}]'
                        tty.debug(msg.format(obj, str(e)))
                specs = []
                for (version_str, objs) in objs_by_version.items():
                    variants = cls.determine_variants(objs, version_str)
                    if not isinstance(variants, list):
                        variants = [variants]
                    for variant in variants:
                        if isinstance(variant, str):
                            variant = (variant, {})
                        (variant_str, extra_attributes) = variant
                        spec_str = '{0}@{1} {2}'.format(cls.name, version_str, variant_str)
                        external_path = extra_attributes.pop('prefix', None)
                        external_modules = extra_attributes.pop('modules', None)
                        try:
                            spec = spack.spec.Spec(spec_str, external_path=external_path, external_modules=external_modules)
                        except Exception as e:
                            msg = 'Parsing failed [spec_str="{0}", error={1}]'
                            tty.debug(msg.format(spec_str, str(e)))
                        else:
                            specs.append(spack.spec.Spec.from_detection(spec, extra_attributes=extra_attributes))
                return sorted(specs)

            @classmethod
            def determine_variants(cls, objs, version_str):
                if False:
                    print('Hello World!')
                return ''
            detectable_packages[cls.namespace].append(cls.name)
            default = False
            if not hasattr(cls, 'determine_spec_details'):
                default = True
                cls.determine_spec_details = determine_spec_details
            if default and (not hasattr(cls, 'determine_version')):
                msg = 'the package "{0}" in the "{1}" repo needs to define the "determine_version" method to be detectable'
                NotImplementedError(msg.format(cls.name, cls.namespace))
            if default and (not hasattr(cls, 'determine_variants')):
                cls.determine_variants = determine_variants
            if 'platform_executables' in cls.__dict__.keys():
                raise PackageError('Packages should not override platform_executables')
            cls.platform_executables = platform_executables
        super(DetectablePackageMeta, cls).__init__(name, bases, attr_dict)

class PackageMeta(spack.builder.PhaseCallbacksMeta, DetectablePackageMeta, spack.directives.DirectiveMeta, spack.multimethod.MultiMethodMeta):
    """
    Package metaclass for supporting directives (e.g., depends_on) and phases
    """

    def __new__(cls, name, bases, attr_dict):
        if False:
            i = 10
            return i + 15
        "\n        FIXME: REWRITE\n        Instance creation is preceded by phase attribute transformations.\n\n        Conveniently transforms attributes to permit extensible phases by\n        iterating over the attribute 'phases' and creating / updating private\n        InstallPhase attributes in the class that will be initialized in\n        __init__.\n        "
        attr_dict['_name'] = None
        return super(PackageMeta, cls).__new__(cls, name, bases, attr_dict)

def on_package_attributes(**attr_dict):
    if False:
        print('Hello World!')
    'Decorator: executes instance function only if object has attr valuses.\n\n    Executes the decorated method only if at the moment of calling the\n    instance has attributes that are equal to certain values.\n\n    Args:\n        attr_dict (dict): dictionary mapping attribute names to their\n            required values\n    '

    def _execute_under_condition(func):
        if False:
            for i in range(10):
                print('nop')

        @functools.wraps(func)
        def _wrapper(instance, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            has_all_attributes = all([hasattr(instance, key) for key in attr_dict])
            if has_all_attributes:
                has_the_right_values = all([getattr(instance, key) == value for (key, value) in attr_dict.items()])
                if has_the_right_values:
                    func(instance, *args, **kwargs)
        return _wrapper
    return _execute_under_condition

class PackageViewMixin:
    """This collects all functionality related to adding installed Spack
    package to views. Packages can customize how they are added to views by
    overriding these functions.
    """

    def view_source(self):
        if False:
            print('Hello World!')
        'The source root directory that will be added to the view: files are\n        added such that their path relative to the view destination matches\n        their path relative to the view source.\n        '
        return self.spec.prefix

    def view_destination(self, view):
        if False:
            print('Hello World!')
        'The target root directory: each file is added relative to this\n        directory.\n        '
        return view.get_projection_for_spec(self.spec)

    def view_file_conflicts(self, view, merge_map):
        if False:
            i = 10
            return i + 15
        'Report any files which prevent adding this package to the view. The\n        default implementation looks for any files which already exist.\n        Alternative implementations may allow some of the files to exist in\n        the view (in this case they would be omitted from the results).\n        '
        return set((dst for dst in merge_map.values() if os.path.lexists(dst)))

    def add_files_to_view(self, view, merge_map, skip_if_exists=True):
        if False:
            while True:
                i = 10
        "Given a map of package files to destination paths in the view, add\n        the files to the view. By default this adds all files. Alternative\n        implementations may skip some files, for example if other packages\n        linked into the view already include the file.\n\n        Args:\n            view (spack.filesystem_view.FilesystemView): the view that's updated\n            merge_map (dict): maps absolute source paths to absolute dest paths for\n                all files in from this package.\n            skip_if_exists (bool): when True, don't link files in view when they\n                already exist. When False, always link files, without checking\n                if they already exist.\n        "
        if skip_if_exists:
            for (src, dst) in merge_map.items():
                if not os.path.lexists(dst):
                    view.link(src, dst, spec=self.spec)
        else:
            for (src, dst) in merge_map.items():
                view.link(src, dst, spec=self.spec)

    def remove_files_from_view(self, view, merge_map):
        if False:
            while True:
                i = 10
        'Given a map of package files to files currently linked in the view,\n        remove the files from the view. The default implementation removes all\n        files. Alternative implementations may not remove all files. For\n        example if two packages include the same file, it should only be\n        removed when both packages are removed.\n        '
        view.remove_files(merge_map.values())
Pb = TypeVar('Pb', bound='PackageBase')

class PackageBase(WindowsRPath, PackageViewMixin, metaclass=PackageMeta):
    """This is the superclass for all spack packages.

    ***The Package class***

    At its core, a package consists of a set of software to be installed.
    A package may focus on a piece of software and its associated software
    dependencies or it may simply be a set, or bundle, of software.  The
    former requires defining how to fetch, verify (via, e.g., sha256), build,
    and install that software and the packages it depends on, so that
    dependencies can be installed along with the package itself.   The latter,
    sometimes referred to as a ``no-source`` package, requires only defining
    the packages to be built.

    Packages are written in pure Python.

    There are two main parts of a Spack package:

      1. **The package class**.  Classes contain ``directives``, which are
         special functions, that add metadata (versions, patches,
         dependencies, and other information) to packages (see
         ``directives.py``). Directives provide the constraints that are
         used as input to the concretizer.

      2. **Package instances**. Once instantiated, a package is
         essentially a software installer.  Spack calls methods like
         ``do_install()`` on the ``Package`` object, and it uses those to
         drive user-implemented methods like ``patch()``, ``install()``, and
         other build steps.  To install software, an instantiated package
         needs a *concrete* spec, which guides the behavior of the various
         install methods.

    Packages are imported from repos (see ``repo.py``).

    **Package DSL**

    Look in ``lib/spack/docs`` or check https://spack.readthedocs.io for
    the full documentation of the package domain-specific language.  That
    used to be partially documented here, but as it grew, the docs here
    became increasingly out of date.

    **Package Lifecycle**

    A package's lifecycle over a run of Spack looks something like this:

    .. code-block:: python

       p = Package()             # Done for you by spack

       p.do_fetch()              # downloads tarball from a URL (or VCS)
       p.do_stage()              # expands tarball in a temp directory
       p.do_patch()              # applies patches to expanded source
       p.do_install()            # calls package's install() function
       p.do_uninstall()          # removes install directory

    although packages that do not have code have nothing to fetch so omit
    ``p.do_fetch()``.

    There are also some other commands that clean the build area:

    .. code-block:: python

       p.do_clean()              # removes the stage directory entirely
       p.do_restage()            # removes the build directory and
                                 # re-expands the archive.

    The convention used here is that a ``do_*`` function is intended to be
    called internally by Spack commands (in ``spack.cmd``).  These aren't for
    package writers to override, and doing so may break the functionality
    of the Package class.

    Package creators have a lot of freedom, and they could technically
    override anything in this class.  That is not usually required.

    For most use cases.  Package creators typically just add attributes
    like ``homepage`` and, for a code-based package, ``url``, or functions
    such as ``install()``.
    There are many custom ``Package`` subclasses in the
    ``spack.build_systems`` package that make things even easier for
    specific build systems.

    """
    versions: dict
    dependencies: dict
    virtual = False
    has_code = True
    parallel = True
    run_tests = False
    keep_werror: Optional[str] = None
    extendable = False
    transitive_rpaths = True
    non_bindable_shared_objects: List[str] = []
    sanity_check_is_file: List[str] = []
    sanity_check_is_dir: List[str] = []
    manual_download = False
    fetch_options: Dict[str, Any] = {}
    license_required = False
    license_comment = '#'
    license_files: List[str] = []
    license_vars: List[str] = []
    license_url = ''
    _verbose = None
    _patches_by_hash = None
    homepage: Optional[str] = None
    list_url: Optional[str] = None
    list_depth = 0
    maintainers: List[str] = []
    metadata_attrs = ['homepage', 'url', 'urls', 'list_url', 'extendable', 'parallel', 'make_jobs', 'maintainers', 'tags']
    test_requires_compiler: bool = False
    test_suite: Optional['TestSuite'] = None

    def __init__(self, spec):
        if False:
            return 10
        self.spec: 'spack.spec.Spec' = spec
        self.path = None
        self.installed_from_binary_cache = False
        if getattr(self, 'url', None) and getattr(self, 'urls', None):
            msg = "a package can have either a 'url' or a 'urls' attribute"
            msg += " [package '{0.name}' defines both]"
            raise ValueError(msg.format(self))
        self._stage = None
        self._fetcher = None
        self._tester: Optional['PackageTest'] = None
        self._fetch_time = 0.0
        self.win_rpath = fsys.WindowsSimulatedRPath(self)
        if self.is_extension:
            pkg_cls = spack.repo.PATH.get_pkg_class(self.extendee_spec.name)
            pkg_cls(self.extendee_spec)._check_extendable()
        super().__init__()

    @classmethod
    def possible_dependencies(cls, transitive=True, expand_virtuals=True, depflag: dt.DepFlag=dt.ALL, visited=None, missing=None, virtuals=None):
        if False:
            return 10
        "Return dict of possible dependencies of this package.\n\n        Args:\n            transitive (bool or None): return all transitive dependencies if\n                True, only direct dependencies if False (default True)..\n            expand_virtuals (bool or None): expand virtual dependencies into\n                all possible implementations (default True)\n            depflag: dependency types to consider\n            visited (dict or None): dict of names of dependencies visited so\n                far, mapped to their immediate dependencies' names.\n            missing (dict or None): dict to populate with packages and their\n                *missing* dependencies.\n            virtuals (set): if provided, populate with virtuals seen so far.\n\n        Returns:\n            (dict): dictionary mapping dependency names to *their*\n                immediate dependencies\n\n        Each item in the returned dictionary maps a (potentially\n        transitive) dependency of this package to its possible\n        *immediate* dependencies. If ``expand_virtuals`` is ``False``,\n        virtual package names wil be inserted as keys mapped to empty\n        sets of dependencies.  Virtuals, if not expanded, are treated as\n        though they have no immediate dependencies.\n\n        Missing dependencies by default are ignored, but if a\n        missing dict is provided, it will be populated with package names\n        mapped to any dependencies they have that are in no\n        repositories. This is only populated if transitive is True.\n\n        Note: the returned dict *includes* the package itself.\n\n        "
        visited = {} if visited is None else visited
        missing = {} if missing is None else missing
        visited.setdefault(cls.name, set())
        for (name, conditions) in cls.dependencies.items():
            depflag_union = 0
            for dep in conditions.values():
                depflag_union |= dep.depflag
            if not depflag & depflag_union:
                continue
            if spack.repo.PATH.is_virtual(name):
                if virtuals is not None:
                    virtuals.add(name)
                if expand_virtuals:
                    providers = spack.repo.PATH.providers_for(name)
                    dep_names = [spec.name for spec in providers]
                else:
                    visited.setdefault(cls.name, set()).add(name)
                    visited.setdefault(name, set())
                    continue
            else:
                dep_names = [name]
            visited.setdefault(cls.name, set()).update(set(dep_names))
            for dep_name in dep_names:
                if dep_name in visited:
                    continue
                visited.setdefault(dep_name, set())
                if not transitive:
                    continue
                try:
                    dep_cls = spack.repo.PATH.get_pkg_class(dep_name)
                except spack.repo.UnknownPackageError:
                    missing.setdefault(cls.name, set()).add(dep_name)
                    continue
                dep_cls.possible_dependencies(transitive, expand_virtuals, depflag, visited, missing, virtuals)
        return visited

    @classproperty
    def package_dir(cls):
        if False:
            for i in range(10):
                print('nop')
        'Directory where the package.py file lives.'
        return os.path.abspath(os.path.dirname(cls.module.__file__))

    @classproperty
    def module(cls):
        if False:
            print('Hello World!')
        'Module object (not just the name) that this package is defined in.\n\n        We use this to add variables to package modules.  This makes\n        install() methods easier to write (e.g., can call configure())\n        '
        return __import__(cls.__module__, fromlist=[cls.__name__])

    @classproperty
    def namespace(cls):
        if False:
            while True:
                i = 10
        'Spack namespace for the package, which identifies its repo.'
        return spack.repo.namespace_from_fullname(cls.__module__)

    @classproperty
    def fullname(cls):
        if False:
            return 10
        'Name of this package, including the namespace'
        return '%s.%s' % (cls.namespace, cls.name)

    @classproperty
    def fullnames(cls):
        if False:
            for i in range(10):
                print('nop')
        'Fullnames for this package and any packages from which it inherits.'
        fullnames = []
        for cls in inspect.getmro(cls):
            namespace = getattr(cls, 'namespace', None)
            if namespace:
                fullnames.append('%s.%s' % (namespace, cls.name))
            if namespace == 'builtin':
                break
        return fullnames

    @classproperty
    def name(cls):
        if False:
            for i in range(10):
                print('nop')
        'The name of this package.\n\n        The name of a package is the name of its Python module, without\n        the containing module names.\n        '
        if cls._name is None:
            cls._name = cls.module.__name__
            if '.' in cls._name:
                cls._name = cls._name[cls._name.rindex('.') + 1:]
        return cls._name

    @classproperty
    def global_license_dir(cls):
        if False:
            for i in range(10):
                print('nop')
        'Returns the directory where license files for all packages are stored.'
        return spack.util.path.canonicalize_path(spack.config.get('config:license_dir'))

    @property
    def global_license_file(self):
        if False:
            return 10
        'Returns the path where a global license file for this\n        particular package should be stored.'
        if not self.license_files:
            return
        return os.path.join(self.global_license_dir, self.name, os.path.basename(self.license_files[0]))

    @property
    def version(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.spec.versions.concrete:
            raise ValueError('Version requested for a package that does not have a concrete version.')
        return self.spec.versions[0]

    @classmethod
    @memoized
    def version_urls(cls):
        if False:
            return 10
        "OrderedDict of explicitly defined URLs for versions of this package.\n\n        Return:\n           An OrderedDict (version -> URL) different versions of this\n           package, sorted by version.\n\n        A version's URL only appears in the result if it has an an\n        explicitly defined ``url`` argument. So, this list may be empty\n        if a package only defines ``url`` at the top level.\n        "
        version_urls = collections.OrderedDict()
        for (v, args) in sorted(cls.versions.items()):
            if 'url' in args:
                version_urls[v] = args['url']
        return version_urls

    def nearest_url(self, version):
        if False:
            print('Hello World!')
        'Finds the URL with the "closest" version to ``version``.\n\n        This uses the following precedence order:\n\n          1. Find the next lowest or equal version with a URL.\n          2. If no lower URL, return the next *higher* URL.\n          3. If no higher URL, return None.\n\n        '
        version_urls = self.version_urls()
        if version in version_urls:
            return version_urls[version]
        last_url = None
        for (v, u) in self.version_urls().items():
            if v > version:
                if last_url:
                    return last_url
            last_url = u
        return last_url

    def url_for_version(self, version):
        if False:
            for i in range(10):
                print('nop')
        'Returns a URL from which the specified version of this package\n        may be downloaded.\n\n        version: class Version\n            The version for which a URL is sought.\n\n        See Class Version (version.py)\n        '
        return self._implement_all_urls_for_version(version)[0]

    def update_external_dependencies(self, extendee_spec=None):
        if False:
            while True:
                i = 10
        '\n        Method to override in package classes to handle external dependencies\n        '
        pass

    def all_urls_for_version(self, version):
        if False:
            i = 10
            return i + 15
        'Return all URLs derived from version_urls(), url, urls, and\n        list_url (if it contains a version) in a package in that order.\n\n        Args:\n            version (spack.version.Version): the version for which a URL is sought\n        '
        uf = None
        if type(self).url_for_version != PackageBase.url_for_version:
            uf = self.url_for_version
        return self._implement_all_urls_for_version(version, uf)

    def _implement_all_urls_for_version(self, version, custom_url_for_version=None):
        if False:
            print('Hello World!')
        if not isinstance(version, StandardVersion):
            version = Version(version)
        urls = []
        version_urls = self.version_urls()
        if version in version_urls:
            urls.append(version_urls[version])
        if custom_url_for_version is not None:
            u = custom_url_for_version(version)
            if u not in urls and u is not None:
                urls.append(u)

        def sub_and_add(u):
            if False:
                while True:
                    i = 10
            if u is None:
                return
            try:
                spack.url.parse_version(u)
            except spack.url.UndetectableVersionError:
                return
            nu = spack.url.substitute_version(u, self.url_version(version))
            urls.append(nu)
        sub_and_add(getattr(self, 'url', None))
        for u in getattr(self, 'urls', []):
            sub_and_add(u)
        sub_and_add(getattr(self, 'list_url', None))
        if not urls:
            default_url = getattr(self, 'url', getattr(self, 'urls', [None])[0])
            if not default_url:
                default_url = self.nearest_url(version)
                if not default_url:
                    raise NoURLError(self.__class__)
            urls.append(spack.url.substitute_version(default_url, self.url_version(version)))
        return urls

    def find_valid_url_for_version(self, version):
        if False:
            return 10
        'Returns a URL from which the specified version of this package\n        may be downloaded after testing whether the url is valid. Will try\n        url, urls, and list_url before failing.\n\n        version: class Version\n            The version for which a URL is sought.\n\n        See Class Version (version.py)\n        '
        urls = self.all_urls_for_version(version)
        for u in urls:
            if spack.util.web.url_exists(u):
                return u
        return None

    def _make_resource_stage(self, root_stage, resource):
        if False:
            while True:
                i = 10
        pretty_resource_name = fsys.polite_filename(f'{resource.name}-{self.version}')
        return ResourceStage(resource.fetcher, root=root_stage, resource=resource, name=self._resource_stage(resource), mirror_paths=spack.mirror.mirror_archive_paths(resource.fetcher, os.path.join(self.name, pretty_resource_name)), path=self.path)

    def _download_search(self):
        if False:
            while True:
                i = 10
        dynamic_fetcher = fs.from_list_url(self)
        return [dynamic_fetcher] if dynamic_fetcher else []

    def _make_root_stage(self, fetcher):
        if False:
            return 10
        format_string = '{name}-{version}'
        pretty_name = self.spec.format_path(format_string)
        mirror_paths = spack.mirror.mirror_archive_paths(fetcher, os.path.join(self.name, pretty_name), self.spec)
        s = self.spec
        stage_name = compute_stage_name(s)
        stage = Stage(fetcher, mirror_paths=mirror_paths, name=stage_name, path=self.path, search_fn=self._download_search)
        return stage

    def _make_stage(self):
        if False:
            while True:
                i = 10
        dev_path_var = self.spec.variants.get('dev_path', None)
        if dev_path_var:
            return DIYStage(dev_path_var.value)
        source_stage = self._make_root_stage(self.fetcher)
        all_stages = StageComposite()
        all_stages.append(source_stage)
        all_stages.extend((self._make_resource_stage(source_stage, r) for r in self._get_needed_resources()))
        all_stages.extend((p.stage for p in self.spec.patches if isinstance(p, spack.patch.UrlPatch)))
        return all_stages

    @property
    def stage(self):
        if False:
            return 10
        "Get the build staging area for this package.\n\n        This automatically instantiates a ``Stage`` object if the package\n        doesn't have one yet, but it does not create the Stage directory\n        on the filesystem.\n        "
        if not self.spec.versions.concrete:
            raise ValueError('Cannot retrieve stage for package without concrete version.')
        if self._stage is None:
            self._stage = self._make_stage()
        return self._stage

    @stage.setter
    def stage(self, stage):
        if False:
            i = 10
            return i + 15
        'Allow a stage object to be set to override the default.'
        self._stage = stage

    @property
    def env_path(self):
        if False:
            return 10
        'Return the build environment file path associated with staging.'
        old_filename = os.path.join(self.stage.path, 'spack-build.env')
        if os.path.exists(old_filename):
            return old_filename
        else:
            return os.path.join(self.stage.path, _spack_build_envfile)

    @property
    def env_mods_path(self):
        if False:
            while True:
                i = 10
        '\n        Return the build environment modifications file path associated with\n        staging.\n        '
        return os.path.join(self.stage.path, _spack_build_envmodsfile)

    @property
    def metadata_dir(self):
        if False:
            return 10
        'Return the install metadata directory.'
        return spack.store.STORE.layout.metadata_path(self.spec)

    @property
    def install_env_path(self):
        if False:
            while True:
                i = 10
        '\n        Return the build environment file path on successful installation.\n        '
        old_filename = os.path.join(self.metadata_dir, 'build.env')
        if os.path.exists(old_filename):
            return old_filename
        else:
            return os.path.join(self.metadata_dir, _spack_build_envfile)

    @property
    def log_path(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the build log file path associated with staging.'
        for filename in ['spack-build.out', 'spack-build.txt']:
            old_log = os.path.join(self.stage.path, filename)
            if os.path.exists(old_log):
                return old_log
        return os.path.join(self.stage.path, _spack_build_logfile)

    @property
    def phase_log_files(self):
        if False:
            while True:
                i = 10
        'Find sorted phase log files written to the staging directory'
        logs_dir = os.path.join(self.stage.path, 'spack-build-*-out.txt')
        log_files = glob.glob(logs_dir)
        log_files.sort()
        return log_files

    @property
    def install_log_path(self):
        if False:
            i = 10
            return i + 15
        'Return the build log file path on successful installation.'
        for filename in ['build.out', 'build.txt']:
            old_log = os.path.join(self.metadata_dir, filename)
            if os.path.exists(old_log):
                return old_log
        return os.path.join(self.metadata_dir, _spack_build_logfile)

    @property
    def configure_args_path(self):
        if False:
            while True:
                i = 10
        'Return the configure args file path associated with staging.'
        return os.path.join(self.stage.path, _spack_configure_argsfile)

    @property
    def times_log_path(self):
        if False:
            while True:
                i = 10
        'Return the times log json file.'
        return os.path.join(self.metadata_dir, spack_times_log)

    @property
    def install_configure_args_path(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the configure args file path on successful installation.'
        return os.path.join(self.metadata_dir, _spack_configure_argsfile)

    @property
    def install_test_root(self):
        if False:
            return 10
        'Return the install test root directory.'
        tty.warn("The 'pkg.install_test_root' property is deprecated with removal expected v0.22. Use 'install_test_root(pkg)' instead.")
        return install_test_root(self)

    def archive_install_test_log(self):
        if False:
            print('Hello World!')
        'Archive the install-phase test log, if present.'
        if getattr(self, 'tester', None):
            self.tester.archive_install_test_log(self.metadata_dir)

    @property
    def tester(self):
        if False:
            return 10
        if not self.spec.versions.concrete:
            raise ValueError('Cannot retrieve tester for package without concrete version.')
        if not self._tester:
            self._tester = PackageTest(self)
        return self._tester

    @property
    def installed(self):
        if False:
            i = 10
            return i + 15
        msg = 'the "PackageBase.installed" property is deprecated and will be removed in Spack v0.19, use "Spec.installed" instead'
        warnings.warn(msg)
        return self.spec.installed

    @property
    def installed_upstream(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'the "PackageBase.installed_upstream" property is deprecated and will be removed in Spack v0.19, use "Spec.installed_upstream" instead'
        warnings.warn(msg)
        return self.spec.installed_upstream

    @property
    def fetcher(self):
        if False:
            while True:
                i = 10
        if not self.spec.versions.concrete:
            raise ValueError('Cannot retrieve fetcher for package without concrete version.')
        if not self._fetcher:
            self._fetcher = fs.for_package_version(self)
        return self._fetcher

    @fetcher.setter
    def fetcher(self, f):
        if False:
            return 10
        self._fetcher = f
        self._fetcher.set_package(self)

    @classmethod
    def dependencies_of_type(cls, deptypes: dt.DepFlag):
        if False:
            return 10
        'Get dependencies that can possibly have these deptypes.\n\n        This analyzes the package and determines which dependencies *can*\n        be a certain kind of dependency. Note that they may not *always*\n        be this kind of dependency, since dependencies can be optional,\n        so something may be a build dependency in one configuration and a\n        run dependency in another.\n        '
        return dict(((name, conds) for (name, conds) in cls.dependencies.items() if any((deptypes & cls.dependencies[name][cond].depflag for cond in conds))))

    @property
    def extendee_spec(self):
        if False:
            return 10
        '\n        Spec of the extendee of this package, or None if it is not an extension\n        '
        if not self.extendees:
            return None
        deps = []
        for dep in self.spec.traverse(deptype=('link', 'run')):
            if dep.name in self.extendees:
                deps.append(dep)
        if deps:
            assert len(deps) == 1
            return deps[0]
        if self.spec._concrete:
            return None
        else:
            (spec_str, kwargs) = next(iter(self.extendees.items()))
            return spack.spec.Spec(spec_str)

    @property
    def extendee_args(self):
        if False:
            print('Hello World!')
        '\n        Spec of the extendee of this package, or None if it is not an extension\n        '
        if not self.extendees:
            return None
        name = next(iter(self.extendees))
        return self.extendees[name][1]

    @property
    def is_extension(self):
        if False:
            i = 10
            return i + 15
        if self.spec._concrete:
            return self.extendee_spec is not None
        else:
            return bool(self.extendees)

    def extends(self, spec):
        if False:
            i = 10
            return i + 15
        '\n        Returns True if this package extends the given spec.\n\n        If ``self.spec`` is concrete, this returns whether this package extends\n        the given spec.\n\n        If ``self.spec`` is not concrete, this returns whether this package may\n        extend the given spec.\n        '
        if spec.name not in self.extendees:
            return False
        s = self.extendee_spec
        return s and spec.satisfies(s)

    def provides(self, vpkg_name):
        if False:
            return 10
        '\n        True if this package provides a virtual package with the specified name\n        '
        return any((any((self.spec.intersects(c) for c in constraints)) for (s, constraints) in self.provided.items() if s.name == vpkg_name))

    @property
    def virtuals_provided(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        virtual packages provided by this package with its spec\n        '
        return [vspec for (vspec, constraints) in self.provided.items() if any((self.spec.satisfies(c) for c in constraints))]

    @property
    def prefix(self):
        if False:
            return 10
        'Get the prefix into which this package should be installed.'
        return self.spec.prefix

    @property
    def home(self):
        if False:
            print('Hello World!')
        return self.prefix

    @property
    @memoized
    def compiler(self):
        if False:
            return 10
        'Get the spack.compiler.Compiler object used to build this package'
        if not self.spec.concrete:
            raise ValueError('Can only get a compiler for a concrete package.')
        return spack.compilers.compiler_for_spec(self.spec.compiler, self.spec.architecture)

    def url_version(self, version):
        if False:
            for i in range(10):
                print('nop')
        "\n        Given a version, this returns a string that should be substituted\n        into the package's URL to download that version.\n\n        By default, this just returns the version string. Subclasses may need\n        to override this, e.g. for boost versions where you need to ensure that\n        there are _'s in the download URL.\n        "
        return str(version)

    def remove_prefix(self):
        if False:
            while True:
                i = 10
        '\n        Removes the prefix for a package along with any empty parent\n        directories\n        '
        spack.store.STORE.layout.remove_install_directory(self.spec)

    @property
    def download_instr(self):
        if False:
            print('Hello World!')
        '\n        Defines the default manual download instructions.  Packages can\n        override the property to provide more information.\n\n        Returns:\n            (str):  default manual download instructions\n        '
        required = 'Manual download is required for {0}. '.format(self.spec.name) if self.manual_download else ''
        return '{0}Refer to {1} for download instructions.'.format(required, self.spec.package.homepage)

    def do_fetch(self, mirror_only=False):
        if False:
            return 10
        '\n        Creates a stage directory and downloads the tarball for this package.\n        Working directory will be set to the stage directory.\n        '
        if not self.has_code or self.spec.external:
            tty.debug('No fetch required for {0}'.format(self.name))
            return
        checksum = spack.config.get('config:checksum')
        fetch = self.stage.managed_by_spack
        if checksum and fetch and (self.version not in self.versions) and (not isinstance(self.version, GitVersion)):
            tty.warn('There is no checksum on file to fetch %s safely.' % self.spec.cformat('{name}{@version}'))
            ck_msg = 'Add a checksum or use --no-checksum to skip this check.'
            ignore_checksum = False
            if sys.stdout.isatty():
                ignore_checksum = tty.get_yes_or_no('  Fetch anyway?', default=False)
                if ignore_checksum:
                    tty.debug('Fetching with no checksum. {0}'.format(ck_msg))
            if not ignore_checksum:
                raise spack.error.FetchError('Will not fetch %s' % self.spec.format('{name}{@version}'), ck_msg)
        deprecated = spack.config.get('config:deprecated')
        if not deprecated and self.versions.get(self.version, {}).get('deprecated', False):
            tty.warn('{0} is deprecated and may be removed in a future Spack release.'.format(self.spec.format('{name}{@version}')))
            dp_msg = 'If you are willing to be a maintainer for this version of the package, submit a PR to remove `deprecated=False`, or use `--deprecated` to skip this check.'
            ignore_deprecation = False
            if sys.stdout.isatty():
                ignore_deprecation = tty.get_yes_or_no('  Fetch anyway?', default=False)
                if ignore_deprecation:
                    tty.debug('Fetching deprecated version. {0}'.format(dp_msg))
            if not ignore_deprecation:
                raise spack.error.FetchError('Will not fetch {0}'.format(self.spec.format('{name}{@version}')), dp_msg)
        self.stage.create()
        err_msg = None if not self.manual_download else self.download_instr
        start_time = time.time()
        self.stage.fetch(mirror_only, err_msg=err_msg)
        self._fetch_time = time.time() - start_time
        if checksum and self.version in self.versions:
            self.stage.check()
        self.stage.cache_local()

    def do_stage(self, mirror_only=False):
        if False:
            i = 10
            return i + 15
        'Unpacks and expands the fetched tarball.'
        self.stage.create()
        if self.has_code:
            self.do_fetch(mirror_only)
            self.stage.expand_archive()
            if not os.listdir(self.stage.path):
                raise spack.error.FetchError('Archive was empty for %s' % self.name)
        else:
            fsys.mkdirp(self.stage.source_path)

    def do_patch(self):
        if False:
            for i in range(10):
                print('nop')
        "Applies patches if they haven't been applied already."
        if not self.spec.concrete:
            raise ValueError('Can only patch concrete packages.')
        self.do_stage()
        has_patch_fun = hasattr(self, 'patch') and callable(self.patch)
        patches = self.spec.patches
        if not patches and (not has_patch_fun):
            tty.msg('No patches needed for {0}'.format(self.name))
            return
        archive_dir = self.stage.source_path
        good_file = os.path.join(archive_dir, '.spack_patched')
        no_patches_file = os.path.join(archive_dir, '.spack_no_patches')
        bad_file = os.path.join(archive_dir, '.spack_patch_failed')
        if os.path.isfile(bad_file):
            if self.stage.managed_by_spack:
                tty.debug('Patching failed last time. Restaging.')
                self.stage.restage()
            else:
                msg = 'A patch failure was detected in %s.' % self.name + ' Build errors may occur due to this.'
                tty.warn(msg)
                return
        if os.path.isfile(good_file):
            tty.msg('Already patched {0}'.format(self.name))
            return
        elif os.path.isfile(no_patches_file):
            tty.msg('No patches needed for {0}'.format(self.name))
            return
        patched = False
        for patch in patches:
            try:
                with fsys.working_dir(self.stage.source_path):
                    patch.apply(self.stage)
                tty.msg('Applied patch {0}'.format(patch.path_or_url))
                patched = True
            except spack.error.SpackError as e:
                tty.debug(e)
                tty.msg('Patch %s failed.' % patch.path_or_url)
                fsys.touch(bad_file)
                raise
        if has_patch_fun:
            try:
                with fsys.working_dir(self.stage.source_path):
                    self.patch()
                tty.msg('Ran patch() for {0}'.format(self.name))
                patched = True
            except spack.multimethod.NoSuchMethodError:
                if not patched:
                    tty.msg('No patches needed for {0}'.format(self.name))
            except spack.error.SpackError as e:
                tty.debug(e)
                tty.msg('patch() function failed for {0}'.format(self.name))
                fsys.touch(bad_file)
                raise
        if os.path.isfile(bad_file):
            os.remove(bad_file)
        if patched:
            fsys.touch(good_file)
        else:
            fsys.touch(no_patches_file)

    @classmethod
    def all_patches(cls):
        if False:
            print('Hello World!')
        'Retrieve all patches associated with the package.\n\n        Retrieves patches on the package itself as well as patches on the\n        dependencies of the package.'
        patches = []
        for (_, patch_list) in cls.patches.items():
            for patch in patch_list:
                patches.append(patch)
        pkg_deps = cls.dependencies
        for dep_name in pkg_deps:
            for (_, dependency) in pkg_deps[dep_name].items():
                for (_, patch_list) in dependency.patches.items():
                    for patch in patch_list:
                        patches.append(patch)
        return patches

    def content_hash(self, content=None):
        if False:
            print('Hello World!')
        "Create a hash based on the artifacts and patches used to build this package.\n\n        This includes:\n            * source artifacts (tarballs, repositories) used to build;\n            * content hashes (``sha256``'s) of all patches applied by Spack; and\n            * canonicalized contents the ``package.py`` recipe used to build.\n\n        This hash is only included in Spack's DAG hash for concrete specs, but if it\n        happens to be called on a package with an abstract spec, only applicable (i.e.,\n        determinable) portions of the hash will be included.\n\n        "
        hash_content = []
        if self.spec.versions.concrete:
            try:
                source_id = fs.for_package_version(self).source_id()
            except (fs.ExtrapolationError, fs.InvalidArgsError):
                source_id = None
            if not source_id:
                env = spack.environment.active_environment()
                from_local_sources = env and env.is_develop(self.spec)
                if self.has_code and (not self.spec.external) and (not from_local_sources):
                    message = 'Missing a source id for {s.name}@{s.version}'
                    tty.debug(message.format(s=self))
                hash_content.append(''.encode('utf-8'))
            else:
                hash_content.append(source_id.encode('utf-8'))
        if self.spec._patches_assigned():
            hash_content.extend((':'.join((p.sha256, str(p.level))).encode('utf-8') for p in self.spec.patches))
        hash_content.append(package_hash(self.spec, source=content).encode('utf-8'))
        b32_hash = base64.b32encode(hashlib.sha256(bytes().join(sorted(hash_content))).digest()).lower()
        b32_hash = b32_hash.decode('utf-8')
        return b32_hash

    @property
    def cmake_prefix_paths(self):
        if False:
            return 10
        return [self.prefix]

    def _has_make_target(self, target):
        if False:
            i = 10
            return i + 15
        "Checks to see if 'target' is a valid target in a Makefile.\n\n        Parameters:\n            target (str): the target to check for\n\n        Returns:\n            bool: True if 'target' is found, else False\n        "
        make = copy.deepcopy(inspect.getmodule(self).make)
        make.add_default_env('LC_ALL', 'C')
        for makefile in ['GNUmakefile', 'Makefile', 'makefile']:
            if os.path.exists(makefile):
                break
        else:
            tty.debug('No Makefile found in the build directory')
            return False
        missing_target_msgs = ["No rule to make target `{0}'.  Stop.", "No rule to make target '{0}'.  Stop.", "don't know how to make {0}. Stop"]
        kwargs = {'fail_on_error': False, 'output': os.devnull, 'error': str}
        stderr = make('-n', target, **kwargs)
        for missing_target_msg in missing_target_msgs:
            if missing_target_msg.format(target) in stderr:
                tty.debug("Target '{0}' not found in {1}".format(target, makefile))
                return False
        return True

    def _if_make_target_execute(self, target, *args, **kwargs):
        if False:
            print('Hello World!')
        "Runs ``make target`` if 'target' is a valid target in the Makefile.\n\n        Parameters:\n            target (str): the target to potentially execute\n        "
        if self._has_make_target(target):
            inspect.getmodule(self).make(target, *args, **kwargs)

    def _has_ninja_target(self, target):
        if False:
            i = 10
            return i + 15
        "Checks to see if 'target' is a valid target in a Ninja build script.\n\n        Parameters:\n            target (str): the target to check for\n\n        Returns:\n            bool: True if 'target' is found, else False\n        "
        ninja = inspect.getmodule(self).ninja
        if not os.path.exists('build.ninja'):
            tty.debug('No Ninja build script found in the build directory')
            return False
        all_targets = ninja('-t', 'targets', 'all', output=str).split('\n')
        matches = [line for line in all_targets if line.startswith(target + ':')]
        if not matches:
            tty.debug("Target '{0}' not found in build.ninja".format(target))
            return False
        return True

    def _if_ninja_target_execute(self, target, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "Runs ``ninja target`` if 'target' is a valid target in the Ninja\n        build script.\n\n        Parameters:\n            target (str): the target to potentially execute\n        "
        if self._has_ninja_target(target):
            inspect.getmodule(self).ninja(target, *args, **kwargs)

    def _get_needed_resources(self):
        if False:
            for i in range(10):
                print('nop')
        resources = []
        if self.spec.concrete:
            for (when_spec, resource_list) in self.resources.items():
                if when_spec in self.spec:
                    resources.extend(resource_list)
        else:
            for (when_spec, resource_list) in self.resources.items():
                if when_spec.intersects(self.spec):
                    resources.extend(resource_list)
        resources = sorted(resources, key=lambda res: len(res.destination))
        return resources

    def _resource_stage(self, resource):
        if False:
            return 10
        pieces = ['resource', resource.name, self.spec.dag_hash()]
        resource_stage_folder = '-'.join(pieces)
        return resource_stage_folder

    def do_install(self, **kwargs):
        if False:
            print('Hello World!')
        "Called by commands to install a package and or its dependencies.\n\n        Package implementations should override install() to describe\n        their build process.\n\n        Args:\n            cache_only (bool): Fail if binary package unavailable.\n            dirty (bool): Don't clean the build environment before installing.\n            explicit (bool): True if package was explicitly installed, False\n                if package was implicitly installed (as a dependency).\n            fail_fast (bool): Fail if any dependency fails to install;\n                otherwise, the default is to install as many dependencies as\n                possible (i.e., best effort installation).\n            fake (bool): Don't really build; install fake stub files instead.\n            force (bool): Install again, even if already installed.\n            install_deps (bool): Install dependencies before installing this\n                package\n            install_source (bool): By default, source is not installed, but\n                for debugging it might be useful to keep it around.\n            keep_prefix (bool): Keep install prefix on failure. By default,\n                destroys it.\n            keep_stage (bool): By default, stage is destroyed only if there\n                are no exceptions during build. Set to True to keep the stage\n                even with exceptions.\n            restage (bool): Force spack to restage the package source.\n            skip_patch (bool): Skip patch stage of build if True.\n            stop_before (str): stop execution before this\n                installation phase (or None)\n            stop_at (str): last installation phase to be executed\n                (or None)\n            tests (bool or list or set): False to run no tests, True to test\n                all packages, or a list of package names to run tests for some\n            use_cache (bool): Install from binary package, if available.\n            verbose (bool): Display verbose build output (by default,\n                suppresses it)\n        "
        PackageInstaller([(self, kwargs)]).install()

    def cache_extra_test_sources(self, srcs):
        if False:
            return 10
        'Copy relative source paths to the corresponding install test subdir\n\n        This method is intended as an optional install test setup helper for\n        grabbing source files/directories during the installation process and\n        copying them to the installation test subdirectory for subsequent use\n        during install testing.\n\n        Args:\n            srcs (str or list): relative path for files and or\n                subdirectories located in the staged source path that are to\n                be copied to the corresponding location(s) under the install\n                testing directory.\n        '
        msg = "'pkg.cache_extra_test_sources(srcs) is deprecated with removal expected in v0.22. Use 'cache_extra_test_sources(pkg, srcs)' instead."
        warnings.warn(msg)
        cache_extra_test_sources(self, srcs)

    def do_test(self, dirty=False, externals=False):
        if False:
            print('Hello World!')
        if self.test_requires_compiler:
            compilers = spack.compilers.compilers_for_spec(self.spec.compiler, arch_spec=self.spec.architecture)
            if not compilers:
                tty.error('Skipping tests for package %s\n' % self.spec.format('{name}-{version}-{hash:7}') + 'Package test requires missing compiler %s' % self.spec.compiler)
                return
        kwargs = {'dirty': dirty, 'fake': False, 'context': 'test', 'externals': externals, 'verbose': tty.is_verbose()}
        self.tester.stand_alone_tests(kwargs)

    @property
    def _test_deprecated_warning(self):
        if False:
            return 10
        alt = f"Use any name starting with 'test_' instead in {self.spec.name}."
        return f"The 'test' method is deprecated. {alt}"

    def test(self):
        if False:
            i = 10
            return i + 15
        warnings.warn(self._test_deprecated_warning)

    def run_test(self, exe, options=[], expected=[], status=0, installed=False, purpose=None, skip_missing=False, work_dir=None):
        if False:
            print('Hello World!')
        'Run the test and confirm the expected results are obtained\n\n        Log any failures and continue, they will be re-raised later\n\n        Args:\n            exe (str): the name of the executable\n            options (str or list): list of options to pass to the runner\n            expected (str or list): list of expected output strings.\n                Each string is a regex expected to match part of the output.\n            status (int or list): possible passing status values\n                with 0 meaning the test is expected to succeed\n            installed (bool): if ``True``, the executable must be in the\n                install prefix\n            purpose (str): message to display before running test\n            skip_missing (bool): skip the test if the executable is not\n                in the install prefix bin directory or the provided work_dir\n            work_dir (str or None): path to the smoke test directory\n        '

        def test_title(purpose, test_name):
            if False:
                return 10
            if not purpose:
                return f'test: {test_name}: execute {test_name}'
            match = re.search('test: ([^:]*): (.*)', purpose)
            if match:
                return purpose
            match = re.search('test: (.*)', purpose)
            if match:
                reason = match.group(1)
                return f'test: {test_name}: {reason}'
            return f'test: {test_name}: {purpose}'
        base_exe = os.path.basename(exe)
        alternate = f"Use 'test_part' instead for {self.spec.name} to process {base_exe}."
        warnings.warn(f"The 'run_test' method is deprecated. {alternate}")
        extra = re.compile('[\\s,\\- ]')
        details = [extra.sub('', options)] if isinstance(options, str) else [extra.sub('', os.path.basename(opt)) for opt in options]
        details = '_'.join([''] + details) if details else ''
        test_name = f'test_{base_exe}{details}'
        tty.info(test_title(purpose, test_name), format='g')
        wdir = '.' if work_dir is None else work_dir
        with fsys.working_dir(wdir, create=True):
            try:
                runner = which(exe)
                if runner is None and skip_missing:
                    self.tester.status(test_name, TestStatus.SKIPPED, f'{exe} is missing')
                    return
                assert runner is not None, f"Failed to find executable '{exe}'"
                self._run_test_helper(runner, options, expected, status, installed, purpose)
                self.tester.status(test_name, TestStatus.PASSED, None)
                return True
            except (AssertionError, BaseException) as e:
                (exc_type, _, tb) = sys.exc_info()
                self.tester.status(test_name, TestStatus.FAILED, str(e))
                import traceback
                stack = traceback.extract_stack()[:-1]
                for (i, entry) in enumerate(stack):
                    (filename, lineno, function, text) = entry
                    if spack.repo.is_package_file(filename):
                        with open(filename, 'r') as f:
                            lines = f.readlines()
                        new_lineno = lineno - 2
                        text = lines[new_lineno]
                        stack[i] = (filename, new_lineno, function, text)
                out = traceback.format_list(stack)
                for line in out:
                    print(line.rstrip('\n'))
                if exc_type is spack.util.executable.ProcessError:
                    out = io.StringIO()
                    spack.build_environment.write_log_summary(out, 'test', self.tester.test_log_file, last=1)
                    m = out.getvalue()
                else:
                    context = spack.build_environment.get_package_context(tb)
                    m = '\n'.join(context) if context else ''
                exc = e
                if spack.config.get('config:fail_fast', False):
                    raise TestFailure([(exc, m)])
                else:
                    self.tester.add_failure(exc, m)
                return False

    def _run_test_helper(self, runner, options, expected, status, installed, purpose):
        if False:
            while True:
                i = 10
        status = [status] if isinstance(status, int) else status
        expected = [expected] if isinstance(expected, str) else expected
        options = [options] if isinstance(options, str) else options
        if installed:
            msg = f"Executable '{runner.name}' expected in prefix, "
            msg += f'found in {runner.path} instead'
            assert runner.path.startswith(self.spec.prefix), msg
        tty.msg(f'Expecting return code in {status}')
        try:
            output = runner(*options, output=str.split, error=str.split)
            assert 0 in status, f'Expected {runner.name} execution to fail'
        except ProcessError as err:
            output = str(err)
            match = re.search('exited with status ([0-9]+)', output)
            if not (match and int(match.group(1)) in status):
                raise
        for check in expected:
            cmd = ' '.join([runner.name] + options)
            msg = f"Expected '{check}' to match output of `{cmd}`"
            msg += f'\n\nOutput: {output}'
            assert re.search(check, output), msg

    def unit_test_check(self):
        if False:
            for i in range(10):
                print('nop')
        'Hook for unit tests to assert things about package internals.\n\n        Unit tests can override this function to perform checks after\n        ``Package.install`` and all post-install hooks run, but before\n        the database is updated.\n\n        The overridden function may indicate that the install procedure\n        should terminate early (before updating the database) by\n        returning ``False`` (or any value such that ``bool(result)`` is\n        ``False``).\n\n        Return:\n            (bool): ``True`` to continue, ``False`` to skip ``install()``\n        '
        return True

    @property
    def build_log_path(self):
        if False:
            print('Hello World!')
        '\n        Return the expected (or current) build log file path.  The path points\n        to the staging build file until the software is successfully installed,\n        when it points to the file in the installation directory.\n        '
        return self.install_log_path if self.spec.installed else self.log_path

    @classmethod
    def inject_flags(cls: Type[Pb], name: str, flags: Iterable[str]) -> FLAG_HANDLER_RETURN_TYPE:
        if False:
            for i in range(10):
                print('nop')
        '\n        flag_handler that injects all flags through the compiler wrapper.\n        '
        return (flags, None, None)

    @classmethod
    def env_flags(cls: Type[Pb], name: str, flags: Iterable[str]) -> FLAG_HANDLER_RETURN_TYPE:
        if False:
            for i in range(10):
                print('nop')
        '\n        flag_handler that adds all flags to canonical environment variables.\n        '
        return (None, flags, None)

    @classmethod
    def build_system_flags(cls: Type[Pb], name: str, flags: Iterable[str]) -> FLAG_HANDLER_RETURN_TYPE:
        if False:
            for i in range(10):
                print('nop')
        '\n        flag_handler that passes flags to the build system arguments.  Any\n        package using `build_system_flags` must also implement\n        `flags_to_build_system_args`, or derive from a class that\n        implements it.  Currently, AutotoolsPackage and CMakePackage\n        implement it.\n        '
        return (None, None, flags)

    def setup_run_environment(self, env):
        if False:
            print('Hello World!')
        'Sets up the run environment for a package.\n\n        Args:\n            env (spack.util.environment.EnvironmentModifications): environment\n                modifications to be applied when the package is run. Package authors\n                can call methods on it to alter the run environment.\n        '
        pass

    def setup_dependent_run_environment(self, env, dependent_spec):
        if False:
            i = 10
            return i + 15
        "Sets up the run environment of packages that depend on this one.\n\n        This is similar to ``setup_run_environment``, but it is used to\n        modify the run environments of packages that *depend* on this one.\n\n        This gives packages like Python and others that follow the extension\n        model a way to implement common environment or run-time settings\n        for dependencies.\n\n        Args:\n            env (spack.util.environment.EnvironmentModifications): environment\n                modifications to be applied when the dependent package is run.\n                Package authors can call methods on it to alter the build environment.\n\n            dependent_spec (spack.spec.Spec): The spec of the dependent package\n                about to be run. This allows the extendee (self) to query\n                the dependent's state. Note that *this* package's spec is\n                available as ``self.spec``\n        "
        pass

    def setup_dependent_package(self, module, dependent_spec):
        if False:
            return 10
        "Set up Python module-scope variables for dependent packages.\n\n        Called before the install() method of dependents.\n\n        Default implementation does nothing, but this can be\n        overridden by an extendable package to set up the module of\n        its extensions. This is useful if there are some common steps\n        to installing all extensions for a certain package.\n\n        Examples:\n\n        1. Extensions often need to invoke the ``python`` interpreter\n           from the Python installation being extended. This routine\n           can put a ``python()`` Executable object in the module scope\n           for the extension package to simplify extension installs.\n\n        2. MPI compilers could set some variables in the dependent's\n           scope that point to ``mpicc``, ``mpicxx``, etc., allowing\n           them to be called by common name regardless of which MPI is used.\n\n        3. BLAS/LAPACK implementations can set some variables\n           indicating the path to their libraries, since these\n           paths differ by BLAS/LAPACK implementation.\n\n        Args:\n            module (spack.package_base.PackageBase.module): The Python ``module``\n                object of the dependent package. Packages can use this to set\n                module-scope variables for the dependent to use.\n\n            dependent_spec (spack.spec.Spec): The spec of the dependent package\n                about to be built. This allows the extendee (self) to\n                query the dependent's state.  Note that *this*\n                package's spec is available as ``self.spec``.\n        "
        pass
    _flag_handler: Optional[FLAG_HANDLER_TYPE] = None

    @property
    def flag_handler(self) -> FLAG_HANDLER_TYPE:
        if False:
            while True:
                i = 10
        if self._flag_handler is None:
            self._flag_handler = PackageBase.inject_flags
        return self._flag_handler

    @flag_handler.setter
    def flag_handler(self, var: FLAG_HANDLER_TYPE) -> None:
        if False:
            while True:
                i = 10
        self._flag_handler = var

    def flags_to_build_system_args(self, flags):
        if False:
            i = 10
            return i + 15
        if any((v for v in flags.values())):
            msg = 'The {0} build system'.format(self.__class__.__name__)
            msg += ' cannot take command line arguments for compiler flags'
            raise NotImplementedError(msg)

    @staticmethod
    def uninstall_by_spec(spec, force=False, deprecator=None):
        if False:
            for i in range(10):
                print('nop')
        if not os.path.isdir(spec.prefix):
            specs = spack.store.STORE.db.query(spec, installed=True)
            if specs:
                if deprecator:
                    spack.store.STORE.db.deprecate(specs[0], deprecator)
                    tty.debug('Deprecating stale DB entry for {0}'.format(spec.short_spec))
                else:
                    spack.store.STORE.db.remove(specs[0])
                    tty.debug('Removed stale DB entry for {0}'.format(spec.short_spec))
                return
            else:
                raise InstallError(str(spec) + ' is not installed.')
        if not force:
            dependents = spack.store.STORE.db.installed_relatives(spec, direction='parents', transitive=True, deptype=('link', 'run'))
            if dependents:
                raise PackageStillNeededError(spec, dependents)
        try:
            pkg = spec.package
        except spack.repo.UnknownEntityError:
            pkg = None
        with spack.store.STORE.prefix_locker.write_lock(spec):
            if pkg is not None:
                try:
                    spack.hooks.pre_uninstall(spec)
                except Exception as error:
                    if force:
                        error_msg = 'One or more pre_uninstall hooks have failed for {0}, but Spack is continuing with the uninstall'.format(str(spec))
                        if isinstance(error, spack.error.SpackError):
                            error_msg += '\n\nError message: {0}'.format(str(error))
                        tty.warn(error_msg)
                    else:
                        raise
            if not spec.external:
                msg = 'Deleting package prefix [{0}]'
                tty.debug(msg.format(spec.short_spec))
                deprecated = bool(spack.store.STORE.db.deprecator(spec))
                spack.store.STORE.layout.remove_install_directory(spec, deprecated)
            if deprecator:
                msg = 'deprecating DB entry [{0}] in favor of [{1}]'
                tty.debug(msg.format(spec.short_spec, deprecator.short_spec))
                spack.store.STORE.db.deprecate(spec, deprecator)
            else:
                msg = 'Deleting DB entry [{0}]'
                tty.debug(msg.format(spec.short_spec))
                spack.store.STORE.db.remove(spec)
        if pkg is not None:
            try:
                spack.hooks.post_uninstall(spec)
            except Exception:
                error_msg = 'One or more post-uninstallation hooks failed for {0}, but the prefix has been removed (if it is not external).'.format(str(spec))
                tb_msg = traceback.format_exc()
                error_msg += '\n\nThe error:\n\n{0}'.format(tb_msg)
                tty.warn(error_msg)
        tty.msg('Successfully uninstalled {0}'.format(spec.short_spec))

    def do_uninstall(self, force=False):
        if False:
            print('Hello World!')
        'Uninstall this package by spec.'
        PackageBase.uninstall_by_spec(self.spec, force)

    def do_deprecate(self, deprecator, link_fn):
        if False:
            i = 10
            return i + 15
        'Deprecate this package in favor of deprecator spec'
        spec = self.spec
        if not spack.store.STORE.db.query(deprecator):
            deprecator.package.do_install()
        old_deprecator = spack.store.STORE.db.deprecator(spec)
        if old_deprecator:
            self_yaml = spack.store.STORE.layout.deprecated_file_path(spec, old_deprecator)
        else:
            self_yaml = spack.store.STORE.layout.spec_file_path(spec)
        depr_yaml = spack.store.STORE.layout.deprecated_file_path(spec, deprecator)
        fsys.mkdirp(os.path.dirname(depr_yaml))
        shutil.copy2(self_yaml, depr_yaml)
        for deprecated in spack.store.STORE.db.specs_deprecated_by(spec):
            deprecated.package.do_deprecate(deprecator, link_fn)
        PackageBase.uninstall_by_spec(spec, force=True, deprecator=deprecator)
        link_fn(deprecator.prefix, spec.prefix)

    def _check_extendable(self):
        if False:
            i = 10
            return i + 15
        if not self.extendable:
            raise ValueError('Package %s is not extendable!' % self.name)

    def view(self):
        if False:
            return 10
        'Create a view with the prefix of this package as the root.\n        Extensions added to this view will modify the installation prefix of\n        this package.\n        '
        return YamlFilesystemView(self.prefix, spack.store.STORE.layout)

    def do_restage(self):
        if False:
            while True:
                i = 10
        'Reverts expanded/checked out source to a pristine state.'
        self.stage.restage()

    def do_clean(self):
        if False:
            for i in range(10):
                print('nop')
        "Removes the package's build stage and source tarball."
        self.stage.destroy()

    @classmethod
    def format_doc(cls, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Wrap doc string at 72 characters and format nicely'
        indent = kwargs.get('indent', 0)
        if not cls.__doc__:
            return ''
        doc = re.sub('\\s+', ' ', cls.__doc__)
        lines = textwrap.wrap(doc, 72)
        results = io.StringIO()
        for line in lines:
            results.write(' ' * indent + line + '\n')
        return results.getvalue()

    @property
    def all_urls(self):
        if False:
            print('Hello World!')
        'A list of all URLs in a package.\n\n        Check both class-level and version-specific URLs.\n\n        Returns:\n            list: a list of URLs\n        '
        urls = []
        if hasattr(self, 'url') and self.url:
            urls.append(self.url)
        if hasattr(self, 'urls') and self.urls:
            urls.append(self.urls[0])
        for args in self.versions.values():
            if 'url' in args:
                urls.append(args['url'])
        return urls

    def fetch_remote_versions(self, concurrency=None):
        if False:
            i = 10
            return i + 15
        'Find remote versions of this package.\n\n        Uses ``list_url`` and any other URLs listed in the package file.\n\n        Returns:\n            dict: a dictionary mapping versions to URLs\n        '
        if not self.all_urls:
            return {}
        try:
            return spack.url.find_versions_of_archive(self.all_urls, self.list_url, self.list_depth, concurrency, reference_package=self)
        except spack.util.web.NoNetworkConnectionError as e:
            tty.die("Package.fetch_versions couldn't connect to:", e.url, e.message)

    @property
    def rpath(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the rpath this package links with, as a list of paths.'
        deps = self.spec.dependencies(deptype='link')
        if sys.platform == 'win32':
            rpaths = [self.prefix.bin]
            rpaths.extend((d.prefix.bin for d in deps if os.path.isdir(d.prefix.bin)))
        else:
            rpaths = [self.prefix.lib, self.prefix.lib64]
            rpaths.extend((d.prefix.lib for d in deps if os.path.isdir(d.prefix.lib)))
            rpaths.extend((d.prefix.lib64 for d in deps if os.path.isdir(d.prefix.lib64)))
        return rpaths

    @property
    def rpath_args(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the rpath args as a string, with -Wl,-rpath, for each element\n        '
        return ' '.join(('-Wl,-rpath,%s' % p for p in self.rpath))

    @property
    def builder(self):
        if False:
            while True:
                i = 10
        return spack.builder.create(self)
inject_flags = PackageBase.inject_flags
env_flags = PackageBase.env_flags
build_system_flags = PackageBase.build_system_flags

def install_dependency_symlinks(pkg, spec, prefix):
    if False:
        return 10
    '\n    Execute a dummy install and flatten dependencies.\n\n    This routine can be used in a ``package.py`` definition by setting\n    ``install = install_dependency_symlinks``.\n\n    This feature comes in handy for creating a common location for the\n    the installation of third-party libraries.\n    '
    flatten_dependencies(spec, prefix)

def use_cray_compiler_names():
    if False:
        for i in range(10):
            print('nop')
    'Compiler names for builds that rely on cray compiler names.'
    os.environ['CC'] = 'cc'
    os.environ['CXX'] = 'CC'
    os.environ['FC'] = 'ftn'
    os.environ['F77'] = 'ftn'

def flatten_dependencies(spec, flat_dir):
    if False:
        print('Hello World!')
    'Make each dependency of spec present in dir via symlink.'
    for dep in spec.traverse(root=False):
        name = dep.name
        dep_path = spack.store.STORE.layout.path_for_spec(dep)
        dep_files = LinkTree(dep_path)
        os.mkdir(flat_dir + '/' + name)
        conflict = dep_files.find_conflict(flat_dir + '/' + name)
        if conflict:
            raise DependencyConflictError(conflict)
        dep_files.merge(flat_dir + '/' + name)

def possible_dependencies(*pkg_or_spec, **kwargs):
    if False:
        print('Hello World!')
    'Get the possible dependencies of a number of packages.\n\n    See ``PackageBase.possible_dependencies`` for details.\n    '
    packages = []
    for pos in pkg_or_spec:
        if isinstance(pos, PackageMeta):
            packages.append(pos)
            continue
        if not isinstance(pos, spack.spec.Spec):
            pos = spack.spec.Spec(pos)
        if spack.repo.PATH.is_virtual(pos.name):
            packages.extend((p.package_class for p in spack.repo.PATH.providers_for(pos.name)))
            continue
        else:
            packages.append(pos.package_class)
    visited = {}
    for pkg in packages:
        pkg.possible_dependencies(visited=visited, **kwargs)
    return visited

class PackageStillNeededError(InstallError):
    """Raised when package is still needed by another on uninstall."""

    def __init__(self, spec, dependents):
        if False:
            i = 10
            return i + 15
        super().__init__('Cannot uninstall %s' % spec)
        self.spec = spec
        self.dependents = dependents

class PackageError(spack.error.SpackError):
    """Raised when something is wrong with a package definition."""

    def __init__(self, message, long_msg=None):
        if False:
            while True:
                i = 10
        super().__init__(message, long_msg)

class NoURLError(PackageError):
    """Raised when someone tries to build a URL for a package with no URLs."""

    def __init__(self, cls):
        if False:
            i = 10
            return i + 15
        super().__init__('Package %s has no version with a URL.' % cls.__name__)

class InvalidPackageOpError(PackageError):
    """Raised when someone tries perform an invalid operation on a package."""

class ExtensionError(PackageError):
    """Superclass for all errors having to do with extension packages."""

class ActivationError(ExtensionError):
    """Raised when there are problems activating an extension."""

    def __init__(self, msg, long_msg=None):
        if False:
            return 10
        super().__init__(msg, long_msg)

class DependencyConflictError(spack.error.SpackError):
    """Raised when the dependencies cannot be flattened as asked for."""

    def __init__(self, conflict):
        if False:
            i = 10
            return i + 15
        super().__init__('%s conflicts with another file in the flattened directory.' % conflict)