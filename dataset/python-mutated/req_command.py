"""Contains the Command base classes that depend on PipSession.

The classes in this module are in a separate module so the commands not
needing download / PackageFinder capability don't unnecessarily import the
PackageFinder machinery and all its vendored dependencies, etc.
"""
import logging
import os
import sys
from functools import partial
from optparse import Values
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from pip._internal.cache import WheelCache
from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.command_context import CommandContextMixIn
from pip._internal.exceptions import CommandError, PreviousBuildDirError
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.models.target_python import TargetPython
from pip._internal.network.session import PipSession
from pip._internal.operations.build.build_tracker import BuildTracker
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.constructors import install_req_from_editable, install_req_from_line, install_req_from_parsed_requirement, install_req_from_req_string
from pip._internal.req.req_file import parse_requirements
from pip._internal.req.req_install import InstallRequirement
from pip._internal.resolution.base import BaseResolver
from pip._internal.self_outdated_check import pip_self_version_check
from pip._internal.utils.temp_dir import TempDirectory, TempDirectoryTypeRegistry, tempdir_kinds
from pip._internal.utils.virtualenv import running_under_virtualenv
if TYPE_CHECKING:
    from ssl import SSLContext
logger = logging.getLogger(__name__)

def _create_truststore_ssl_context() -> Optional['SSLContext']:
    if False:
        for i in range(10):
            print('nop')
    if sys.version_info < (3, 10):
        raise CommandError('The truststore feature is only available for Python 3.10+')
    try:
        import ssl
    except ImportError:
        logger.warning('Disabling truststore since ssl support is missing')
        return None
    try:
        from pip._vendor import truststore
    except ImportError as e:
        raise CommandError(f'The truststore feature is unavailable: {e}')
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

class SessionCommandMixin(CommandContextMixIn):
    """
    A class mixin for command classes needing _build_session().
    """

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        self._session: Optional[PipSession] = None

    @classmethod
    def _get_index_urls(cls, options: Values) -> Optional[List[str]]:
        if False:
            for i in range(10):
                print('nop')
        'Return a list of index urls from user-provided options.'
        index_urls = []
        if not getattr(options, 'no_index', False):
            url = getattr(options, 'index_url', None)
            if url:
                index_urls.append(url)
        urls = getattr(options, 'extra_index_urls', None)
        if urls:
            index_urls.extend(urls)
        return index_urls or None

    def get_default_session(self, options: Values) -> PipSession:
        if False:
            return 10
        'Get a default-managed session.'
        if self._session is None:
            self._session = self.enter_context(self._build_session(options))
            assert self._session is not None
        return self._session

    def _build_session(self, options: Values, retries: Optional[int]=None, timeout: Optional[int]=None, fallback_to_certifi: bool=False) -> PipSession:
        if False:
            for i in range(10):
                print('nop')
        cache_dir = options.cache_dir
        assert not cache_dir or os.path.isabs(cache_dir)
        if 'truststore' in options.features_enabled:
            try:
                ssl_context = _create_truststore_ssl_context()
            except Exception:
                if not fallback_to_certifi:
                    raise
                ssl_context = None
        else:
            ssl_context = None
        session = PipSession(cache=os.path.join(cache_dir, 'http-v2') if cache_dir else None, retries=retries if retries is not None else options.retries, trusted_hosts=options.trusted_hosts, index_urls=self._get_index_urls(options), ssl_context=ssl_context)
        if options.cert:
            session.verify = options.cert
        if options.client_cert:
            session.cert = options.client_cert
        if options.timeout or timeout:
            session.timeout = timeout if timeout is not None else options.timeout
        if options.proxy:
            session.proxies = {'http': options.proxy, 'https': options.proxy}
        session.auth.prompting = not options.no_input
        session.auth.keyring_provider = options.keyring_provider
        return session

class IndexGroupCommand(Command, SessionCommandMixin):
    """
    Abstract base class for commands with the index_group options.

    This also corresponds to the commands that permit the pip version check.
    """

    def handle_pip_version_check(self, options: Values) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Do the pip version check if not disabled.\n\n        This overrides the default behavior of not doing the check.\n        '
        assert hasattr(options, 'no_index')
        if options.disable_pip_version_check or options.no_index:
            return
        session = self._build_session(options, retries=0, timeout=min(5, options.timeout), fallback_to_certifi=True)
        with session:
            pip_self_version_check(session, options)
KEEPABLE_TEMPDIR_TYPES = [tempdir_kinds.BUILD_ENV, tempdir_kinds.EPHEM_WHEEL_CACHE, tempdir_kinds.REQ_BUILD]

def warn_if_run_as_root() -> None:
    if False:
        return 10
    'Output a warning for sudo users on Unix.\n\n    In a virtual environment, sudo pip still writes to virtualenv.\n    On Windows, users may run pip as Administrator without issues.\n    This warning only applies to Unix root users outside of virtualenv.\n    '
    if running_under_virtualenv():
        return
    if not hasattr(os, 'getuid'):
        return
    if sys.platform == 'win32' or sys.platform == 'cygwin':
        return
    if os.getuid() != 0:
        return
    logger.warning("Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv")

def with_cleanup(func: Any) -> Any:
    if False:
        print('Hello World!')
    'Decorator for common logic related to managing temporary\n    directories.\n    '

    def configure_tempdir_registry(registry: TempDirectoryTypeRegistry) -> None:
        if False:
            return 10
        for t in KEEPABLE_TEMPDIR_TYPES:
            registry.set_delete(t, False)

    def wrapper(self: RequirementCommand, options: Values, args: List[Any]) -> Optional[int]:
        if False:
            while True:
                i = 10
        assert self.tempdir_registry is not None
        if options.no_clean:
            configure_tempdir_registry(self.tempdir_registry)
        try:
            return func(self, options, args)
        except PreviousBuildDirError:
            configure_tempdir_registry(self.tempdir_registry)
            raise
    return wrapper

class RequirementCommand(IndexGroupCommand):

    def __init__(self, *args: Any, **kw: Any) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kw)
        self.cmd_opts.add_option(cmdoptions.no_clean())

    @staticmethod
    def determine_resolver_variant(options: Values) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Determines which resolver should be used, based on the given options.'
        if 'legacy-resolver' in options.deprecated_features_enabled:
            return 'legacy'
        return 'resolvelib'

    @classmethod
    def make_requirement_preparer(cls, temp_build_dir: TempDirectory, options: Values, build_tracker: BuildTracker, session: PipSession, finder: PackageFinder, use_user_site: bool, download_dir: Optional[str]=None, verbosity: int=0) -> RequirementPreparer:
        if False:
            i = 10
            return i + 15
        '\n        Create a RequirementPreparer instance for the given parameters.\n        '
        temp_build_dir_path = temp_build_dir.path
        assert temp_build_dir_path is not None
        legacy_resolver = False
        resolver_variant = cls.determine_resolver_variant(options)
        if resolver_variant == 'resolvelib':
            lazy_wheel = 'fast-deps' in options.features_enabled
            if lazy_wheel:
                logger.warning('pip is using lazily downloaded wheels using HTTP range requests to obtain dependency information. This experimental feature is enabled through --use-feature=fast-deps and it is not ready for production.')
        else:
            legacy_resolver = True
            lazy_wheel = False
            if 'fast-deps' in options.features_enabled:
                logger.warning('fast-deps has no effect when used with the legacy resolver.')
        return RequirementPreparer(build_dir=temp_build_dir_path, src_dir=options.src_dir, download_dir=download_dir, build_isolation=options.build_isolation, check_build_deps=options.check_build_deps, build_tracker=build_tracker, session=session, progress_bar=options.progress_bar, finder=finder, require_hashes=options.require_hashes, use_user_site=use_user_site, lazy_wheel=lazy_wheel, verbosity=verbosity, legacy_resolver=legacy_resolver)

    @classmethod
    def make_resolver(cls, preparer: RequirementPreparer, finder: PackageFinder, options: Values, wheel_cache: Optional[WheelCache]=None, use_user_site: bool=False, ignore_installed: bool=True, ignore_requires_python: bool=False, force_reinstall: bool=False, upgrade_strategy: str='to-satisfy-only', use_pep517: Optional[bool]=None, py_version_info: Optional[Tuple[int, ...]]=None) -> BaseResolver:
        if False:
            i = 10
            return i + 15
        '\n        Create a Resolver instance for the given parameters.\n        '
        make_install_req = partial(install_req_from_req_string, isolated=options.isolated_mode, use_pep517=use_pep517)
        resolver_variant = cls.determine_resolver_variant(options)
        if resolver_variant == 'resolvelib':
            import pip._internal.resolution.resolvelib.resolver
            return pip._internal.resolution.resolvelib.resolver.Resolver(preparer=preparer, finder=finder, wheel_cache=wheel_cache, make_install_req=make_install_req, use_user_site=use_user_site, ignore_dependencies=options.ignore_dependencies, ignore_installed=ignore_installed, ignore_requires_python=ignore_requires_python, force_reinstall=force_reinstall, upgrade_strategy=upgrade_strategy, py_version_info=py_version_info)
        import pip._internal.resolution.legacy.resolver
        return pip._internal.resolution.legacy.resolver.Resolver(preparer=preparer, finder=finder, wheel_cache=wheel_cache, make_install_req=make_install_req, use_user_site=use_user_site, ignore_dependencies=options.ignore_dependencies, ignore_installed=ignore_installed, ignore_requires_python=ignore_requires_python, force_reinstall=force_reinstall, upgrade_strategy=upgrade_strategy, py_version_info=py_version_info)

    def get_requirements(self, args: List[str], options: Values, finder: PackageFinder, session: PipSession) -> List[InstallRequirement]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Parse command-line arguments into the corresponding requirements.\n        '
        requirements: List[InstallRequirement] = []
        for filename in options.constraints:
            for parsed_req in parse_requirements(filename, constraint=True, finder=finder, options=options, session=session):
                req_to_add = install_req_from_parsed_requirement(parsed_req, isolated=options.isolated_mode, user_supplied=False)
                requirements.append(req_to_add)
        for req in args:
            req_to_add = install_req_from_line(req, comes_from=None, isolated=options.isolated_mode, use_pep517=options.use_pep517, user_supplied=True, config_settings=getattr(options, 'config_settings', None))
            requirements.append(req_to_add)
        for req in options.editables:
            req_to_add = install_req_from_editable(req, user_supplied=True, isolated=options.isolated_mode, use_pep517=options.use_pep517, config_settings=getattr(options, 'config_settings', None))
            requirements.append(req_to_add)
        for filename in options.requirements:
            for parsed_req in parse_requirements(filename, finder=finder, options=options, session=session):
                req_to_add = install_req_from_parsed_requirement(parsed_req, isolated=options.isolated_mode, use_pep517=options.use_pep517, user_supplied=True, config_settings=parsed_req.options.get('config_settings') if parsed_req.options else None)
                requirements.append(req_to_add)
        if any((req.has_hash_options for req in requirements)):
            options.require_hashes = True
        if not (args or options.editables or options.requirements):
            opts = {'name': self.name}
            if options.find_links:
                raise CommandError('You must give at least one requirement to {name} (maybe you meant "pip {name} {links}"?)'.format(**dict(opts, links=' '.join(options.find_links))))
            else:
                raise CommandError('You must give at least one requirement to {name} (see "pip help {name}")'.format(**opts))
        return requirements

    @staticmethod
    def trace_basic_info(finder: PackageFinder) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Trace basic information about the provided objects.\n        '
        search_scope = finder.search_scope
        locations = search_scope.get_formatted_locations()
        if locations:
            logger.info(locations)

    def _build_package_finder(self, options: Values, session: PipSession, target_python: Optional[TargetPython]=None, ignore_requires_python: Optional[bool]=None) -> PackageFinder:
        if False:
            return 10
        '\n        Create a package finder appropriate to this requirement command.\n\n        :param ignore_requires_python: Whether to ignore incompatible\n            "Requires-Python" values in links. Defaults to False.\n        '
        link_collector = LinkCollector.create(session, options=options)
        selection_prefs = SelectionPreferences(allow_yanked=True, format_control=options.format_control, allow_all_prereleases=options.pre, prefer_binary=options.prefer_binary, ignore_requires_python=ignore_requires_python)
        return PackageFinder.create(link_collector=link_collector, selection_prefs=selection_prefs, target_python=target_python)