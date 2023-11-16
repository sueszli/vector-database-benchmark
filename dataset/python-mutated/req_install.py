import functools
import logging
import os
import shutil
import sys
import uuid
import zipfile
from optparse import Values
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Union
from pip._vendor.packaging.markers import Marker
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import Version
from pip._vendor.packaging.version import parse as parse_version
from pip._vendor.pyproject_hooks import BuildBackendHookCaller
from pip._internal.build_env import BuildEnvironment, NoOpBuildEnvironment
from pip._internal.exceptions import InstallationError, PreviousBuildDirError
from pip._internal.locations import get_scheme
from pip._internal.metadata import BaseDistribution, get_default_environment, get_directory_distribution, get_wheel_distribution
from pip._internal.metadata.base import FilesystemWheel
from pip._internal.models.direct_url import DirectUrl
from pip._internal.models.link import Link
from pip._internal.operations.build.metadata import generate_metadata
from pip._internal.operations.build.metadata_editable import generate_editable_metadata
from pip._internal.operations.build.metadata_legacy import generate_metadata as generate_metadata_legacy
from pip._internal.operations.install.editable_legacy import install_editable as install_editable_legacy
from pip._internal.operations.install.wheel import install_wheel
from pip._internal.pyproject import load_pyproject_toml, make_pyproject_path
from pip._internal.req.req_uninstall import UninstallPathSet
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.misc import ConfiguredBuildBackendHookCaller, ask_path_exists, backup_dir, display_path, hide_url, is_installable_dir, redact_auth_from_requirement, redact_auth_from_url
from pip._internal.utils.packaging import safe_extra
from pip._internal.utils.subprocess import runner_with_spinner_message
from pip._internal.utils.temp_dir import TempDirectory, tempdir_kinds
from pip._internal.utils.unpacking import unpack_file
from pip._internal.utils.virtualenv import running_under_virtualenv
from pip._internal.vcs import vcs
logger = logging.getLogger(__name__)

class InstallRequirement:
    """
    Represents something that may be installed later on, may have information
    about where to fetch the relevant requirement and also contains logic for
    installing the said requirement.
    """

    def __init__(self, req: Optional[Requirement], comes_from: Optional[Union[str, 'InstallRequirement']], editable: bool=False, link: Optional[Link]=None, markers: Optional[Marker]=None, use_pep517: Optional[bool]=None, isolated: bool=False, *, global_options: Optional[List[str]]=None, hash_options: Optional[Dict[str, List[str]]]=None, config_settings: Optional[Dict[str, Union[str, List[str]]]]=None, constraint: bool=False, extras: Collection[str]=(), user_supplied: bool=False, permit_editable_wheels: bool=False) -> None:
        if False:
            return 10
        assert req is None or isinstance(req, Requirement), req
        self.req = req
        self.comes_from = comes_from
        self.constraint = constraint
        self.editable = editable
        self.permit_editable_wheels = permit_editable_wheels
        self.source_dir: Optional[str] = None
        if self.editable:
            assert link
            if link.is_file:
                self.source_dir = os.path.normpath(os.path.abspath(link.file_path))
        if link is None and req and req.url:
            link = Link(req.url)
        self.link = self.original_link = link
        self.cached_wheel_source_link: Optional[Link] = None
        self.download_info: Optional[DirectUrl] = None
        self.local_file_path: Optional[str] = None
        if self.link and self.link.is_file:
            self.local_file_path = self.link.file_path
        if extras:
            self.extras = extras
        elif req:
            self.extras = req.extras
        else:
            self.extras = set()
        if markers is None and req:
            markers = req.marker
        self.markers = markers
        self.satisfied_by: Optional[BaseDistribution] = None
        self.should_reinstall = False
        self._temp_build_dir: Optional[TempDirectory] = None
        self.install_succeeded: Optional[bool] = None
        self.global_options = global_options if global_options else []
        self.hash_options = hash_options if hash_options else {}
        self.config_settings = config_settings
        self.prepared = False
        self.user_supplied = user_supplied
        self.isolated = isolated
        self.build_env: BuildEnvironment = NoOpBuildEnvironment()
        self.metadata_directory: Optional[str] = None
        self.pyproject_requires: Optional[List[str]] = None
        self.requirements_to_check: List[str] = []
        self.pep517_backend: Optional[BuildBackendHookCaller] = None
        self.use_pep517 = use_pep517
        self.needs_more_preparation = False
        self._archive_source: Optional[Path] = None

    def __str__(self) -> str:
        if False:
            return 10
        if self.req:
            s = redact_auth_from_requirement(self.req)
            if self.link:
                s += f' from {redact_auth_from_url(self.link.url)}'
        elif self.link:
            s = redact_auth_from_url(self.link.url)
        else:
            s = '<InstallRequirement>'
        if self.satisfied_by is not None:
            if self.satisfied_by.location is not None:
                location = display_path(self.satisfied_by.location)
            else:
                location = '<memory>'
            s += f' in {location}'
        if self.comes_from:
            if isinstance(self.comes_from, str):
                comes_from: Optional[str] = self.comes_from
            else:
                comes_from = self.comes_from.from_path()
            if comes_from:
                s += f' (from {comes_from})'
        return s

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return '<{} object: {} editable={!r}>'.format(self.__class__.__name__, str(self), self.editable)

    def format_debug(self) -> str:
        if False:
            i = 10
            return i + 15
        'An un-tested helper for getting state, for debugging.'
        attributes = vars(self)
        names = sorted(attributes)
        state = (f'{attr}={attributes[attr]!r}' for attr in sorted(names))
        return '<{name} object: {{{state}}}>'.format(name=self.__class__.__name__, state=', '.join(state))

    @property
    def name(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        if self.req is None:
            return None
        return self.req.name

    @functools.lru_cache()
    def supports_pyproject_editable(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not self.use_pep517:
            return False
        assert self.pep517_backend
        with self.build_env:
            runner = runner_with_spinner_message('Checking if build backend supports build_editable')
            with self.pep517_backend.subprocess_runner(runner):
                return 'build_editable' in self.pep517_backend._supported_features()

    @property
    def specifier(self) -> SpecifierSet:
        if False:
            while True:
                i = 10
        assert self.req is not None
        return self.req.specifier

    @property
    def is_direct(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Whether this requirement was specified as a direct URL.'
        return self.original_link is not None

    @property
    def is_pinned(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Return whether I am pinned to an exact version.\n\n        For example, some-package==1.2 is pinned; some-package>1.2 is not.\n        '
        assert self.req is not None
        specifiers = self.req.specifier
        return len(specifiers) == 1 and next(iter(specifiers)).operator in {'==', '==='}

    def match_markers(self, extras_requested: Optional[Iterable[str]]=None) -> bool:
        if False:
            while True:
                i = 10
        if not extras_requested:
            extras_requested = ('',)
        if self.markers is not None:
            return any((self.markers.evaluate({'extra': extra}) or self.markers.evaluate({'extra': safe_extra(extra)}) or self.markers.evaluate({'extra': canonicalize_name(extra)}) for extra in extras_requested))
        else:
            return True

    @property
    def has_hash_options(self) -> bool:
        if False:
            print('Hello World!')
        'Return whether any known-good hashes are specified as options.\n\n        These activate --require-hashes mode; hashes specified as part of a\n        URL do not.\n\n        '
        return bool(self.hash_options)

    def hashes(self, trust_internet: bool=True) -> Hashes:
        if False:
            return 10
        'Return a hash-comparer that considers my option- and URL-based\n        hashes to be known-good.\n\n        Hashes in URLs--ones embedded in the requirements file, not ones\n        downloaded from an index server--are almost peers with ones from\n        flags. They satisfy --require-hashes (whether it was implicitly or\n        explicitly activated) but do not activate it. md5 and sha224 are not\n        allowed in flags, which should nudge people toward good algos. We\n        always OR all hashes together, even ones from URLs.\n\n        :param trust_internet: Whether to trust URL-based (#md5=...) hashes\n            downloaded from the internet, as by populate_link()\n\n        '
        good_hashes = self.hash_options.copy()
        if trust_internet:
            link = self.link
        elif self.is_direct and self.user_supplied:
            link = self.original_link
        else:
            link = None
        if link and link.hash:
            assert link.hash_name is not None
            good_hashes.setdefault(link.hash_name, []).append(link.hash)
        return Hashes(good_hashes)

    def from_path(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'Format a nice indicator to show where this "comes from" '
        if self.req is None:
            return None
        s = str(self.req)
        if self.comes_from:
            comes_from: Optional[str]
            if isinstance(self.comes_from, str):
                comes_from = self.comes_from
            else:
                comes_from = self.comes_from.from_path()
            if comes_from:
                s += '->' + comes_from
        return s

    def ensure_build_location(self, build_dir: str, autodelete: bool, parallel_builds: bool) -> str:
        if False:
            while True:
                i = 10
        assert build_dir is not None
        if self._temp_build_dir is not None:
            assert self._temp_build_dir.path
            return self._temp_build_dir.path
        if self.req is None:
            self._temp_build_dir = TempDirectory(kind=tempdir_kinds.REQ_BUILD, globally_managed=True)
            return self._temp_build_dir.path
        dir_name: str = canonicalize_name(self.req.name)
        if parallel_builds:
            dir_name = f'{dir_name}_{uuid.uuid4().hex}'
        if not os.path.exists(build_dir):
            logger.debug('Creating directory %s', build_dir)
            os.makedirs(build_dir)
        actual_build_dir = os.path.join(build_dir, dir_name)
        delete_arg = None if autodelete else False
        return TempDirectory(path=actual_build_dir, delete=delete_arg, kind=tempdir_kinds.REQ_BUILD, globally_managed=True).path

    def _set_requirement(self) -> None:
        if False:
            i = 10
            return i + 15
        'Set requirement after generating metadata.'
        assert self.req is None
        assert self.metadata is not None
        assert self.source_dir is not None
        if isinstance(parse_version(self.metadata['Version']), Version):
            op = '=='
        else:
            op = '==='
        self.req = Requirement(''.join([self.metadata['Name'], op, self.metadata['Version']]))

    def warn_on_mismatching_name(self) -> None:
        if False:
            return 10
        assert self.req is not None
        metadata_name = canonicalize_name(self.metadata['Name'])
        if canonicalize_name(self.req.name) == metadata_name:
            return
        logger.warning('Generating metadata for package %s produced metadata for project name %s. Fix your #egg=%s fragments.', self.name, metadata_name, self.name)
        self.req = Requirement(metadata_name)

    def check_if_exists(self, use_user_site: bool) -> None:
        if False:
            return 10
        'Find an installed distribution that satisfies or conflicts\n        with this requirement, and set self.satisfied_by or\n        self.should_reinstall appropriately.\n        '
        if self.req is None:
            return
        existing_dist = get_default_environment().get_distribution(self.req.name)
        if not existing_dist:
            return
        version_compatible = self.req.specifier.contains(existing_dist.version, prereleases=True)
        if not version_compatible:
            self.satisfied_by = None
            if use_user_site:
                if existing_dist.in_usersite:
                    self.should_reinstall = True
                elif running_under_virtualenv() and existing_dist.in_site_packages:
                    raise InstallationError(f'Will not install to the user site because it will lack sys.path precedence to {existing_dist.raw_name} in {existing_dist.location}')
            else:
                self.should_reinstall = True
        elif self.editable:
            self.should_reinstall = True
            self.satisfied_by = None
        else:
            self.satisfied_by = existing_dist

    @property
    def is_wheel(self) -> bool:
        if False:
            return 10
        if not self.link:
            return False
        return self.link.is_wheel

    @property
    def is_wheel_from_cache(self) -> bool:
        if False:
            while True:
                i = 10
        return self.cached_wheel_source_link is not None

    @property
    def unpacked_source_directory(self) -> str:
        if False:
            print('Hello World!')
        assert self.source_dir, f'No source dir for {self}'
        return os.path.join(self.source_dir, self.link and self.link.subdirectory_fragment or '')

    @property
    def setup_py_path(self) -> str:
        if False:
            while True:
                i = 10
        assert self.source_dir, f'No source dir for {self}'
        setup_py = os.path.join(self.unpacked_source_directory, 'setup.py')
        return setup_py

    @property
    def setup_cfg_path(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        assert self.source_dir, f'No source dir for {self}'
        setup_cfg = os.path.join(self.unpacked_source_directory, 'setup.cfg')
        return setup_cfg

    @property
    def pyproject_toml_path(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        assert self.source_dir, f'No source dir for {self}'
        return make_pyproject_path(self.unpacked_source_directory)

    def load_pyproject_toml(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Load the pyproject.toml file.\n\n        After calling this routine, all of the attributes related to PEP 517\n        processing for this requirement have been set. In particular, the\n        use_pep517 attribute can be used to determine whether we should\n        follow the PEP 517 or legacy (setup.py) code path.\n        '
        pyproject_toml_data = load_pyproject_toml(self.use_pep517, self.pyproject_toml_path, self.setup_py_path, str(self))
        if pyproject_toml_data is None:
            if self.config_settings:
                deprecated(reason=f'Config settings are ignored for project {self}.', replacement='to use --use-pep517 or add a pyproject.toml file to the project', gone_in='24.0')
            self.use_pep517 = False
            return
        self.use_pep517 = True
        (requires, backend, check, backend_path) = pyproject_toml_data
        self.requirements_to_check = check
        self.pyproject_requires = requires
        self.pep517_backend = ConfiguredBuildBackendHookCaller(self, self.unpacked_source_directory, backend, backend_path=backend_path)

    def isolated_editable_sanity_check(self) -> None:
        if False:
            while True:
                i = 10
        'Check that an editable requirement if valid for use with PEP 517/518.\n\n        This verifies that an editable that has a pyproject.toml either supports PEP 660\n        or as a setup.py or a setup.cfg\n        '
        if self.editable and self.use_pep517 and (not self.supports_pyproject_editable()) and (not os.path.isfile(self.setup_py_path)) and (not os.path.isfile(self.setup_cfg_path)):
            raise InstallationError(f"Project {self} has a 'pyproject.toml' and its build backend is missing the 'build_editable' hook. Since it does not have a 'setup.py' nor a 'setup.cfg', it cannot be installed in editable mode. Consider using a build backend that supports PEP 660.")

    def prepare_metadata(self) -> None:
        if False:
            print('Hello World!')
        'Ensure that project metadata is available.\n\n        Under PEP 517 and PEP 660, call the backend hook to prepare the metadata.\n        Under legacy processing, call setup.py egg-info.\n        '
        assert self.source_dir, f'No source dir for {self}'
        details = self.name or f'from {self.link}'
        if self.use_pep517:
            assert self.pep517_backend is not None
            if self.editable and self.permit_editable_wheels and self.supports_pyproject_editable():
                self.metadata_directory = generate_editable_metadata(build_env=self.build_env, backend=self.pep517_backend, details=details)
            else:
                self.metadata_directory = generate_metadata(build_env=self.build_env, backend=self.pep517_backend, details=details)
        else:
            self.metadata_directory = generate_metadata_legacy(build_env=self.build_env, setup_py_path=self.setup_py_path, source_dir=self.unpacked_source_directory, isolated=self.isolated, details=details)
        if not self.name:
            self._set_requirement()
        else:
            self.warn_on_mismatching_name()
        self.assert_source_matches_version()

    @property
    def metadata(self) -> Any:
        if False:
            print('Hello World!')
        if not hasattr(self, '_metadata'):
            self._metadata = self.get_dist().metadata
        return self._metadata

    def get_dist(self) -> BaseDistribution:
        if False:
            while True:
                i = 10
        if self.metadata_directory:
            return get_directory_distribution(self.metadata_directory)
        elif self.local_file_path and self.is_wheel:
            assert self.req is not None
            return get_wheel_distribution(FilesystemWheel(self.local_file_path), canonicalize_name(self.req.name))
        raise AssertionError(f"InstallRequirement {self} has no metadata directory and no wheel: can't make a distribution.")

    def assert_source_matches_version(self) -> None:
        if False:
            return 10
        assert self.source_dir, f'No source dir for {self}'
        version = self.metadata['version']
        if self.req and self.req.specifier and (version not in self.req.specifier):
            logger.warning('Requested %s, but installing version %s', self, version)
        else:
            logger.debug('Source in %s has version %s, which satisfies requirement %s', display_path(self.source_dir), version, self)

    def ensure_has_source_dir(self, parent_dir: str, autodelete: bool=False, parallel_builds: bool=False) -> None:
        if False:
            return 10
        "Ensure that a source_dir is set.\n\n        This will create a temporary build dir if the name of the requirement\n        isn't known yet.\n\n        :param parent_dir: The ideal pip parent_dir for the source_dir.\n            Generally src_dir for editables and build_dir for sdists.\n        :return: self.source_dir\n        "
        if self.source_dir is None:
            self.source_dir = self.ensure_build_location(parent_dir, autodelete=autodelete, parallel_builds=parallel_builds)

    def needs_unpacked_archive(self, archive_source: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert self._archive_source is None
        self._archive_source = archive_source

    def ensure_pristine_source_checkout(self) -> None:
        if False:
            while True:
                i = 10
        'Ensure the source directory has not yet been built in.'
        assert self.source_dir is not None
        if self._archive_source is not None:
            unpack_file(str(self._archive_source), self.source_dir)
        elif is_installable_dir(self.source_dir):
            raise PreviousBuildDirError(f"pip can't proceed with requirements '{self}' due to a pre-existing build directory ({self.source_dir}). This is likely due to a previous installation that failed . pip is being responsible and not assuming it can delete this. Please delete it and try again.")

    def update_editable(self) -> None:
        if False:
            print('Hello World!')
        if not self.link:
            logger.debug('Cannot update repository at %s; repository location is unknown', self.source_dir)
            return
        assert self.editable
        assert self.source_dir
        if self.link.scheme == 'file':
            return
        vcs_backend = vcs.get_backend_for_scheme(self.link.scheme)
        assert vcs_backend, f'Unsupported VCS URL {self.link.url}'
        hidden_url = hide_url(self.link.url)
        vcs_backend.obtain(self.source_dir, url=hidden_url, verbosity=0)

    def uninstall(self, auto_confirm: bool=False, verbose: bool=False) -> Optional[UninstallPathSet]:
        if False:
            print('Hello World!')
        '\n        Uninstall the distribution currently satisfying this requirement.\n\n        Prompts before removing or modifying files unless\n        ``auto_confirm`` is True.\n\n        Refuses to delete or modify files outside of ``sys.prefix`` -\n        thus uninstallation within a virtual environment can only\n        modify that virtual environment, even if the virtualenv is\n        linked to global site-packages.\n\n        '
        assert self.req
        dist = get_default_environment().get_distribution(self.req.name)
        if not dist:
            logger.warning('Skipping %s as it is not installed.', self.name)
            return None
        logger.info('Found existing installation: %s', dist)
        uninstalled_pathset = UninstallPathSet.from_dist(dist)
        uninstalled_pathset.remove(auto_confirm, verbose)
        return uninstalled_pathset

    def _get_archive_name(self, path: str, parentdir: str, rootdir: str) -> str:
        if False:
            print('Hello World!')

        def _clean_zip_name(name: str, prefix: str) -> str:
            if False:
                while True:
                    i = 10
            assert name.startswith(prefix + os.path.sep), f"name {name!r} doesn't start with prefix {prefix!r}"
            name = name[len(prefix) + 1:]
            name = name.replace(os.path.sep, '/')
            return name
        assert self.req is not None
        path = os.path.join(parentdir, path)
        name = _clean_zip_name(path, rootdir)
        return self.req.name + '/' + name

    def archive(self, build_dir: Optional[str]) -> None:
        if False:
            return 10
        'Saves archive to provided build_dir.\n\n        Used for saving downloaded VCS requirements as part of `pip download`.\n        '
        assert self.source_dir
        if build_dir is None:
            return
        create_archive = True
        archive_name = '{}-{}.zip'.format(self.name, self.metadata['version'])
        archive_path = os.path.join(build_dir, archive_name)
        if os.path.exists(archive_path):
            response = ask_path_exists(f'The file {display_path(archive_path)} exists. (i)gnore, (w)ipe, (b)ackup, (a)bort ', ('i', 'w', 'b', 'a'))
            if response == 'i':
                create_archive = False
            elif response == 'w':
                logger.warning('Deleting %s', display_path(archive_path))
                os.remove(archive_path)
            elif response == 'b':
                dest_file = backup_dir(archive_path)
                logger.warning('Backing up %s to %s', display_path(archive_path), display_path(dest_file))
                shutil.move(archive_path, dest_file)
            elif response == 'a':
                sys.exit(-1)
        if not create_archive:
            return
        zip_output = zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
        with zip_output:
            dir = os.path.normcase(os.path.abspath(self.unpacked_source_directory))
            for (dirpath, dirnames, filenames) in os.walk(dir):
                for dirname in dirnames:
                    dir_arcname = self._get_archive_name(dirname, parentdir=dirpath, rootdir=dir)
                    zipdir = zipfile.ZipInfo(dir_arcname + '/')
                    zipdir.external_attr = 493 << 16
                    zip_output.writestr(zipdir, '')
                for filename in filenames:
                    file_arcname = self._get_archive_name(filename, parentdir=dirpath, rootdir=dir)
                    filename = os.path.join(dirpath, filename)
                    zip_output.write(filename, file_arcname)
        logger.info('Saved %s', display_path(archive_path))

    def install(self, global_options: Optional[Sequence[str]]=None, root: Optional[str]=None, home: Optional[str]=None, prefix: Optional[str]=None, warn_script_location: bool=True, use_user_site: bool=False, pycompile: bool=True) -> None:
        if False:
            print('Hello World!')
        assert self.req is not None
        scheme = get_scheme(self.req.name, user=use_user_site, home=home, root=root, isolated=self.isolated, prefix=prefix)
        if self.editable and (not self.is_wheel):
            install_editable_legacy(global_options=global_options if global_options is not None else [], prefix=prefix, home=home, use_user_site=use_user_site, name=self.req.name, setup_py_path=self.setup_py_path, isolated=self.isolated, build_env=self.build_env, unpacked_source_directory=self.unpacked_source_directory)
            self.install_succeeded = True
            return
        assert self.is_wheel
        assert self.local_file_path
        install_wheel(self.req.name, self.local_file_path, scheme=scheme, req_description=str(self.req), pycompile=pycompile, warn_script_location=warn_script_location, direct_url=self.download_info if self.is_direct else None, requested=self.user_supplied)
        self.install_succeeded = True

def check_invalid_constraint_type(req: InstallRequirement) -> str:
    if False:
        return 10
    problem = ''
    if not req.name:
        problem = 'Unnamed requirements are not allowed as constraints'
    elif req.editable:
        problem = 'Editable requirements are not allowed as constraints'
    elif req.extras:
        problem = 'Constraints cannot have extras'
    if problem:
        deprecated(reason='Constraints are only allowed to take the form of a package name and a version specifier. Other forms were originally permitted as an accident of the implementation, but were undocumented. The new implementation of the resolver no longer supports these forms.', replacement='replacing the constraint with a requirement', gone_in=None, issue=8210)
    return problem

def _has_option(options: Values, reqs: List[InstallRequirement], option: str) -> bool:
    if False:
        return 10
    if getattr(options, option, None):
        return True
    for req in reqs:
        if getattr(req, option, None):
            return True
    return False

def check_legacy_setup_py_options(options: Values, reqs: List[InstallRequirement]) -> None:
    if False:
        for i in range(10):
            print('nop')
    has_build_options = _has_option(options, reqs, 'build_options')
    has_global_options = _has_option(options, reqs, 'global_options')
    if has_build_options or has_global_options:
        deprecated(reason='--build-option and --global-option are deprecated.', issue=11859, replacement='to use --config-settings', gone_in='24.0')
        logger.warning('Implying --no-binary=:all: due to the presence of --build-option / --global-option. ')
        options.format_control.disallow_binaries()