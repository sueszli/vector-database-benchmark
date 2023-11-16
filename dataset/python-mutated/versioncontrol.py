"""Handles all VCS (version control) support"""
import logging
import os
import shutil
import sys
import urllib.parse
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Type, Union
from pip._internal.cli.spinners import SpinnerInterface
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import HiddenText, ask_path_exists, backup_dir, display_path, hide_url, hide_value, is_installable_dir, rmtree
from pip._internal.utils.subprocess import CommandArgs, call_subprocess, format_command_args, make_command
from pip._internal.utils.urls import get_url_scheme
if TYPE_CHECKING:
    from typing import Literal
__all__ = ['vcs']
logger = logging.getLogger(__name__)
AuthInfo = Tuple[Optional[str], Optional[str]]

def is_url(name: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Return true if the name looks like a URL.\n    '
    scheme = get_url_scheme(name)
    if scheme is None:
        return False
    return scheme in ['http', 'https', 'file', 'ftp'] + vcs.all_schemes

def make_vcs_requirement_url(repo_url: str, rev: str, project_name: str, subdir: Optional[str]=None) -> str:
    if False:
        while True:
            i = 10
    '\n    Return the URL for a VCS requirement.\n\n    Args:\n      repo_url: the remote VCS url, with any needed VCS prefix (e.g. "git+").\n      project_name: the (unescaped) project name.\n    '
    egg_project_name = project_name.replace('-', '_')
    req = f'{repo_url}@{rev}#egg={egg_project_name}'
    if subdir:
        req += f'&subdirectory={subdir}'
    return req

def find_path_to_project_root_from_repo_root(location: str, repo_root: str) -> Optional[str]:
    if False:
        while True:
            i = 10
    "\n    Find the the Python project's root by searching up the filesystem from\n    `location`. Return the path to project root relative to `repo_root`.\n    Return None if the project root is `repo_root`, or cannot be found.\n    "
    orig_location = location
    while not is_installable_dir(location):
        last_location = location
        location = os.path.dirname(location)
        if location == last_location:
            logger.warning('Could not find a Python project for directory %s (tried all parent directories)', orig_location)
            return None
    if os.path.samefile(repo_root, location):
        return None
    return os.path.relpath(location, repo_root)

class RemoteNotFoundError(Exception):
    pass

class RemoteNotValidError(Exception):

    def __init__(self, url: str):
        if False:
            return 10
        super().__init__(url)
        self.url = url

class RevOptions:
    """
    Encapsulates a VCS-specific revision to install, along with any VCS
    install options.

    Instances of this class should be treated as if immutable.
    """

    def __init__(self, vc_class: Type['VersionControl'], rev: Optional[str]=None, extra_args: Optional[CommandArgs]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Args:\n          vc_class: a VersionControl subclass.\n          rev: the name of the revision to install.\n          extra_args: a list of extra options.\n        '
        if extra_args is None:
            extra_args = []
        self.extra_args = extra_args
        self.rev = rev
        self.vc_class = vc_class
        self.branch_name: Optional[str] = None

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'<RevOptions {self.vc_class.name}: rev={self.rev!r}>'

    @property
    def arg_rev(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        if self.rev is None:
            return self.vc_class.default_arg_rev
        return self.rev

    def to_args(self) -> CommandArgs:
        if False:
            print('Hello World!')
        '\n        Return the VCS-specific command arguments.\n        '
        args: CommandArgs = []
        rev = self.arg_rev
        if rev is not None:
            args += self.vc_class.get_base_rev_args(rev)
        args += self.extra_args
        return args

    def to_display(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        if not self.rev:
            return ''
        return f' (to revision {self.rev})'

    def make_new(self, rev: str) -> 'RevOptions':
        if False:
            while True:
                i = 10
        '\n        Make a copy of the current instance, but with a new rev.\n\n        Args:\n          rev: the name of the revision for the new object.\n        '
        return self.vc_class.make_rev_options(rev, extra_args=self.extra_args)

class VcsSupport:
    _registry: Dict[str, 'VersionControl'] = {}
    schemes = ['ssh', 'git', 'hg', 'bzr', 'sftp', 'svn']

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        urllib.parse.uses_netloc.extend(self.schemes)
        super().__init__()

    def __iter__(self) -> Iterator[str]:
        if False:
            print('Hello World!')
        return self._registry.__iter__()

    @property
    def backends(self) -> List['VersionControl']:
        if False:
            return 10
        return list(self._registry.values())

    @property
    def dirnames(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        return [backend.dirname for backend in self.backends]

    @property
    def all_schemes(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        schemes: List[str] = []
        for backend in self.backends:
            schemes.extend(backend.schemes)
        return schemes

    def register(self, cls: Type['VersionControl']) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(cls, 'name'):
            logger.warning('Cannot register VCS %s', cls.__name__)
            return
        if cls.name not in self._registry:
            self._registry[cls.name] = cls()
            logger.debug('Registered VCS backend: %s', cls.name)

    def unregister(self, name: str) -> None:
        if False:
            print('Hello World!')
        if name in self._registry:
            del self._registry[name]

    def get_backend_for_dir(self, location: str) -> Optional['VersionControl']:
        if False:
            i = 10
            return i + 15
        '\n        Return a VersionControl object if a repository of that type is found\n        at the given directory.\n        '
        vcs_backends = {}
        for vcs_backend in self._registry.values():
            repo_path = vcs_backend.get_repository_root(location)
            if not repo_path:
                continue
            logger.debug('Determine that %s uses VCS: %s', location, vcs_backend.name)
            vcs_backends[repo_path] = vcs_backend
        if not vcs_backends:
            return None
        inner_most_repo_path = max(vcs_backends, key=len)
        return vcs_backends[inner_most_repo_path]

    def get_backend_for_scheme(self, scheme: str) -> Optional['VersionControl']:
        if False:
            return 10
        '\n        Return a VersionControl object or None.\n        '
        for vcs_backend in self._registry.values():
            if scheme in vcs_backend.schemes:
                return vcs_backend
        return None

    def get_backend(self, name: str) -> Optional['VersionControl']:
        if False:
            print('Hello World!')
        '\n        Return a VersionControl object or None.\n        '
        name = name.lower()
        return self._registry.get(name)
vcs = VcsSupport()

class VersionControl:
    name = ''
    dirname = ''
    repo_name = ''
    schemes: Tuple[str, ...] = ()
    unset_environ: Tuple[str, ...] = ()
    default_arg_rev: Optional[str] = None

    @classmethod
    def should_add_vcs_url_prefix(cls, remote_url: str) -> bool:
        if False:
            return 10
        '\n        Return whether the vcs prefix (e.g. "git+") should be added to a\n        repository\'s remote url when used in a requirement.\n        '
        return not remote_url.lower().startswith(f'{cls.name}:')

    @classmethod
    def get_subdirectory(cls, location: str) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the path to Python project root, relative to the repo root.\n        Return None if the project root is in the repo root.\n        '
        return None

    @classmethod
    def get_requirement_revision(cls, repo_dir: str) -> str:
        if False:
            print('Hello World!')
        '\n        Return the revision string that should be used in a requirement.\n        '
        return cls.get_revision(repo_dir)

    @classmethod
    def get_src_requirement(cls, repo_dir: str, project_name: str) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Return the requirement string to use to redownload the files\n        currently at the given repository directory.\n\n        Args:\n          project_name: the (unescaped) project name.\n\n        The return value has a form similar to the following:\n\n            {repository_url}@{revision}#egg={project_name}\n        '
        repo_url = cls.get_remote_url(repo_dir)
        if cls.should_add_vcs_url_prefix(repo_url):
            repo_url = f'{cls.name}+{repo_url}'
        revision = cls.get_requirement_revision(repo_dir)
        subdir = cls.get_subdirectory(repo_dir)
        req = make_vcs_requirement_url(repo_url, revision, project_name, subdir=subdir)
        return req

    @staticmethod
    def get_base_rev_args(rev: str) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        Return the base revision arguments for a vcs command.\n\n        Args:\n          rev: the name of a revision to install.  Cannot be None.\n        '
        raise NotImplementedError

    def is_immutable_rev_checkout(self, url: str, dest: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return true if the commit hash checked out at dest matches\n        the revision in url.\n\n        Always return False, if the VCS does not support immutable commit\n        hashes.\n\n        This method does not check if there are local uncommitted changes\n        in dest after checkout, as pip currently has no use case for that.\n        '
        return False

    @classmethod
    def make_rev_options(cls, rev: Optional[str]=None, extra_args: Optional[CommandArgs]=None) -> RevOptions:
        if False:
            i = 10
            return i + 15
        '\n        Return a RevOptions object.\n\n        Args:\n          rev: the name of a revision to install.\n          extra_args: a list of extra options.\n        '
        return RevOptions(cls, rev, extra_args=extra_args)

    @classmethod
    def _is_local_repository(cls, repo: str) -> bool:
        if False:
            print('Hello World!')
        '\n        posix absolute paths start with os.path.sep,\n        win32 ones start with drive (like c:\\folder)\n        '
        (drive, tail) = os.path.splitdrive(repo)
        return repo.startswith(os.path.sep) or bool(drive)

    @classmethod
    def get_netloc_and_auth(cls, netloc: str, scheme: str) -> Tuple[str, Tuple[Optional[str], Optional[str]]]:
        if False:
            return 10
        "\n        Parse the repository URL's netloc, and return the new netloc to use\n        along with auth information.\n\n        Args:\n          netloc: the original repository URL netloc.\n          scheme: the repository URL's scheme without the vcs prefix.\n\n        This is mainly for the Subversion class to override, so that auth\n        information can be provided via the --username and --password options\n        instead of through the URL.  For other subclasses like Git without\n        such an option, auth information must stay in the URL.\n\n        Returns: (netloc, (username, password)).\n        "
        return (netloc, (None, None))

    @classmethod
    def get_url_rev_and_auth(cls, url: str) -> Tuple[str, Optional[str], AuthInfo]:
        if False:
            i = 10
            return i + 15
        '\n        Parse the repository URL to use, and return the URL, revision,\n        and auth info to use.\n\n        Returns: (url, rev, (username, password)).\n        '
        (scheme, netloc, path, query, frag) = urllib.parse.urlsplit(url)
        if '+' not in scheme:
            raise ValueError(f'Sorry, {url!r} is a malformed VCS url. The format is <vcs>+<protocol>://<url>, e.g. svn+http://myrepo/svn/MyApp#egg=MyApp')
        scheme = scheme.split('+', 1)[1]
        (netloc, user_pass) = cls.get_netloc_and_auth(netloc, scheme)
        rev = None
        if '@' in path:
            (path, rev) = path.rsplit('@', 1)
            if not rev:
                raise InstallationError(f'The URL {url!r} has an empty revision (after @) which is not supported. Include a revision after @ or remove @ from the URL.')
        url = urllib.parse.urlunsplit((scheme, netloc, path, query, ''))
        return (url, rev, user_pass)

    @staticmethod
    def make_rev_args(username: Optional[str], password: Optional[HiddenText]) -> CommandArgs:
        if False:
            while True:
                i = 10
        '\n        Return the RevOptions "extra arguments" to use in obtain().\n        '
        return []

    def get_url_rev_options(self, url: HiddenText) -> Tuple[HiddenText, RevOptions]:
        if False:
            print('Hello World!')
        '\n        Return the URL and RevOptions object to use in obtain(),\n        as a tuple (url, rev_options).\n        '
        (secret_url, rev, user_pass) = self.get_url_rev_and_auth(url.secret)
        (username, secret_password) = user_pass
        password: Optional[HiddenText] = None
        if secret_password is not None:
            password = hide_value(secret_password)
        extra_args = self.make_rev_args(username, password)
        rev_options = self.make_rev_options(rev, extra_args=extra_args)
        return (hide_url(secret_url), rev_options)

    @staticmethod
    def normalize_url(url: str) -> str:
        if False:
            return 10
        '\n        Normalize a URL for comparison by unquoting it and removing any\n        trailing slash.\n        '
        return urllib.parse.unquote(url).rstrip('/')

    @classmethod
    def compare_urls(cls, url1: str, url2: str) -> bool:
        if False:
            print('Hello World!')
        '\n        Compare two repo URLs for identity, ignoring incidental differences.\n        '
        return cls.normalize_url(url1) == cls.normalize_url(url2)

    def fetch_new(self, dest: str, url: HiddenText, rev_options: RevOptions, verbosity: int) -> None:
        if False:
            return 10
        '\n        Fetch a revision from a repository, in the case that this is the\n        first fetch from the repository.\n\n        Args:\n          dest: the directory to fetch the repository to.\n          rev_options: a RevOptions object.\n          verbosity: verbosity level.\n        '
        raise NotImplementedError

    def switch(self, dest: str, url: HiddenText, rev_options: RevOptions) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Switch the repo at ``dest`` to point to ``URL``.\n\n        Args:\n          rev_options: a RevOptions object.\n        '
        raise NotImplementedError

    def update(self, dest: str, url: HiddenText, rev_options: RevOptions) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update an already-existing repo to the given ``rev_options``.\n\n        Args:\n          rev_options: a RevOptions object.\n        '
        raise NotImplementedError

    @classmethod
    def is_commit_id_equal(cls, dest: str, name: Optional[str]) -> bool:
        if False:
            print('Hello World!')
        '\n        Return whether the id of the current commit equals the given name.\n\n        Args:\n          dest: the repository directory.\n          name: a string name.\n        '
        raise NotImplementedError

    def obtain(self, dest: str, url: HiddenText, verbosity: int) -> None:
        if False:
            print('Hello World!')
        '\n        Install or update in editable mode the package represented by this\n        VersionControl object.\n\n        :param dest: the repository directory in which to install or update.\n        :param url: the repository URL starting with a vcs prefix.\n        :param verbosity: verbosity level.\n        '
        (url, rev_options) = self.get_url_rev_options(url)
        if not os.path.exists(dest):
            self.fetch_new(dest, url, rev_options, verbosity=verbosity)
            return
        rev_display = rev_options.to_display()
        if self.is_repository_directory(dest):
            existing_url = self.get_remote_url(dest)
            if self.compare_urls(existing_url, url.secret):
                logger.debug('%s in %s exists, and has correct URL (%s)', self.repo_name.title(), display_path(dest), url)
                if not self.is_commit_id_equal(dest, rev_options.rev):
                    logger.info('Updating %s %s%s', display_path(dest), self.repo_name, rev_display)
                    self.update(dest, url, rev_options)
                else:
                    logger.info('Skipping because already up-to-date.')
                return
            logger.warning('%s %s in %s exists with URL %s', self.name, self.repo_name, display_path(dest), existing_url)
            prompt = ('(s)witch, (i)gnore, (w)ipe, (b)ackup ', ('s', 'i', 'w', 'b'))
        else:
            logger.warning('Directory %s already exists, and is not a %s %s.', dest, self.name, self.repo_name)
            prompt = ('(i)gnore, (w)ipe, (b)ackup ', ('i', 'w', 'b'))
        logger.warning('The plan is to install the %s repository %s', self.name, url)
        response = ask_path_exists(f'What to do?  {prompt[0]}', prompt[1])
        if response == 'a':
            sys.exit(-1)
        if response == 'w':
            logger.warning('Deleting %s', display_path(dest))
            rmtree(dest)
            self.fetch_new(dest, url, rev_options, verbosity=verbosity)
            return
        if response == 'b':
            dest_dir = backup_dir(dest)
            logger.warning('Backing up %s to %s', display_path(dest), dest_dir)
            shutil.move(dest, dest_dir)
            self.fetch_new(dest, url, rev_options, verbosity=verbosity)
            return
        if response == 's':
            logger.info('Switching %s %s to %s%s', self.repo_name, display_path(dest), url, rev_display)
            self.switch(dest, url, rev_options)

    def unpack(self, location: str, url: HiddenText, verbosity: int) -> None:
        if False:
            while True:
                i = 10
        '\n        Clean up current location and download the url repository\n        (and vcs infos) into location\n\n        :param url: the repository URL starting with a vcs prefix.\n        :param verbosity: verbosity level.\n        '
        if os.path.exists(location):
            rmtree(location)
        self.obtain(location, url=url, verbosity=verbosity)

    @classmethod
    def get_remote_url(cls, location: str) -> str:
        if False:
            while True:
                i = 10
        '\n        Return the url used at location\n\n        Raises RemoteNotFoundError if the repository does not have a remote\n        url configured.\n        '
        raise NotImplementedError

    @classmethod
    def get_revision(cls, location: str) -> str:
        if False:
            return 10
        '\n        Return the current commit id of the files at the given location.\n        '
        raise NotImplementedError

    @classmethod
    def run_command(cls, cmd: Union[List[str], CommandArgs], show_stdout: bool=True, cwd: Optional[str]=None, on_returncode: 'Literal["raise", "warn", "ignore"]'='raise', extra_ok_returncodes: Optional[Iterable[int]]=None, command_desc: Optional[str]=None, extra_environ: Optional[Mapping[str, Any]]=None, spinner: Optional[SpinnerInterface]=None, log_failed_cmd: bool=True, stdout_only: bool=False) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Run a VCS subcommand\n        This is simply a wrapper around call_subprocess that adds the VCS\n        command name, and checks that the VCS is available\n        '
        cmd = make_command(cls.name, *cmd)
        if command_desc is None:
            command_desc = format_command_args(cmd)
        try:
            return call_subprocess(cmd, show_stdout, cwd, on_returncode=on_returncode, extra_ok_returncodes=extra_ok_returncodes, command_desc=command_desc, extra_environ=extra_environ, unset_environ=cls.unset_environ, spinner=spinner, log_failed_cmd=log_failed_cmd, stdout_only=stdout_only)
        except FileNotFoundError:
            raise BadCommand(f'Cannot find command {cls.name!r} - do you have {cls.name!r} installed and in your PATH?')
        except PermissionError:
            raise BadCommand(f'No permission to execute {cls.name!r} - install it locally, globally (ask admin), or check your PATH. See possible solutions at https://pip.pypa.io/en/latest/reference/pip_freeze/#fixing-permission-denied.')

    @classmethod
    def is_repository_directory(cls, path: str) -> bool:
        if False:
            print('Hello World!')
        '\n        Return whether a directory path is a repository directory.\n        '
        logger.debug('Checking in %s for %s (%s)...', path, cls.dirname, cls.name)
        return os.path.exists(os.path.join(path, cls.dirname))

    @classmethod
    def get_repository_root(cls, location: str) -> Optional[str]:
        if False:
            return 10
        '\n        Return the "root" (top-level) directory controlled by the vcs,\n        or `None` if the directory is not in any.\n\n        It is meant to be overridden to implement smarter detection\n        mechanisms for specific vcs.\n\n        This can do more than is_repository_directory() alone. For\n        example, the Git override checks that Git is actually available.\n        '
        if cls.is_repository_directory(location):
            return location
        return None