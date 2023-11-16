"""
Fetch strategies are used to download source code into a staging area
in order to build it.  They need to define the following methods:

    * fetch()
        This should attempt to download/check out source from somewhere.
    * check()
        Apply a checksum to the downloaded source code, e.g. for an archive.
        May not do anything if the fetch method was safe to begin with.
    * expand()
        Expand (e.g., an archive) downloaded file to source, with the
        standard stage source path as the destination directory.
    * reset()
        Restore original state of downloaded code.  Used by clean commands.
        This may just remove the expanded source and re-expand an archive,
        or it may run something like git reset --hard.
    * archive()
        Archive a source directory, e.g. for creating a mirror.
"""
import copy
import functools
import os
import os.path
import re
import shutil
import urllib.error
import urllib.parse
from typing import List, Optional
import llnl.url
import llnl.util
import llnl.util.filesystem as fs
import llnl.util.tty as tty
from llnl.string import comma_and, quote
from llnl.util.filesystem import get_single_file, mkdirp, temp_cwd, temp_rename, working_dir
from llnl.util.symlink import symlink
import spack.config
import spack.error
import spack.oci.opener
import spack.url
import spack.util.crypto as crypto
import spack.util.git
import spack.util.url as url_util
import spack.util.web as web_util
import spack.version
import spack.version.git_ref_lookup
from spack.util.compression import decompressor_for
from spack.util.executable import CommandNotFoundError, which
all_strategies = []
CONTENT_TYPE_MISMATCH_WARNING_TEMPLATE = "The contents of {subject} look like {content_type}.  Either the URL you are trying to use does not exist or you have an internet gateway issue.  You can remove the bad archive using 'spack clean <package>', then try again using the correct URL."

def warn_content_type_mismatch(subject, content_type='HTML'):
    if False:
        print('Hello World!')
    tty.warn(CONTENT_TYPE_MISMATCH_WARNING_TEMPLATE.format(subject=subject, content_type=content_type))

def _needs_stage(fun):
    if False:
        return 10
    'Many methods on fetch strategies require a stage to be set\n    using set_stage().  This decorator adds a check for self.stage.'

    @functools.wraps(fun)
    def wrapper(self, *args, **kwargs):
        if False:
            return 10
        if not self.stage:
            raise NoStageError(fun)
        return fun(self, *args, **kwargs)
    return wrapper

def _ensure_one_stage_entry(stage_path):
    if False:
        return 10
    'Ensure there is only one stage entry in the stage path.'
    stage_entries = os.listdir(stage_path)
    assert len(stage_entries) == 1
    return os.path.join(stage_path, stage_entries[0])

def fetcher(cls):
    if False:
        for i in range(10):
            print('nop')
    'Decorator used to register fetch strategies.'
    all_strategies.append(cls)
    return cls

class FetchStrategy:
    """Superclass of all fetch strategies."""
    url_attr: Optional[str] = None
    optional_attrs: List[str] = []

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        self.stage = None
        self.cache_enabled = not kwargs.pop('no_cache', False)
        self.package = None

    def set_package(self, package):
        if False:
            i = 10
            return i + 15
        self.package = package

    def fetch(self):
        if False:
            print('Hello World!')
        'Fetch source code archive or repo.\n\n        Returns:\n            bool: True on success, False on failure.\n        '

    def check(self):
        if False:
            while True:
                i = 10
        'Checksum the archive fetched by this FetchStrategy.'

    def expand(self):
        if False:
            print('Hello World!')
        'Expand the downloaded archive into the stage source path.'

    def reset(self):
        if False:
            while True:
                i = 10
        'Revert to freshly downloaded state.\n\n        For archive files, this may just re-expand the archive.\n        '

    def archive(self, destination):
        if False:
            i = 10
            return i + 15
        'Create an archive of the downloaded data for a mirror.\n\n        For downloaded files, this should preserve the checksum of the\n        original file. For repositories, it should just create an\n        expandable tarball out of the downloaded repository.\n        '

    @property
    def cachable(self):
        if False:
            for i in range(10):
                print('nop')
        'Whether fetcher is capable of caching the resource it retrieves.\n\n        This generally is determined by whether the resource is\n        identifiably associated with a specific package version.\n\n        Returns:\n            bool: True if can cache, False otherwise.\n        '

    def source_id(self):
        if False:
            return 10
        'A unique ID for the source.\n\n        It is intended that a human could easily generate this themselves using\n        the information available to them in the Spack package.\n\n        The returned value is added to the content which determines the full\n        hash for a package using `str()`.\n        '
        raise NotImplementedError

    def mirror_id(self):
        if False:
            i = 10
            return i + 15
        'This is a unique ID for a source that is intended to help identify\n        reuse of resources across packages.\n\n        It is unique like source-id, but it does not include the package name\n        and is not necessarily easy for a human to create themselves.\n        '
        raise NotImplementedError

    def __str__(self):
        if False:
            return 10
        return 'FetchStrategy.__str___'

    @classmethod
    def matches(cls, args):
        if False:
            while True:
                i = 10
        'Predicate that matches fetch strategies to arguments of\n        the version directive.\n\n        Args:\n            args: arguments of the version directive\n        '
        return cls.url_attr in args

@fetcher
class BundleFetchStrategy(FetchStrategy):
    """
    Fetch strategy associated with bundle, or no-code, packages.

    Having a basic fetch strategy is a requirement for executing post-install
    hooks.  Consequently, this class provides the API but does little more
    than log messages.

    TODO: Remove this class by refactoring resource handling and the link
    between composite stages and composite fetch strategies (see #11981).
    """
    url_attr = ''

    def fetch(self):
        if False:
            for i in range(10):
                print('nop')
        'Simply report success -- there is no code to fetch.'
        return True

    @property
    def cachable(self):
        if False:
            while True:
                i = 10
        'Report False as there is no code to cache.'
        return False

    def source_id(self):
        if False:
            while True:
                i = 10
        "BundlePackages don't have a source id."
        return ''

    def mirror_id(self):
        if False:
            print('Hello World!')
        "BundlePackages don't have a mirror id."

@fetcher
class URLFetchStrategy(FetchStrategy):
    """URLFetchStrategy pulls source code from a URL for an archive, check the
    archive against a checksum, and decompresses the archive.

    The destination for the resulting file(s) is the standard stage path.
    """
    url_attr = 'url'
    optional_attrs = list(crypto.hashes.keys()) + ['checksum']

    def __init__(self, url=None, checksum=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.url = kwargs.get('url', url)
        self.mirrors = kwargs.get('mirrors', [])
        self.digest = kwargs.get('checksum', checksum)
        for h in self.optional_attrs:
            if h in kwargs:
                self.digest = kwargs[h]
        self.expand_archive = kwargs.get('expand', True)
        self.extra_options = kwargs.get('fetch_options', {})
        self._curl = None
        self.extension = kwargs.get('extension', None)
        if not self.url:
            raise ValueError('URLFetchStrategy requires a url for fetching.')

    @property
    def curl(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._curl:
            try:
                self._curl = which('curl', required=True)
            except CommandNotFoundError as exc:
                tty.error(str(exc))
        return self._curl

    def source_id(self):
        if False:
            for i in range(10):
                print('nop')
        return self.digest

    def mirror_id(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.digest:
            return None
        return os.path.sep.join(['archive', self.digest[:2], self.digest])

    @property
    def candidate_urls(self):
        if False:
            return 10
        return [self.url] + (self.mirrors or [])

    @_needs_stage
    def fetch(self):
        if False:
            return 10
        if self.archive_file:
            tty.debug('Already downloaded {0}'.format(self.archive_file))
            return
        url = None
        errors = []
        for url in self.candidate_urls:
            if not web_util.url_exists(url):
                tty.debug('URL does not exist: ' + url)
                continue
            try:
                self._fetch_from_url(url)
                break
            except FailedDownloadError as e:
                errors.append(str(e))
        for msg in errors:
            tty.debug(msg)
        if not self.archive_file:
            raise FailedDownloadError(url)

    def _fetch_from_url(self, url):
        if False:
            while True:
                i = 10
        if spack.config.get('config:url_fetch_method') == 'curl':
            return self._fetch_curl(url)
        else:
            return self._fetch_urllib(url)

    def _check_headers(self, headers):
        if False:
            for i in range(10):
                print('nop')
        content_types = re.findall('Content-Type:[^\\r\\n]+', headers, flags=re.IGNORECASE)
        if content_types and 'text/html' in content_types[-1]:
            warn_content_type_mismatch(self.archive_file or 'the archive')

    @_needs_stage
    def _fetch_urllib(self, url):
        if False:
            for i in range(10):
                print('nop')
        save_file = self.stage.save_filename
        tty.msg('Fetching {0}'.format(url))
        try:
            (url, headers, response) = web_util.read_from_url(url)
        except web_util.SpackWebError as e:
            if self.archive_file:
                os.remove(self.archive_file)
            if os.path.lexists(save_file):
                os.remove(save_file)
            msg = 'urllib failed to fetch with error {0}'.format(e)
            raise FailedDownloadError(url, msg)
        if os.path.lexists(save_file):
            os.remove(save_file)
        with open(save_file, 'wb') as _open_file:
            shutil.copyfileobj(response, _open_file)
        self._check_headers(str(headers))

    @_needs_stage
    def _fetch_curl(self, url):
        if False:
            while True:
                i = 10
        save_file = None
        partial_file = None
        if self.stage.save_filename:
            save_file = self.stage.save_filename
            partial_file = self.stage.save_filename + '.part'
        tty.msg('Fetching {0}'.format(url))
        if partial_file:
            save_args = ['-C', '-', '-o', partial_file]
        else:
            save_args = ['-O']
        timeout = 0
        cookie_args = []
        if self.extra_options:
            cookie = self.extra_options.get('cookie')
            if cookie:
                cookie_args.append('-j')
                cookie_args.append('-b')
                cookie_args.append(cookie)
            timeout = self.extra_options.get('timeout')
        base_args = web_util.base_curl_fetch_args(url, timeout)
        curl_args = save_args + base_args + cookie_args
        curl = self.curl
        with working_dir(self.stage.path):
            headers = curl(*curl_args, output=str, fail_on_error=False)
        if curl.returncode != 0:
            if self.archive_file:
                os.remove(self.archive_file)
            if partial_file and os.path.lexists(partial_file):
                os.remove(partial_file)
            try:
                web_util.check_curl_code(curl.returncode)
            except spack.error.FetchError as err:
                raise spack.fetch_strategy.FailedDownloadError(url, str(err))
        self._check_headers(headers)
        if save_file and partial_file is not None:
            fs.rename(partial_file, save_file)

    @property
    @_needs_stage
    def archive_file(self):
        if False:
            i = 10
            return i + 15
        'Path to the source archive within this stage directory.'
        return self.stage.archive_file

    @property
    def cachable(self):
        if False:
            return 10
        return self.cache_enabled and bool(self.digest)

    @_needs_stage
    def expand(self):
        if False:
            return 10
        if not self.expand_archive:
            tty.debug('Staging unexpanded archive {0} in {1}'.format(self.archive_file, self.stage.source_path))
            if not self.stage.expanded:
                mkdirp(self.stage.source_path)
            dest = os.path.join(self.stage.source_path, os.path.basename(self.archive_file))
            shutil.move(self.archive_file, dest)
            return
        tty.debug('Staging archive: {0}'.format(self.archive_file))
        if not self.archive_file:
            raise NoArchiveFileError("Couldn't find archive file", 'Failed on expand() for URL %s' % self.url)
        if not self.extension:
            self.extension = llnl.url.determine_url_file_extension(self.url)
        if self.stage.expanded:
            tty.debug('Source already staged to %s' % self.stage.source_path)
            return
        decompress = decompressor_for(self.archive_file, self.extension)
        with fs.exploding_archive_catch(self.stage):
            decompress(self.archive_file)

    def archive(self, destination):
        if False:
            for i in range(10):
                print('nop')
        'Just moves this archive to the destination.'
        if not self.archive_file:
            raise NoArchiveFileError('Cannot call archive() before fetching.')
        web_util.push_to_url(self.archive_file, url_util.path_to_file_url(destination), keep_original=True)

    @_needs_stage
    def check(self):
        if False:
            i = 10
            return i + 15
        'Check the downloaded archive against a checksum digest.\n        No-op if this stage checks code out of a repository.'
        if not self.digest:
            raise NoDigestError('Attempt to check URLFetchStrategy with no digest.')
        verify_checksum(self.archive_file, self.digest)

    @_needs_stage
    def reset(self):
        if False:
            print('Hello World!')
        '\n        Removes the source path if it exists, then re-expands the archive.\n        '
        if not self.archive_file:
            raise NoArchiveFileError('Tried to reset URLFetchStrategy before fetching', 'Failed on reset() for URL %s' % self.url)
        for filename in os.listdir(self.stage.path):
            abspath = os.path.join(self.stage.path, filename)
            if abspath != self.archive_file:
                shutil.rmtree(abspath, ignore_errors=True)
        self.expand()

    def __repr__(self):
        if False:
            return 10
        url = self.url if self.url else 'no url'
        return '%s<%s>' % (self.__class__.__name__, url)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.url:
            return self.url
        else:
            return '[no url]'

@fetcher
class CacheURLFetchStrategy(URLFetchStrategy):
    """The resource associated with a cache URL may be out of date."""

    @_needs_stage
    def fetch(self):
        if False:
            for i in range(10):
                print('nop')
        path = url_util.file_url_string_to_path(self.url)
        if not os.path.isfile(path):
            raise NoCacheError('No cache of %s' % path)
        filename = self.stage.save_filename
        if os.path.lexists(filename):
            os.remove(filename)
        symlink(path, filename)
        if self.digest:
            try:
                self.check()
            except ChecksumError:
                os.remove(self.archive_file)
                raise
        tty.msg('Using cached archive: {0}'.format(path))

class OCIRegistryFetchStrategy(URLFetchStrategy):

    def __init__(self, url=None, checksum=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(url, checksum, **kwargs)
        self._urlopen = kwargs.get('_urlopen', spack.oci.opener.urlopen)

    @_needs_stage
    def fetch(self):
        if False:
            print('Hello World!')
        file = self.stage.save_filename
        tty.msg(f'Fetching {self.url}')
        try:
            response = self._urlopen(self.url)
        except urllib.error.URLError as e:
            if self.archive_file:
                os.remove(self.archive_file)
            if os.path.lexists(file):
                os.remove(file)
            raise FailedDownloadError(self.url, f'Failed to fetch {self.url}: {e}') from e
        if os.path.lexists(file):
            os.remove(file)
        with open(file, 'wb') as f:
            shutil.copyfileobj(response, f)

class VCSFetchStrategy(FetchStrategy):
    """Superclass for version control system fetch strategies.

    Like all fetchers, VCS fetchers are identified by the attributes
    passed to the ``version`` directive.  The optional_attrs for a VCS
    fetch strategy represent types of revisions, e.g. tags, branches,
    commits, etc.

    The required attributes (git, svn, etc.) are used to specify the URL
    and to distinguish a VCS fetch strategy from a URL fetch strategy.

    """

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.url = kwargs.get(self.url_attr, None)
        if not self.url:
            raise ValueError('%s requires %s argument.' % (self.__class__, self.url_attr))
        for attr in self.optional_attrs:
            setattr(self, attr, kwargs.get(attr, None))

    @_needs_stage
    def check(self):
        if False:
            while True:
                i = 10
        tty.debug('No checksum needed when fetching with {0}'.format(self.url_attr))

    @_needs_stage
    def expand(self):
        if False:
            return 10
        tty.debug('Source fetched with %s is already expanded.' % self.url_attr)

    @_needs_stage
    def archive(self, destination, **kwargs):
        if False:
            print('Hello World!')
        assert llnl.url.extension_from_path(destination) == 'tar.gz'
        assert self.stage.source_path.startswith(self.stage.path)
        tar = which('tar', required=True)
        patterns = kwargs.get('exclude', None)
        if patterns is not None:
            if isinstance(patterns, str):
                patterns = [patterns]
            for p in patterns:
                tar.add_default_arg('--exclude=%s' % p)
        with working_dir(self.stage.path):
            if self.stage.srcdir:
                with temp_rename(self.stage.source_path, self.stage.srcdir):
                    tar('-czf', destination, self.stage.srcdir)
            else:
                tar('-czf', destination, os.path.basename(self.stage.source_path))

    def __str__(self):
        if False:
            print('Hello World!')
        return 'VCS: %s' % self.url

    def __repr__(self):
        if False:
            return 10
        return '%s<%s>' % (self.__class__, self.url)

@fetcher
class GoFetchStrategy(VCSFetchStrategy):
    """Fetch strategy that employs the `go get` infrastructure.

    Use like this in a package:

       version('name',
               go='github.com/monochromegane/the_platinum_searcher/...')

    Go get does not natively support versions, they can be faked with git.

    The fetched source will be moved to the standard stage sourcepath directory
    during the expand step.
    """
    url_attr = 'go'

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        forwarded_args = copy.copy(kwargs)
        forwarded_args.pop('name', None)
        super().__init__(**forwarded_args)
        self._go = None

    @property
    def go_version(self):
        if False:
            while True:
                i = 10
        vstring = self.go('version', output=str).split(' ')[2]
        return spack.version.Version(vstring)

    @property
    def go(self):
        if False:
            i = 10
            return i + 15
        if not self._go:
            self._go = which('go', required=True)
        return self._go

    @_needs_stage
    def fetch(self):
        if False:
            for i in range(10):
                print('nop')
        tty.debug('Getting go resource: {0}'.format(self.url))
        with working_dir(self.stage.path):
            try:
                os.mkdir('go')
            except OSError:
                pass
            env = dict(os.environ)
            env['GOPATH'] = os.path.join(os.getcwd(), 'go')
            self.go('get', '-v', '-d', self.url, env=env)

    def archive(self, destination):
        if False:
            while True:
                i = 10
        super().archive(destination, exclude='.git')

    @_needs_stage
    def expand(self):
        if False:
            return 10
        tty.debug('Source fetched with %s is already expanded.' % self.url_attr)
        repo_root = _ensure_one_stage_entry(self.stage.path)
        shutil.move(repo_root, self.stage.source_path)

    @_needs_stage
    def reset(self):
        if False:
            print('Hello World!')
        with working_dir(self.stage.source_path):
            self.go('clean')

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '[go] %s' % self.url

@fetcher
class GitFetchStrategy(VCSFetchStrategy):
    """
    Fetch strategy that gets source code from a git repository.
    Use like this in a package:

        version('name', git='https://github.com/project/repo.git')

    Optionally, you can provide a branch, or commit to check out, e.g.:

        version('1.1', git='https://github.com/project/repo.git', tag='v1.1')

    You can use these three optional attributes in addition to ``git``:

        * ``branch``: Particular branch to build from (default is the
                      repository's default branch)
        * ``tag``: Particular tag to check out
        * ``commit``: Particular commit hash in the repo

    Repositories are cloned into the standard stage source path directory.
    """
    url_attr = 'git'
    optional_attrs = ['tag', 'branch', 'commit', 'submodules', 'get_full_repo', 'submodules_delete']
    git_version_re = 'git version (\\S+)'

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        forwarded_args = copy.copy(kwargs)
        forwarded_args.pop('name', None)
        super().__init__(**forwarded_args)
        self._git = None
        self.submodules = kwargs.get('submodules', False)
        self.submodules_delete = kwargs.get('submodules_delete', False)
        self.get_full_repo = kwargs.get('get_full_repo', False)

    @property
    def git_version(self):
        if False:
            print('Hello World!')
        return GitFetchStrategy.version_from_git(self.git)

    @staticmethod
    def version_from_git(git_exe):
        if False:
            i = 10
            return i + 15
        'Given a git executable, return the Version (this will fail if\n        the output cannot be parsed into a valid Version).\n        '
        version_output = git_exe('--version', output=str)
        m = re.search(GitFetchStrategy.git_version_re, version_output)
        return spack.version.Version(m.group(1))

    @property
    def git(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._git:
            try:
                self._git = spack.util.git.git(required=True)
            except CommandNotFoundError as exc:
                tty.error(str(exc))
                raise
            if self.git_version >= spack.version.Version('1.7.2'):
                self._git.add_default_arg('-c', 'advice.detachedHead=false')
            if not spack.config.get('config:verify_ssl'):
                self._git.add_default_env('GIT_SSL_NO_VERIFY', 'true')
        return self._git

    @property
    def cachable(self):
        if False:
            return 10
        return self.cache_enabled and bool(self.commit or self.tag)

    def source_id(self):
        if False:
            i = 10
            return i + 15
        return self.commit or self.tag

    def mirror_id(self):
        if False:
            for i in range(10):
                print('nop')
        repo_ref = self.commit or self.tag or self.branch
        if repo_ref:
            repo_path = urllib.parse.urlparse(self.url).path
            result = os.path.sep.join(['git', repo_path, repo_ref])
            return result

    def _repo_info(self):
        if False:
            print('Hello World!')
        args = ''
        if self.commit:
            args = ' at commit {0}'.format(self.commit)
        elif self.tag:
            args = ' at tag {0}'.format(self.tag)
        elif self.branch:
            args = ' on branch {0}'.format(self.branch)
        return '{0}{1}'.format(self.url, args)

    @_needs_stage
    def fetch(self):
        if False:
            i = 10
            return i + 15
        if self.stage.expanded:
            tty.debug('Already fetched {0}'.format(self.stage.source_path))
            return
        self.clone(commit=self.commit, branch=self.branch, tag=self.tag)

    def clone(self, dest=None, commit=None, branch=None, tag=None, bare=False):
        if False:
            while True:
                i = 10
        '\n        Clone a repository to a path.\n\n        This method handles cloning from git, but does not require a stage.\n\n        Arguments:\n            dest (str or None): The path into which the code is cloned. If None,\n                requires a stage and uses the stage\'s source path.\n            commit (str or None): A commit to fetch from the remote. Only one of\n                commit, branch, and tag may be non-None.\n            branch (str or None): A branch to fetch from the remote.\n            tag (str or None): A tag to fetch from the remote.\n            bare (bool): Execute a "bare" git clone (--bare option to git)\n        '
        dest = dest or self.stage.source_path
        tty.debug('Cloning git repository: {0}'.format(self._repo_info()))
        git = self.git
        debug = spack.config.get('config:debug')
        if bare:
            clone_args = ['clone', '--bare']
            if not debug:
                clone_args.append('--quiet')
            clone_args.extend([self.url, dest])
            git(*clone_args)
        elif commit:
            clone_args = ['clone', self.url]
            if not debug:
                clone_args.insert(1, '--quiet')
            with temp_cwd():
                git(*clone_args)
                repo_name = get_single_file('.')
                if self.stage:
                    self.stage.srcdir = repo_name
                shutil.copytree(repo_name, dest, symlinks=True)
                shutil.rmtree(repo_name, ignore_errors=False, onerror=fs.readonly_file_handler(ignore_errors=True))
            with working_dir(dest):
                checkout_args = ['checkout', commit]
                if not debug:
                    checkout_args.insert(1, '--quiet')
                git(*checkout_args)
        else:
            args = ['clone']
            if not debug:
                args.append('--quiet')
            if branch:
                args.extend(['--branch', branch])
            elif tag and self.git_version >= spack.version.Version('1.8.5.2'):
                args.extend(['--branch', tag])
            if self.git_version >= spack.version.Version('1.7.10'):
                if self.get_full_repo:
                    args.append('--no-single-branch')
                else:
                    args.append('--single-branch')
            with temp_cwd():
                if not self.get_full_repo and self.git_version >= spack.version.Version('1.7.1') and self.protocol_supports_shallow_clone():
                    args.extend(['--depth', '1'])
                args.extend([self.url])
                git(*args)
                repo_name = get_single_file('.')
                if self.stage:
                    self.stage.srcdir = repo_name
                shutil.move(repo_name, dest)
            with working_dir(dest):
                if tag and self.git_version < spack.version.Version('1.8.5.2'):
                    pull_args = ['pull', '--tags']
                    co_args = ['checkout', self.tag]
                    if not spack.config.get('config:debug'):
                        pull_args.insert(1, '--quiet')
                        co_args.insert(1, '--quiet')
                    git(*pull_args, ignore_errors=1)
                    git(*co_args)
        if self.submodules_delete:
            with working_dir(dest):
                for submodule_to_delete in self.submodules_delete:
                    args = ['rm', submodule_to_delete]
                    if not spack.config.get('config:debug'):
                        args.insert(1, '--quiet')
                    git(*args)
        git_commands = []
        submodules = self.submodules
        if callable(submodules):
            submodules = list(submodules(self.package))
            git_commands.append(['submodule', 'init', '--'] + submodules)
            git_commands.append(['submodule', 'update', '--recursive'])
        elif submodules:
            git_commands.append(['submodule', 'update', '--init', '--recursive'])
        if not git_commands:
            return
        with working_dir(dest):
            for args in git_commands:
                if not spack.config.get('config:debug'):
                    args.insert(1, '--quiet')
                git(*args)

    def archive(self, destination):
        if False:
            return 10
        super().archive(destination, exclude='.git')

    @_needs_stage
    def reset(self):
        if False:
            print('Hello World!')
        with working_dir(self.stage.source_path):
            co_args = ['checkout', '.']
            clean_args = ['clean', '-f']
            if spack.config.get('config:debug'):
                co_args.insert(1, '--quiet')
                clean_args.insert(1, '--quiet')
            self.git(*co_args)
            self.git(*clean_args)

    def protocol_supports_shallow_clone(self):
        if False:
            return 10
        'Shallow clone operations (--depth #) are not supported by the basic\n        HTTP protocol or by no-protocol file specifications.\n        Use (e.g.) https:// or file:// instead.'
        return not (self.url.startswith('http://') or self.url.startswith('/'))

    def __str__(self):
        if False:
            while True:
                i = 10
        return '[git] {0}'.format(self._repo_info())

@fetcher
class CvsFetchStrategy(VCSFetchStrategy):
    """Fetch strategy that gets source code from a CVS repository.
       Use like this in a package:

           version('name',
                   cvs=':pserver:anonymous@www.example.com:/cvsroot%module=modulename')

       Optionally, you can provide a branch and/or a date for the URL:

           version('name',
                   cvs=':pserver:anonymous@www.example.com:/cvsroot%module=modulename',
                   branch='branchname', date='date')

    Repositories are checked out into the standard stage source path directory.
    """
    url_attr = 'cvs'
    optional_attrs = ['branch', 'date']

    def __init__(self, **kwargs):
        if False:
            return 10
        forwarded_args = copy.copy(kwargs)
        forwarded_args.pop('name', None)
        super().__init__(**forwarded_args)
        self._cvs = None
        if self.branch is not None:
            self.branch = str(self.branch)
        if self.date is not None:
            self.date = str(self.date)

    @property
    def cvs(self):
        if False:
            while True:
                i = 10
        if not self._cvs:
            self._cvs = which('cvs', required=True)
        return self._cvs

    @property
    def cachable(self):
        if False:
            print('Hello World!')
        return self.cache_enabled and (bool(self.branch) or bool(self.date))

    def source_id(self):
        if False:
            return 10
        if not (self.branch or self.date):
            return None
        id = 'id'
        if self.branch:
            id += '-branch=' + self.branch
        if self.date:
            id += '-date=' + self.date
        return id

    def mirror_id(self):
        if False:
            for i in range(10):
                print('nop')
        if not (self.branch or self.date):
            return None
        elements = self.url.split(':')
        final = elements[-1]
        elements = final.split('/')
        elements = elements[1:]
        result = os.path.sep.join(['cvs'] + elements)
        if self.branch:
            result += '%branch=' + self.branch
        if self.date:
            result += '%date=' + self.date
        return result

    @_needs_stage
    def fetch(self):
        if False:
            print('Hello World!')
        if self.stage.expanded:
            tty.debug('Already fetched {0}'.format(self.stage.source_path))
            return
        tty.debug('Checking out CVS repository: {0}'.format(self.url))
        with temp_cwd():
            (url, module) = self.url.split('%module=')
            args = ['-z9', '-d', url, 'checkout']
            if self.branch is not None:
                args.extend(['-r', self.branch])
            if self.date is not None:
                args.extend(['-D', self.date])
            args.append(module)
            self.cvs(*args)
            repo_name = get_single_file('.')
            self.stage.srcdir = repo_name
            shutil.move(repo_name, self.stage.source_path)

    def _remove_untracked_files(self):
        if False:
            while True:
                i = 10
        'Removes untracked files in a CVS repository.'
        with working_dir(self.stage.source_path):
            status = self.cvs('-qn', 'update', output=str)
            for line in status.split('\n'):
                if re.match('^[?]', line):
                    path = line[2:].strip()
                    if os.path.isfile(path):
                        os.unlink(path)

    def archive(self, destination):
        if False:
            for i in range(10):
                print('nop')
        super().archive(destination, exclude='CVS')

    @_needs_stage
    def reset(self):
        if False:
            i = 10
            return i + 15
        self._remove_untracked_files()
        with working_dir(self.stage.source_path):
            self.cvs('update', '-C', '.')

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '[cvs] %s' % self.url

@fetcher
class SvnFetchStrategy(VCSFetchStrategy):
    """Fetch strategy that gets source code from a subversion repository.
       Use like this in a package:

           version('name', svn='http://www.example.com/svn/trunk')

       Optionally, you can provide a revision for the URL:

           version('name', svn='http://www.example.com/svn/trunk',
                   revision='1641')

    Repositories are checked out into the standard stage source path directory.
    """
    url_attr = 'svn'
    optional_attrs = ['revision']

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        forwarded_args = copy.copy(kwargs)
        forwarded_args.pop('name', None)
        super().__init__(**forwarded_args)
        self._svn = None
        if self.revision is not None:
            self.revision = str(self.revision)

    @property
    def svn(self):
        if False:
            i = 10
            return i + 15
        if not self._svn:
            self._svn = which('svn', required=True)
        return self._svn

    @property
    def cachable(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cache_enabled and bool(self.revision)

    def source_id(self):
        if False:
            i = 10
            return i + 15
        return self.revision

    def mirror_id(self):
        if False:
            while True:
                i = 10
        if self.revision:
            repo_path = urllib.parse.urlparse(self.url).path
            result = os.path.sep.join(['svn', repo_path, self.revision])
            return result

    @_needs_stage
    def fetch(self):
        if False:
            return 10
        if self.stage.expanded:
            tty.debug('Already fetched {0}'.format(self.stage.source_path))
            return
        tty.debug('Checking out subversion repository: {0}'.format(self.url))
        args = ['checkout', '--force', '--quiet']
        if self.revision:
            args += ['-r', self.revision]
        args.extend([self.url])
        with temp_cwd():
            self.svn(*args)
            repo_name = get_single_file('.')
            self.stage.srcdir = repo_name
            shutil.move(repo_name, self.stage.source_path)

    def _remove_untracked_files(self):
        if False:
            i = 10
            return i + 15
        'Removes untracked files in an svn repository.'
        with working_dir(self.stage.source_path):
            status = self.svn('status', '--no-ignore', output=str)
            self.svn('status', '--no-ignore')
            for line in status.split('\n'):
                if not re.match('^[I?]', line):
                    continue
                path = line[8:].strip()
                if os.path.isfile(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)

    def archive(self, destination):
        if False:
            return 10
        super().archive(destination, exclude='.svn')

    @_needs_stage
    def reset(self):
        if False:
            while True:
                i = 10
        self._remove_untracked_files()
        with working_dir(self.stage.source_path):
            self.svn('revert', '.', '-R')

    def __str__(self):
        if False:
            return 10
        return '[svn] %s' % self.url

@fetcher
class HgFetchStrategy(VCSFetchStrategy):
    """
    Fetch strategy that gets source code from a Mercurial repository.
    Use like this in a package:

        version('name', hg='https://jay.grs.rwth-aachen.de/hg/lwm2')

    Optionally, you can provide a branch, or revision to check out, e.g.:

        version('torus',
                hg='https://jay.grs.rwth-aachen.de/hg/lwm2', branch='torus')

    You can use the optional 'revision' attribute to check out a
    branch, tag, or particular revision in hg.  To prevent
    non-reproducible builds, using a moving target like a branch is
    discouraged.

        * ``revision``: Particular revision, branch, or tag.

    Repositories are cloned into the standard stage source path directory.
    """
    url_attr = 'hg'
    optional_attrs = ['revision']

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        forwarded_args = copy.copy(kwargs)
        forwarded_args.pop('name', None)
        super().__init__(**forwarded_args)
        self._hg = None

    @property
    def hg(self):
        if False:
            while True:
                i = 10
        '\n        Returns:\n            Executable: the hg executable\n        '
        if not self._hg:
            self._hg = which('hg', required=True)
            self._hg.add_default_env('PYTHONPATH', '')
        return self._hg

    @property
    def cachable(self):
        if False:
            return 10
        return self.cache_enabled and bool(self.revision)

    def source_id(self):
        if False:
            print('Hello World!')
        return self.revision

    def mirror_id(self):
        if False:
            print('Hello World!')
        if self.revision:
            repo_path = urllib.parse.urlparse(self.url).path
            result = os.path.sep.join(['hg', repo_path, self.revision])
            return result

    @_needs_stage
    def fetch(self):
        if False:
            while True:
                i = 10
        if self.stage.expanded:
            tty.debug('Already fetched {0}'.format(self.stage.source_path))
            return
        args = []
        if self.revision:
            args.append('at revision %s' % self.revision)
        tty.debug('Cloning mercurial repository: {0} {1}'.format(self.url, args))
        args = ['clone']
        if not spack.config.get('config:verify_ssl'):
            args.append('--insecure')
        if self.revision:
            args.extend(['-r', self.revision])
        args.extend([self.url])
        with temp_cwd():
            self.hg(*args)
            repo_name = get_single_file('.')
            self.stage.srcdir = repo_name
            shutil.move(repo_name, self.stage.source_path)

    def archive(self, destination):
        if False:
            for i in range(10):
                print('nop')
        super().archive(destination, exclude='.hg')

    @_needs_stage
    def reset(self):
        if False:
            return 10
        with working_dir(self.stage.path):
            source_path = self.stage.source_path
            scrubbed = 'scrubbed-source-tmp'
            args = ['clone']
            if self.revision:
                args += ['-r', self.revision]
            args += [source_path, scrubbed]
            self.hg(*args)
            shutil.rmtree(source_path, ignore_errors=True)
            shutil.move(scrubbed, source_path)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '[hg] %s' % self.url

@fetcher
class S3FetchStrategy(URLFetchStrategy):
    """FetchStrategy that pulls from an S3 bucket."""
    url_attr = 's3'

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            super().__init__(*args, **kwargs)
        except ValueError:
            if not kwargs.get('url'):
                raise ValueError('S3FetchStrategy requires a url for fetching.')

    @_needs_stage
    def fetch(self):
        if False:
            while True:
                i = 10
        if self.archive_file:
            tty.debug('Already downloaded {0}'.format(self.archive_file))
            return
        parsed_url = urllib.parse.urlparse(self.url)
        if parsed_url.scheme != 's3':
            raise spack.error.FetchError('S3FetchStrategy can only fetch from s3:// urls.')
        tty.debug('Fetching {0}'.format(self.url))
        basename = os.path.basename(parsed_url.path)
        with working_dir(self.stage.path):
            (_, headers, stream) = web_util.read_from_url(self.url)
            with open(basename, 'wb') as f:
                shutil.copyfileobj(stream, f)
            content_type = web_util.get_header(headers, 'Content-type')
        if content_type == 'text/html':
            warn_content_type_mismatch(self.archive_file or 'the archive')
        if self.stage.save_filename:
            llnl.util.filesystem.rename(os.path.join(self.stage.path, basename), self.stage.save_filename)
        if not self.archive_file:
            raise FailedDownloadError(self.url)

@fetcher
class GCSFetchStrategy(URLFetchStrategy):
    """FetchStrategy that pulls from a GCS bucket."""
    url_attr = 'gs'

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            super().__init__(*args, **kwargs)
        except ValueError:
            if not kwargs.get('url'):
                raise ValueError('GCSFetchStrategy requires a url for fetching.')

    @_needs_stage
    def fetch(self):
        if False:
            print('Hello World!')
        if self.archive_file:
            tty.debug('Already downloaded {0}'.format(self.archive_file))
            return
        parsed_url = urllib.parse.urlparse(self.url)
        if parsed_url.scheme != 'gs':
            raise spack.error.FetchError('GCSFetchStrategy can only fetch from gs:// urls.')
        tty.debug('Fetching {0}'.format(self.url))
        basename = os.path.basename(parsed_url.path)
        with working_dir(self.stage.path):
            (_, headers, stream) = web_util.read_from_url(self.url)
            with open(basename, 'wb') as f:
                shutil.copyfileobj(stream, f)
            content_type = web_util.get_header(headers, 'Content-type')
        if content_type == 'text/html':
            warn_content_type_mismatch(self.archive_file or 'the archive')
        if self.stage.save_filename:
            os.rename(os.path.join(self.stage.path, basename), self.stage.save_filename)
        if not self.archive_file:
            raise FailedDownloadError(self.url)

@fetcher
class FetchAndVerifyExpandedFile(URLFetchStrategy):
    """Fetch strategy that verifies the content digest during fetching,
    as well as after expanding it."""

    def __init__(self, url, archive_sha256: str, expanded_sha256: str):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(url, archive_sha256)
        self.expanded_sha256 = expanded_sha256

    def expand(self):
        if False:
            i = 10
            return i + 15
        'Verify checksum after expanding the archive.'
        super().expand()
        src_dir = self.stage.source_path
        files = os.listdir(src_dir)
        if len(files) != 1:
            raise ChecksumError(self, f'Expected a single file in {src_dir}.')
        verify_checksum(os.path.join(src_dir, files[0]), self.expanded_sha256)

def verify_checksum(file, digest):
    if False:
        return 10
    checker = crypto.Checker(digest)
    if not checker.check(file):
        (size, contents) = fs.filesummary(file)
        raise ChecksumError(f'{checker.hash_name} checksum failed for {file}', f'Expected {digest} but got {checker.sum}. File size = {size} bytes. Contents = {contents!r}')

def stable_target(fetcher):
    if False:
        for i in range(10):
            print('nop')
    'Returns whether the fetcher target is expected to have a stable\n    checksum. This is only true if the target is a preexisting archive\n    file.'
    if isinstance(fetcher, URLFetchStrategy) and fetcher.cachable:
        return True
    return False

def from_url(url):
    if False:
        print('Hello World!')
    'Given a URL, find an appropriate fetch strategy for it.\n    Currently just gives you a URLFetchStrategy that uses curl.\n\n    TODO: make this return appropriate fetch strategies for other\n          types of URLs.\n    '
    return URLFetchStrategy(url)

def from_kwargs(**kwargs):
    if False:
        return 10
    'Construct an appropriate FetchStrategy from the given keyword arguments.\n\n    Args:\n        **kwargs: dictionary of keyword arguments, e.g. from a\n            ``version()`` directive in a package.\n\n    Returns:\n        typing.Callable: The fetch strategy that matches the args, based\n            on attribute names (e.g., ``git``, ``hg``, etc.)\n\n    Raises:\n        spack.error.FetchError: If no ``fetch_strategy`` matches the args.\n    '
    for fetcher in all_strategies:
        if fetcher.matches(kwargs):
            return fetcher(**kwargs)
    raise InvalidArgsError(**kwargs)

def check_pkg_attributes(pkg):
    if False:
        print('Hello World!')
    'Find ambiguous top-level fetch attributes in a package.\n\n    Currently this only ensures that two or more VCS fetch strategies are\n    not specified at once.\n    '
    conflicts = set([s.url_attr for s in all_strategies if hasattr(pkg, s.url_attr)])
    conflicts -= set(['url'])
    if len(conflicts) > 1:
        raise FetcherConflict('Package %s cannot specify %s together. Pick at most one.' % (pkg.name, comma_and(quote(conflicts))))

def _check_version_attributes(fetcher, pkg, version):
    if False:
        print('Hello World!')
    'Ensure that the fetcher for a version is not ambiguous.\n\n    This assumes that we have already determined the fetcher for the\n    specific version using ``for_package_version()``\n    '
    all_optionals = set((a for s in all_strategies for a in s.optional_attrs))
    args = pkg.versions[version]
    extra = set(args) - set(fetcher.optional_attrs) - set([fetcher.url_attr, 'no_cache'])
    extra.intersection_update(all_optionals)
    if extra:
        legal_attrs = [fetcher.url_attr] + list(fetcher.optional_attrs)
        raise FetcherConflict("%s version '%s' has extra arguments: %s" % (pkg.name, version, comma_and(quote(extra))), 'Valid arguments for a %s fetcher are: \n    %s' % (fetcher.url_attr, comma_and(quote(legal_attrs))))

def _extrapolate(pkg, version):
    if False:
        for i in range(10):
            print('nop')
    'Create a fetcher from an extrapolated URL for this version.'
    try:
        return URLFetchStrategy(pkg.url_for_version(version), fetch_options=pkg.fetch_options)
    except spack.package_base.NoURLError:
        msg = "Can't extrapolate a URL for version %s because package %s defines no URLs"
        raise ExtrapolationError(msg % (version, pkg.name))

def _from_merged_attrs(fetcher, pkg, version):
    if False:
        for i in range(10):
            print('nop')
    'Create a fetcher from merged package and version attributes.'
    if fetcher.url_attr == 'url':
        mirrors = pkg.all_urls_for_version(version)
        url = mirrors[0]
        mirrors = mirrors[1:]
        attrs = {fetcher.url_attr: url, 'mirrors': mirrors}
    else:
        url = getattr(pkg, fetcher.url_attr)
        attrs = {fetcher.url_attr: url}
    attrs['fetch_options'] = pkg.fetch_options
    attrs.update(pkg.versions[version])
    if fetcher.url_attr == 'git' and hasattr(pkg, 'submodules'):
        attrs.setdefault('submodules', pkg.submodules)
    return fetcher(**attrs)

def for_package_version(pkg, version=None):
    if False:
        for i in range(10):
            print('nop')
    'Determine a fetch strategy based on the arguments supplied to\n    version() in the package description.'
    if not pkg.has_code:
        return BundleFetchStrategy()
    check_pkg_attributes(pkg)
    if version is not None:
        assert not pkg.spec.concrete, "concrete specs should not pass the 'version=' argument"
        if not isinstance(version, spack.version.StandardVersion):
            version = spack.version.Version(version)
        version_list = spack.version.VersionList()
        version_list.add(version)
        pkg.spec.versions = version_list
    else:
        version = pkg.version
    if isinstance(version, spack.version.GitVersion):
        if not hasattr(pkg, 'git'):
            raise spack.error.FetchError(f"Cannot fetch git version for {pkg.name}. Package has no 'git' attribute")
        version.attach_lookup(spack.version.git_ref_lookup.GitRefLookup(pkg.name))
        ref_type = 'commit' if version.is_commit else 'tag'
        kwargs = {'git': pkg.git, ref_type: version.ref, 'no_cache': True}
        kwargs['submodules'] = getattr(pkg, 'submodules', False)
        ref_version_attributes = pkg.versions.get(pkg.version.ref_version)
        if ref_version_attributes:
            kwargs['submodules'] = ref_version_attributes.get('submodules', kwargs['submodules'])
        fetcher = GitFetchStrategy(**kwargs)
        return fetcher
    if version not in pkg.versions:
        return _extrapolate(pkg, version)
    args = {'fetch_options': pkg.fetch_options}
    args.update(pkg.versions[version])
    for fetcher in all_strategies:
        if fetcher.url_attr in args:
            _check_version_attributes(fetcher, pkg, version)
            if fetcher.url_attr == 'git' and hasattr(pkg, 'submodules'):
                args.setdefault('submodules', pkg.submodules)
            return fetcher(**args)
    for fetcher in all_strategies:
        if hasattr(pkg, fetcher.url_attr) or fetcher.url_attr == 'url':
            optionals = fetcher.optional_attrs
            if optionals and any((a in args for a in optionals)):
                _check_version_attributes(fetcher, pkg, version)
                return _from_merged_attrs(fetcher, pkg, version)
    for fetcher in all_strategies:
        if hasattr(pkg, fetcher.url_attr):
            _check_version_attributes(fetcher, pkg, version)
            return _from_merged_attrs(fetcher, pkg, version)
    raise InvalidArgsError(pkg, version, **args)

def from_url_scheme(url, *args, **kwargs):
    if False:
        return 10
    'Finds a suitable FetchStrategy by matching its url_attr with the scheme\n    in the given url.'
    url = kwargs.get('url', url)
    parsed_url = urllib.parse.urlparse(url, scheme='file')
    scheme_mapping = kwargs.get('scheme_mapping') or {'file': 'url', 'http': 'url', 'https': 'url', 'ftp': 'url', 'ftps': 'url'}
    scheme = parsed_url.scheme
    scheme = scheme_mapping.get(scheme, scheme)
    for fetcher in all_strategies:
        url_attr = getattr(fetcher, 'url_attr', None)
        if url_attr and url_attr == scheme:
            return fetcher(url, *args, **kwargs)
    raise ValueError('No FetchStrategy found for url with scheme: "{SCHEME}"'.format(SCHEME=parsed_url.scheme))

def from_list_url(pkg):
    if False:
        while True:
            i = 10
    "If a package provides a URL which lists URLs for resources by\n    version, this can can create a fetcher for a URL discovered for\n    the specified package's version."
    if pkg.list_url:
        try:
            versions = pkg.fetch_remote_versions()
            try:
                url_from_list = versions[pkg.version]
                checksum = None
                version = pkg.version
                if version in pkg.versions:
                    args = pkg.versions[version]
                    checksum = next((v for (k, v) in args.items() if k in crypto.hashes), args.get('checksum'))
                return URLFetchStrategy(url_from_list, checksum, fetch_options=pkg.fetch_options)
            except KeyError as e:
                tty.debug(e)
                tty.msg('Cannot find version %s in url_list' % pkg.version)
        except BaseException as e:
            tty.debug(e)
            tty.msg('Could not determine url from list_url.')

class FsCache:

    def __init__(self, root):
        if False:
            while True:
                i = 10
        self.root = os.path.abspath(root)

    def store(self, fetcher, relative_dest):
        if False:
            return 10
        if not fetcher.cachable:
            return
        if isinstance(fetcher, CacheURLFetchStrategy):
            return
        dst = os.path.join(self.root, relative_dest)
        mkdirp(os.path.dirname(dst))
        fetcher.archive(dst)

    def fetcher(self, target_path, digest, **kwargs):
        if False:
            while True:
                i = 10
        path = os.path.join(self.root, target_path)
        url = url_util.path_to_file_url(path)
        return CacheURLFetchStrategy(url, digest, **kwargs)

    def destroy(self):
        if False:
            return 10
        shutil.rmtree(self.root, ignore_errors=True)

class NoCacheError(spack.error.FetchError):
    """Raised when there is no cached archive for a package."""

class FailedDownloadError(spack.error.FetchError):
    """Raised when a download fails."""

    def __init__(self, url, msg=''):
        if False:
            print('Hello World!')
        super().__init__('Failed to fetch file from URL: %s' % url, msg)
        self.url = url

class NoArchiveFileError(spack.error.FetchError):
    """Raised when an archive file is expected but none exists."""

class NoDigestError(spack.error.FetchError):
    """Raised after attempt to checksum when URL has no digest."""

class ExtrapolationError(spack.error.FetchError):
    """Raised when we can't extrapolate a version for a package."""

class FetcherConflict(spack.error.FetchError):
    """Raised for packages with invalid fetch attributes."""

class InvalidArgsError(spack.error.FetchError):
    """Raised when a version can't be deduced from a set of arguments."""

    def __init__(self, pkg=None, version=None, **args):
        if False:
            print('Hello World!')
        msg = 'Could not guess a fetch strategy'
        if pkg:
            msg += ' for {pkg}'.format(pkg=pkg)
            if version:
                msg += '@{version}'.format(version=version)
        long_msg = 'with arguments: {args}'.format(args=args)
        super().__init__(msg, long_msg)

class ChecksumError(spack.error.FetchError):
    """Raised when archive fails to checksum."""

class NoStageError(spack.error.FetchError):
    """Raised when fetch operations are called before set_stage()."""

    def __init__(self, method):
        if False:
            while True:
                i = 10
        super().__init__('Must call FetchStrategy.set_stage() before calling %s' % method.__name__)