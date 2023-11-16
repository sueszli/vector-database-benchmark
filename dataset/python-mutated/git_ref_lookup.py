import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
from llnl.util.filesystem import mkdirp, working_dir
import spack.caches
import spack.fetch_strategy
import spack.paths
import spack.repo
import spack.util.executable
import spack.util.hash
import spack.util.spack_json as sjson
import spack.version
from .common import VersionLookupError
from .lookup import AbstractRefLookup
_VERSION_CORE = '\\d+\\.\\d+\\.\\d+'
_IDENT = '[0-9A-Za-z-]+'
_SEPARATED_IDENT = f'{_IDENT}(?:\\.{_IDENT})*'
_PRERELEASE = f'\\-{_SEPARATED_IDENT}'
_BUILD = f'\\+{_SEPARATED_IDENT}'
_SEMVER = f'{_VERSION_CORE}(?:{_PRERELEASE})?(?:{_BUILD})?'
SEMVER_REGEX = re.compile(f'{_SEMVER}$')

class GitRefLookup(AbstractRefLookup):
    """An object for cached lookups of git refs

    GitRefLookup objects delegate to the MISC_CACHE for locking. GitRefLookup objects may
    be attached to a GitVersion to allow for comparisons between git refs and versions as
    represented by tags in the git repository.
    """

    def __init__(self, pkg_name):
        if False:
            for i in range(10):
                print('nop')
        self.pkg_name = pkg_name
        self.data: Dict[str, Tuple[Optional[str], int]] = {}
        self._pkg = None
        self._fetcher = None
        self._cache_key = None
        self._cache_path = None

    @property
    def cache_key(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._cache_key:
            key_base = 'git_metadata'
            self._cache_key = (Path(key_base) / self.repository_uri).as_posix()
            spack.caches.MISC_CACHE.init_entry(self.cache_key)
        return self._cache_key

    @property
    def cache_path(self):
        if False:
            return 10
        if not self._cache_path:
            self._cache_path = spack.caches.MISC_CACHE.cache_path(self.cache_key)
        return self._cache_path

    @property
    def pkg(self):
        if False:
            return 10
        if not self._pkg:
            try:
                pkg = spack.repo.PATH.get_pkg_class(self.pkg_name)
                pkg.git
            except (spack.repo.RepoError, AttributeError) as e:
                raise VersionLookupError(f"Couldn't get the git repo for {self.pkg_name}") from e
            self._pkg = pkg
        return self._pkg

    @property
    def fetcher(self):
        if False:
            while True:
                i = 10
        if not self._fetcher:
            fetcher = spack.fetch_strategy.GitFetchStrategy(git=self.pkg.git)
            fetcher.get_full_repo = True
            self._fetcher = fetcher
        return self._fetcher

    @property
    def repository_uri(self):
        if False:
            return 10
        'Identifier for git repos used within the repo and metadata caches.'
        return Path(spack.util.hash.b32_hash(self.pkg.git)[-7:])

    def save(self):
        if False:
            print('Hello World!')
        'Save the data to file'
        with spack.caches.MISC_CACHE.write_transaction(self.cache_key) as (old, new):
            sjson.dump(self.data, new)

    def load_data(self):
        if False:
            print('Hello World!')
        'Load data if the path already exists.'
        if os.path.isfile(self.cache_path):
            with spack.caches.MISC_CACHE.read_transaction(self.cache_key) as cache_file:
                self.data = sjson.load(cache_file)

    def get(self, ref) -> Tuple[Optional[str], int]:
        if False:
            while True:
                i = 10
        if not self.data:
            self.load_data()
        if ref not in self.data:
            self.data[ref] = self.lookup_ref(ref)
            self.save()
        return self.data[ref]

    def lookup_ref(self, ref) -> Tuple[Optional[str], int]:
        if False:
            while True:
                i = 10
        'Lookup the previous version and distance for a given commit.\n\n        We use git to compare the known versions from package to the git tags,\n        as well as any git tags that are SEMVER versions, and find the latest\n        known version prior to the commit, as well as the distance from that version\n        to the commit in the git repo. Those values are used to compare Version objects.\n        '
        pathlib_dest = Path(spack.paths.user_repos_cache_path) / self.repository_uri
        dest = str(pathlib_dest)
        dest_parent = os.path.dirname(dest)
        if not os.path.exists(dest_parent):
            mkdirp(dest_parent)
        if not os.path.exists(dest):
            self.fetcher.clone(dest, bare=True)
        with working_dir(dest):
            self.fetcher.git('fetch', '--tags', output=os.devnull, error=os.devnull)
            try:
                self.fetcher.git('cat-file', '-e', '%s^{commit}' % ref, output=os.devnull, error=os.devnull)
            except spack.util.executable.ProcessError:
                raise VersionLookupError('%s is not a valid git ref for %s' % (ref, self.pkg_name))
            tag_info = self.fetcher.git('for-each-ref', '--sort=creatordate', '--format', '%(objectname) %(refname)', 'refs/tags', output=str).split('\n')
            commit_to_version = {}
            for entry in tag_info:
                if not entry:
                    continue
                (tag_commit, tag) = entry.split()
                tag = tag.replace('refs/tags/', '', 1)
                for v in [v.string for v in self.pkg.versions]:
                    if v == tag or 'v' + v == tag:
                        commit_to_version[tag_commit] = v
                        break
                else:
                    match = SEMVER_REGEX.search(tag)
                    if match:
                        commit_to_version[tag_commit] = match.group()
            ancestor_commits = []
            for tag_commit in commit_to_version:
                self.fetcher.git('merge-base', '--is-ancestor', tag_commit, ref, ignore_errors=[1])
                if self.fetcher.git.returncode == 0:
                    distance = self.fetcher.git('rev-list', '%s..%s' % (tag_commit, ref), '--count', output=str, error=str).strip()
                    ancestor_commits.append((tag_commit, int(distance)))
            if ancestor_commits:
                (prev_version_commit, distance) = min(ancestor_commits, key=lambda x: x[1])
                prev_version = commit_to_version[prev_version_commit]
            else:
                ref_info = self.fetcher.git('log', '--all', '--pretty=format:%H', output=str)
                commits = [c for c in ref_info.split('\n') if c]
                prev_version = None
                distance = int(self.fetcher.git('rev-list', '%s..%s' % (commits[-1], ref), '--count', output=str, error=str).strip())
        return (prev_version, distance)