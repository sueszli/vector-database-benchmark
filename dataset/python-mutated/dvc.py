import errno
import functools
import ntpath
import os
import posixpath
import threading
from contextlib import ExitStack, suppress
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type, Union
from fsspec.spec import AbstractFileSystem
from funcy import wrap_with
from dvc.log import logger
from dvc_objects.fs.base import FileSystem
from dvc_objects.fs.path import Path
from .data import DataFileSystem
if TYPE_CHECKING:
    from dvc.repo import Repo
    from dvc.types import DictStrAny, StrPath
logger = logger.getChild(__name__)
RepoFactory = Union[Callable[..., 'Repo'], Type['Repo']]
Key = Tuple[str, ...]

def as_posix(path: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return path.replace(ntpath.sep, posixpath.sep)

def _is_dvc_file(fname):
    if False:
        for i in range(10):
            print('nop')
    from dvc.dvcfile import is_valid_filename
    from dvc.ignore import DvcIgnore
    return is_valid_filename(fname) or fname == DvcIgnore.DVCIGNORE_FILE

def _merge_info(repo, key, fs_info, dvc_info):
    if False:
        return 10
    from . import utils
    ret = {'repo': repo}
    if dvc_info:
        dvc_info['isout'] = any((len(out_key) <= len(key) and key[:len(out_key)] == out_key for out_key in repo.index.data_keys['repo']))
        dvc_info['isdvc'] = dvc_info['isout']
        ret['dvc_info'] = dvc_info
        ret['type'] = dvc_info['type']
        ret['size'] = dvc_info['size']
        if not fs_info and 'md5' in dvc_info:
            ret['md5'] = dvc_info['md5']
        if not fs_info and 'md5-dos2unix' in dvc_info:
            ret['md5-dos2unix'] = dvc_info['md5-dos2unix']
    if fs_info:
        ret['type'] = fs_info['type']
        ret['size'] = fs_info['size']
        isexec = False
        if fs_info['type'] == 'file':
            isexec = utils.is_exec(fs_info['mode'])
        ret['isexec'] = isexec
    return ret

def _get_dvc_path(dvc_fs, subkey):
    if False:
        while True:
            i = 10
    return dvc_fs.path.join(*subkey) if subkey else ''

class _DVCFileSystem(AbstractFileSystem):
    cachable = False
    root_marker = '/'

    def __init__(self, url: Optional[str]=None, rev: Optional[str]=None, repo: Optional['Repo']=None, subrepos: bool=False, repo_factory: Optional[RepoFactory]=None, fo: Optional[str]=None, target_options: Optional[Dict[str, Any]]=None, target_protocol: Optional[str]=None, config: Optional['DictStrAny']=None, remote: Optional[str]=None, remote_config: Optional['DictStrAny']=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        'DVC + git-tracked files fs.\n\n        Args:\n            path (str, optional): URL or path to a DVC/Git repository.\n                Defaults to a DVC repository in the current working directory.\n                Both HTTP and SSH protocols are supported for remote Git repos\n                (e.g. [user@]server:project.git).\n            rev (str, optional): Any Git revision such as a branch or tag name,\n                a commit hash or a dvc experiment name.\n                Defaults to the default branch in case of remote repositories.\n                In case of a local repository, if rev is unspecified, it will\n                default to the working directory.\n                If the repo is not a Git repo, this option is ignored.\n            repo (:obj:`Repo`, optional): `Repo` instance.\n            subrepos (bool): traverse to subrepos.\n                By default, it ignores subrepos.\n            repo_factory (callable): A function to initialize subrepo with.\n                The default is `Repo`.\n            config (dict): Repo config to be passed into `repo_factory`.\n            remote (str): Remote name to be passed into `repo_factory`.\n            remote_config(dict): Remote config to be passed into `repo_factory`.\n\n        Examples:\n            - Opening a filesystem from repo in current working directory\n\n            >>> fs = DVCFileSystem()\n\n            - Opening a filesystem from local repository\n\n            >>> fs = DVCFileSystem("path/to/local/repository")\n\n            - Opening a remote repository\n\n            >>> fs = DVCFileSystem(\n            ...    "https://github.com/iterative/example-get-started",\n            ...    rev="main",\n            ... )\n        '
        from pygtrie import Trie
        super().__init__()
        self._repo_stack = ExitStack()
        if repo is None:
            url = url if url is not None else fo
            repo = self._make_repo(url=url, rev=rev, subrepos=subrepos, config=config, remote=remote, remote_config=remote_config)
            assert repo is not None
            repo_factory = repo._fs_conf['repo_factory']
            self._repo_stack.enter_context(repo)
        if not repo_factory:
            from dvc.repo import Repo
            self.repo_factory: RepoFactory = Repo
        else:
            self.repo_factory = repo_factory

        def _getcwd():
            if False:
                for i in range(10):
                    print('nop')
            relparts: Tuple[str, ...] = ()
            assert repo is not None
            if repo.fs.path.isin(repo.fs.path.getcwd(), repo.root_dir):
                relparts = repo.fs.path.relparts(repo.fs.path.getcwd(), repo.root_dir)
            return self.root_marker + self.sep.join(relparts)
        self.path = Path(self.sep, getcwd=_getcwd)
        self.repo = repo
        self.hash_jobs = repo.fs.hash_jobs
        self._traverse_subrepos = subrepos
        self._subrepos_trie = Trie()
        'Keeps track of each and every path with the corresponding repo.'
        key = self._get_key(self.repo.root_dir)
        self._subrepos_trie[key] = repo
        self._datafss = {}
        'Keep a datafs instance of each repo.'
        if hasattr(repo, 'dvc_dir'):
            self._datafss[key] = DataFileSystem(index=repo.index.data['repo'])

    @functools.cached_property
    def fsid(self) -> str:
        if False:
            print('Hello World!')
        from fsspec.utils import tokenize
        from dvc.scm import NoSCM
        return 'dvcfs_' + tokenize(self.repo.url or self.repo.root_dir, self.repo.get_rev() if not isinstance(self.repo.scm, NoSCM) else None)

    def _get_key(self, path: 'StrPath') -> Key:
        if False:
            for i in range(10):
                print('nop')
        path = os.fspath(path)
        parts = self.repo.fs.path.relparts(path, self.repo.root_dir)
        if parts == (os.curdir,):
            return ()
        return parts

    def _get_key_from_relative(self, path) -> Key:
        if False:
            i = 10
            return i + 15
        path = self._strip_protocol(path)
        parts = self.path.relparts(path, self.root_marker)
        if parts and parts[0] == os.curdir:
            return parts[1:]
        return parts

    def _from_key(self, parts: Key) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.repo.fs.path.join(self.repo.root_dir, *parts)

    @property
    def repo_url(self):
        if False:
            print('Hello World!')
        return self.repo.url

    @classmethod
    def _make_repo(cls, **kwargs) -> 'Repo':
        if False:
            while True:
                i = 10
        from dvc.repo import Repo
        with Repo.open(uninitialized=True, **kwargs) as repo:
            return repo

    def _get_repo(self, key: Key) -> 'Repo':
        if False:
            return 10
        "Returns repo that the path falls in, using prefix.\n\n        If the path is already tracked/collected, it just returns the repo.\n\n        Otherwise, it collects the repos that might be in the path's parents\n        and then returns the appropriate one.\n        "
        repo = self._subrepos_trie.get(key)
        if repo:
            return repo
        (prefix_key, repo) = self._subrepos_trie.longest_prefix(key)
        dir_keys = (key[:i] for i in range(len(prefix_key) + 1, len(key) + 1))
        self._update(dir_keys, starting_repo=repo)
        return self._subrepos_trie.get(key) or self.repo

    @wrap_with(threading.Lock())
    def _update(self, dir_keys, starting_repo):
        if False:
            for i in range(10):
                print('nop')
        'Checks for subrepo in directories and updates them.'
        repo = starting_repo
        for key in dir_keys:
            d = self._from_key(key)
            if self._is_dvc_repo(d):
                repo = self.repo_factory(d, fs=self.repo.fs, scm=self.repo.scm, repo_factory=self.repo_factory)
                self._repo_stack.enter_context(repo)
                self._datafss[key] = DataFileSystem(index=repo.index.data['repo'])
            self._subrepos_trie[key] = repo

    def _is_dvc_repo(self, dir_path):
        if False:
            i = 10
            return i + 15
        'Check if the directory is a dvc repo.'
        if not self._traverse_subrepos:
            return False
        from dvc.repo import Repo
        repo_path = self.repo.fs.path.join(dir_path, Repo.DVC_DIR)
        return self.repo.fs.isdir(repo_path)

    def _get_subrepo_info(self, key: Key) -> Tuple['Repo', Optional[DataFileSystem], Key]:
        if False:
            return 10
        '\n        Returns information about the subrepo the key is part of.\n        '
        repo = self._get_repo(key)
        repo_key: Key
        if repo is self.repo:
            repo_key = ()
            subkey = key
        else:
            repo_key = self._get_key(repo.root_dir)
            subkey = key[len(repo_key):]
        dvc_fs = self._datafss.get(repo_key)
        return (repo, dvc_fs, subkey)

    def _open(self, path, mode='rb', **kwargs):
        if False:
            i = 10
            return i + 15
        if mode != 'rb':
            raise OSError(errno.EROFS, os.strerror(errno.EROFS))
        key = self._get_key_from_relative(path)
        fs_path = self._from_key(key)
        try:
            return self.repo.fs.open(fs_path, mode=mode)
        except FileNotFoundError:
            (_, dvc_fs, subkey) = self._get_subrepo_info(key)
            if not dvc_fs:
                raise
        dvc_path = _get_dvc_path(dvc_fs, subkey)
        return dvc_fs.open(dvc_path, mode=mode, cache=kwargs.get('cache', False))

    def isdvc(self, path, **kwargs) -> bool:
        if False:
            print('Hello World!')
        'Is this entry dvc-tracked?'
        try:
            return self.info(path).get('dvc_info', {}).get('isout', False)
        except FileNotFoundError:
            return False

    def ls(self, path, detail=True, dvc_only=False, **kwargs):
        if False:
            i = 10
            return i + 15
        key = self._get_key_from_relative(path)
        (repo, dvc_fs, subkey) = self._get_subrepo_info(key)
        dvc_exists = False
        dvc_infos = {}
        if dvc_fs:
            dvc_path = _get_dvc_path(dvc_fs, subkey)
            with suppress(FileNotFoundError, NotADirectoryError):
                for info in dvc_fs.ls(dvc_path, detail=True):
                    dvc_infos[dvc_fs.path.name(info['name'])] = info
                dvc_exists = True
        fs_exists = False
        fs_infos = {}
        ignore_subrepos = kwargs.get('ignore_subrepos', True)
        if not dvc_only:
            fs = self.repo.fs
            fs_path = self._from_key(key)
            try:
                for info in repo.dvcignore.ls(fs, fs_path, detail=True, ignore_subrepos=ignore_subrepos):
                    fs_infos[fs.path.name(info['name'])] = info
                fs_exists = True
            except (FileNotFoundError, NotADirectoryError):
                pass
        if not (dvc_exists or fs_exists):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        dvcfiles = kwargs.get('dvcfiles', False)
        infos = []
        paths = []
        names = set(dvc_infos.keys()) | set(fs_infos.keys())
        for name in names:
            if not dvcfiles and _is_dvc_file(name):
                continue
            entry_path = self.path.join(path, name)
            info = _merge_info(repo, (*subkey, name), fs_infos.get(name), dvc_infos.get(name))
            info['name'] = entry_path
            infos.append(info)
            paths.append(entry_path)
        if not detail:
            return paths
        return infos

    def info(self, path, **kwargs):
        if False:
            while True:
                i = 10
        key = self._get_key_from_relative(path)
        ignore_subrepos = kwargs.get('ignore_subrepos', True)
        return self._info(key, path, ignore_subrepos=ignore_subrepos)

    def _info(self, key, path, ignore_subrepos=True, check_ignored=True):
        if False:
            for i in range(10):
                print('nop')
        (repo, dvc_fs, subkey) = self._get_subrepo_info(key)
        dvc_info = None
        if dvc_fs:
            try:
                dvc_info = dvc_fs.fs.index.info(subkey)
                dvc_path = _get_dvc_path(dvc_fs, subkey)
                dvc_info['name'] = dvc_path
            except KeyError:
                pass
        fs_info = None
        fs = self.repo.fs
        fs_path = self._from_key(key)
        try:
            fs_info = fs.info(fs_path)
            if check_ignored and repo.dvcignore.is_ignored(fs, fs_path, ignore_subrepos=ignore_subrepos):
                fs_info = None
        except (FileNotFoundError, NotADirectoryError):
            if not dvc_info:
                raise
        if dvc_info and (not fs_info):
            for parent in fs.path.parents(fs_path):
                try:
                    if fs.info(parent)['type'] != 'directory':
                        dvc_info = None
                        break
                except FileNotFoundError:
                    continue
        if not dvc_info and (not fs_info):
            raise FileNotFoundError
        info = _merge_info(repo, subkey, fs_info, dvc_info)
        info['name'] = path
        return info

    def get_file(self, rpath, lpath, **kwargs):
        if False:
            while True:
                i = 10
        key = self._get_key_from_relative(rpath)
        fs_path = self._from_key(key)
        try:
            return self.repo.fs.get_file(fs_path, lpath, **kwargs)
        except FileNotFoundError:
            (_, dvc_fs, subkey) = self._get_subrepo_info(key)
            if not dvc_fs:
                raise
        dvc_path = _get_dvc_path(dvc_fs, subkey)
        return dvc_fs.get_file(dvc_path, lpath, **kwargs)

    def close(self):
        if False:
            while True:
                i = 10
        self._repo_stack.close()

class DVCFileSystem(FileSystem):
    protocol = 'local'
    PARAM_CHECKSUM = 'md5'

    def _prepare_credentials(self, **config) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return config

    @functools.cached_property
    def fs(self) -> '_DVCFileSystem':
        if False:
            for i in range(10):
                print('nop')
        return _DVCFileSystem(**self.fs_args)

    @property
    def fsid(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.fs.fsid

    def isdvc(self, path, **kwargs) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.fs.isdvc(path, **kwargs)

    @property
    def path(self) -> Path:
        if False:
            print('Hello World!')
        return self.fs.path

    @property
    def repo(self) -> 'Repo':
        if False:
            print('Hello World!')
        return self.fs.repo

    @property
    def repo_url(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.fs.repo_url

    def from_os_path(self, path: str) -> str:
        if False:
            i = 10
            return i + 15
        if os.path.isabs(path):
            path = os.path.relpath(path, self.repo.root_dir)
        return as_posix(path)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        if 'fs' in self.__dict__:
            self.fs.close()