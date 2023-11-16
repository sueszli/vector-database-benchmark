""" PEP 610 """
import json
import re
import urllib.parse
from typing import Any, Dict, Iterable, Optional, Type, TypeVar, Union
__all__ = ['DirectUrl', 'DirectUrlValidationError', 'DirInfo', 'ArchiveInfo', 'VcsInfo']
T = TypeVar('T')
DIRECT_URL_METADATA_NAME = 'direct_url.json'
ENV_VAR_RE = re.compile('^\\$\\{[A-Za-z0-9-_]+\\}(:\\$\\{[A-Za-z0-9-_]+\\})?$')

class DirectUrlValidationError(Exception):
    pass

def _get(d: Dict[str, Any], expected_type: Type[T], key: str, default: Optional[T]=None) -> Optional[T]:
    if False:
        while True:
            i = 10
    'Get value from dictionary and verify expected type.'
    if key not in d:
        return default
    value = d[key]
    if not isinstance(value, expected_type):
        raise DirectUrlValidationError(f'{value!r} has unexpected type for {key} (expected {expected_type})')
    return value

def _get_required(d: Dict[str, Any], expected_type: Type[T], key: str, default: Optional[T]=None) -> T:
    if False:
        while True:
            i = 10
    value = _get(d, expected_type, key, default)
    if value is None:
        raise DirectUrlValidationError(f'{key} must have a value')
    return value

def _exactly_one_of(infos: Iterable[Optional['InfoType']]) -> 'InfoType':
    if False:
        print('Hello World!')
    infos = [info for info in infos if info is not None]
    if not infos:
        raise DirectUrlValidationError('missing one of archive_info, dir_info, vcs_info')
    if len(infos) > 1:
        raise DirectUrlValidationError('more than one of archive_info, dir_info, vcs_info')
    assert infos[0] is not None
    return infos[0]

def _filter_none(**kwargs: Any) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    'Make dict excluding None values.'
    return {k: v for (k, v) in kwargs.items() if v is not None}

class VcsInfo:
    name = 'vcs_info'

    def __init__(self, vcs: str, commit_id: str, requested_revision: Optional[str]=None) -> None:
        if False:
            return 10
        self.vcs = vcs
        self.requested_revision = requested_revision
        self.commit_id = commit_id

    @classmethod
    def _from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional['VcsInfo']:
        if False:
            i = 10
            return i + 15
        if d is None:
            return None
        return cls(vcs=_get_required(d, str, 'vcs'), commit_id=_get_required(d, str, 'commit_id'), requested_revision=_get(d, str, 'requested_revision'))

    def _to_dict(self) -> Dict[str, Any]:
        if False:
            return 10
        return _filter_none(vcs=self.vcs, requested_revision=self.requested_revision, commit_id=self.commit_id)

class ArchiveInfo:
    name = 'archive_info'

    def __init__(self, hash: Optional[str]=None, hashes: Optional[Dict[str, str]]=None) -> None:
        if False:
            while True:
                i = 10
        self.hashes = hashes
        self.hash = hash

    @property
    def hash(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._hash

    @hash.setter
    def hash(self, value: Optional[str]) -> None:
        if False:
            while True:
                i = 10
        if value is not None:
            try:
                (hash_name, hash_value) = value.split('=', 1)
            except ValueError:
                raise DirectUrlValidationError(f'invalid archive_info.hash format: {value!r}')
            if self.hashes is None:
                self.hashes = {hash_name: hash_value}
            elif hash_name not in self.hashes:
                self.hashes = self.hashes.copy()
                self.hashes[hash_name] = hash_value
        self._hash = value

    @classmethod
    def _from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional['ArchiveInfo']:
        if False:
            i = 10
            return i + 15
        if d is None:
            return None
        return cls(hash=_get(d, str, 'hash'), hashes=_get(d, dict, 'hashes'))

    def _to_dict(self) -> Dict[str, Any]:
        if False:
            return 10
        return _filter_none(hash=self.hash, hashes=self.hashes)

class DirInfo:
    name = 'dir_info'

    def __init__(self, editable: bool=False) -> None:
        if False:
            print('Hello World!')
        self.editable = editable

    @classmethod
    def _from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional['DirInfo']:
        if False:
            i = 10
            return i + 15
        if d is None:
            return None
        return cls(editable=_get_required(d, bool, 'editable', default=False))

    def _to_dict(self) -> Dict[str, Any]:
        if False:
            return 10
        return _filter_none(editable=self.editable or None)
InfoType = Union[ArchiveInfo, DirInfo, VcsInfo]

class DirectUrl:

    def __init__(self, url: str, info: InfoType, subdirectory: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        self.url = url
        self.info = info
        self.subdirectory = subdirectory

    def _remove_auth_from_netloc(self, netloc: str) -> str:
        if False:
            return 10
        if '@' not in netloc:
            return netloc
        (user_pass, netloc_no_user_pass) = netloc.split('@', 1)
        if isinstance(self.info, VcsInfo) and self.info.vcs == 'git' and (user_pass == 'git'):
            return netloc
        if ENV_VAR_RE.match(user_pass):
            return netloc
        return netloc_no_user_pass

    @property
    def redacted_url(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'url with user:password part removed unless it is formed with\n        environment variables as specified in PEP 610, or it is ``git``\n        in the case of a git URL.\n        '
        purl = urllib.parse.urlsplit(self.url)
        netloc = self._remove_auth_from_netloc(purl.netloc)
        surl = urllib.parse.urlunsplit((purl.scheme, netloc, purl.path, purl.query, purl.fragment))
        return surl

    def validate(self) -> None:
        if False:
            return 10
        self.from_dict(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DirectUrl':
        if False:
            i = 10
            return i + 15
        return DirectUrl(url=_get_required(d, str, 'url'), subdirectory=_get(d, str, 'subdirectory'), info=_exactly_one_of([ArchiveInfo._from_dict(_get(d, dict, 'archive_info')), DirInfo._from_dict(_get(d, dict, 'dir_info')), VcsInfo._from_dict(_get(d, dict, 'vcs_info'))]))

    def to_dict(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        res = _filter_none(url=self.redacted_url, subdirectory=self.subdirectory)
        res[self.info.name] = self.info._to_dict()
        return res

    @classmethod
    def from_json(cls, s: str) -> 'DirectUrl':
        if False:
            for i in range(10):
                print('nop')
        return cls.from_dict(json.loads(s))

    def to_json(self) -> str:
        if False:
            print('Hello World!')
        return json.dumps(self.to_dict(), sort_keys=True)

    def is_local_editable(self) -> bool:
        if False:
            return 10
        return isinstance(self.info, DirInfo) and self.info.editable