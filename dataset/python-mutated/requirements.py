from __future__ import annotations
import dataclasses
import functools
import inspect
import json
import os
import posixpath
import re
import secrets
import urllib.parse as urlparse
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, TypeVar, cast
from packaging.markers import InvalidMarker
from packaging.requirements import InvalidRequirement
from packaging.requirements import Requirement as PackageRequirement
from packaging.specifiers import SpecifierSet
from packaging.utils import parse_sdist_filename, parse_wheel_filename
from pdm.compat import Distribution
from pdm.exceptions import ExtrasWarning, RequirementError
from pdm.models.backends import BuildBackend, get_relative_path
from pdm.models.markers import Marker, get_marker, split_marker_extras
from pdm.models.setup import Setup
from pdm.models.specifiers import PySpecSet, fix_legacy_specifier, get_specifier
from pdm.utils import PACKAGING_22, add_ssh_scheme_to_git_uri, comparable_version, normalize_name, path_to_url, path_without_fragments, url_to_path, url_without_fragments
if TYPE_CHECKING:
    from unearth import Link
    from pdm._types import RequirementDict
VCS_SCHEMA = ('git', 'hg', 'svn', 'bzr')
_vcs_req_re = re.compile(f"(?P<url>(?P<vcs>{'|'.join(VCS_SCHEMA)})\\+[^\\s;]+)(?P<marker>[\\t ]*;[^\\n]+)?", flags=re.IGNORECASE)
_file_req_re = re.compile('(?:(?P<url>\\S+://[^\\s\\[\\];]+)|(?P<path>(?:[^\\s;\\[\\]]|\\\\ )*|\'(?:[^\']|\\\\\')*\'|\\"(?:[^\\"]|\\\\\\")*\\"))(?P<extras>\\[[^\\[\\]]+\\])?(?P<marker>[\\t ]*;[^\\n]+)?')
_egg_info_re = re.compile('([a-z0-9_.]+)-([a-z0-9_.!+-]+)', re.IGNORECASE)
T = TypeVar('T', bound='Requirement')

def strip_extras(line: str) -> tuple[str, tuple[str, ...] | None]:
    if False:
        for i in range(10):
            print('nop')
    match = re.match('^(.+?)(?:\\[([^\\]]+)\\])?$', line)
    assert match is not None
    (name, extras_str) = match.groups()
    extras = tuple({e.strip() for e in extras_str.split(',')}) if extras_str else None
    return (name, extras)

@functools.lru_cache(maxsize=None)
def _get_random_key(req: Requirement) -> str:
    if False:
        return 10
    return f':empty:{secrets.token_urlsafe(8)}'

@dataclasses.dataclass(eq=False)
class Requirement:
    """Base class of a package requirement.
    A requirement is a (virtual) specification of a package which contains
    some constraints of version, python version, or other marker.
    """
    name: str | None = None
    marker: Marker | None = None
    extras: Sequence[str] | None = None
    specifier: SpecifierSet | None = None
    editable: bool = False
    prerelease: bool = False

    def __post_init__(self) -> None:
        if False:
            print('Hello World!')
        self.requires_python = self.marker.split_pyspec()[1] if self.marker else PySpecSet()

    @property
    def project_name(self) -> str | None:
        if False:
            i = 10
            return i + 15
        return normalize_name(self.name, lowercase=False) if self.name else None

    @property
    def key(self) -> str | None:
        if False:
            print('Hello World!')
        return self.project_name.lower() if self.project_name else None

    @property
    def is_pinned(self) -> bool:
        if False:
            while True:
                i = 10
        if not self.specifier:
            return False
        if len(self.specifier) != 1:
            return False
        sp = next(iter(self.specifier))
        return sp.operator == '===' or (sp.operator == '==' and '*' not in sp.version)

    def as_pinned_version(self: T, other_version: str | None) -> T:
        if False:
            i = 10
            return i + 15
        'Return a new requirement with the given pinned version.'
        if self.is_pinned or not other_version:
            return self
        normalized = comparable_version(other_version)
        return dataclasses.replace(self, specifier=get_specifier(f'=={normalized}'))

    def _hash_key(self) -> tuple:
        if False:
            while True:
                i = 10
        return (self.key, frozenset(self.extras) if self.extras else None, str(self.marker) if self.marker else None)

    def __hash__(self) -> int:
        if False:
            return 10
        return hash(self._hash_key())

    def __eq__(self, o: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return isinstance(o, Requirement) and self._hash_key() == o._hash_key()

    def identify(self) -> str:
        if False:
            i = 10
            return i + 15
        if not self.key:
            return _get_random_key(self)
        extras = '[{}]'.format(','.join(sorted(self.extras))) if self.extras else ''
        return self.key + extras

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'<{self.__class__.__name__} {self.as_line()}>'

    def __str__(self) -> str:
        if False:
            return 10
        return self.as_line()

    @classmethod
    def create(cls: type[T], **kwargs: Any) -> T:
        if False:
            print('Hello World!')
        if 'marker' in kwargs:
            try:
                kwargs['marker'] = get_marker(kwargs['marker'])
            except InvalidMarker as e:
                raise RequirementError('Invalid marker: %s' % str(e)) from None
        if 'extras' in kwargs and isinstance(kwargs['extras'], str):
            kwargs['extras'] = tuple((e.strip() for e in kwargs['extras'][1:-1].split(',')))
        version = kwargs.pop('version', '')
        kwargs['specifier'] = get_specifier(version)
        return cls(**{k: v for (k, v) in kwargs.items() if k in inspect.signature(cls).parameters})

    @classmethod
    def from_dist(cls, dist: Distribution) -> Requirement:
        if False:
            while True:
                i = 10
        direct_url_json = dist.read_text('direct_url.json')
        if direct_url_json is not None:
            direct_url = json.loads(direct_url_json)
            data = {'name': dist.metadata['Name'], 'url': direct_url.get('url'), 'editable': direct_url.get('dir_info', {}).get('editable'), 'subdirectory': direct_url.get('subdirectory')}
            if 'vcs_info' in direct_url:
                vcs_info = direct_url['vcs_info']
                data.update(url=f"{vcs_info['vcs']}+{direct_url['url']}", ref=vcs_info.get('requested_revision'), revision=vcs_info.get('commit_id'))
                return VcsRequirement.create(**data)
            return FileRequirement.create(**data)
        return NamedRequirement.create(name=dist.metadata['Name'], version=f'=={dist.version}')

    @classmethod
    def from_req_dict(cls, name: str, req_dict: RequirementDict) -> Requirement:
        if False:
            while True:
                i = 10
        if isinstance(req_dict, str):
            return NamedRequirement(name=name, specifier=get_specifier(req_dict))
        for vcs in VCS_SCHEMA:
            if vcs in req_dict:
                repo = cast(str, req_dict.pop(vcs, None))
                url = f'{vcs}+{repo}'
                return VcsRequirement.create(name=name, vcs=vcs, url=url, **req_dict)
        if 'path' in req_dict or 'url' in req_dict:
            return FileRequirement.create(name=name, **req_dict)
        return NamedRequirement.create(name=name, **req_dict)

    @property
    def is_named(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return isinstance(self, NamedRequirement)

    @property
    def is_vcs(self) -> bool:
        if False:
            print('Hello World!')
        return isinstance(self, VcsRequirement)

    @property
    def is_file_or_url(self) -> bool:
        if False:
            print('Hello World!')
        return type(self) is FileRequirement

    def as_line(self) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def matches(self, line: str) -> bool:
        if False:
            return 10
        'Return whether the passed in PEP 508 string\n        is the same requirement as this one.\n        '
        if line.strip().startswith('-e '):
            req = parse_requirement(line.split('-e ', 1)[-1], True)
        else:
            req = parse_requirement(line, False)
        return self.key == req.key

    @classmethod
    def from_pkg_requirement(cls, req: PackageRequirement) -> Requirement:
        if False:
            return 10
        from unearth import Link
        kwargs = {'name': req.name, 'extras': req.extras, 'specifier': req.specifier, 'marker': get_marker(req.marker)}
        if getattr(req, 'url', None):
            link = Link(cast(str, req.url))
            klass = VcsRequirement if link.is_vcs else FileRequirement
            return klass(url=req.url, **kwargs)
        else:
            return NamedRequirement(**kwargs)

    def _format_marker(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        if self.marker:
            return f'; {self.marker!s}'
        return ''

@dataclasses.dataclass(eq=False)
class NamedRequirement(Requirement):

    def as_line(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        extras = f"[{','.join(sorted(self.extras))}]" if self.extras else ''
        return f"{self.project_name}{extras}{self.specifier or ''}{self._format_marker()}"

@dataclasses.dataclass(eq=False)
class FileRequirement(Requirement):
    url: str = ''
    path: Path | None = None
    subdirectory: str | None = None

    def __post_init__(self) -> None:
        if False:
            return 10
        super().__post_init__()
        self._parse_url()
        if self.is_local_dir:
            self._check_installable()

    def _hash_key(self) -> tuple:
        if False:
            i = 10
            return i + 15
        return (*super()._hash_key(), self.get_full_url(), self.editable)

    def guess_name(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        filename = os.path.basename(urlparse.unquote(url_without_fragments(self.url))).rsplit('@', 1)[0]
        if self.is_vcs:
            if self.vcs == 'git':
                name = filename
                if name.endswith('.git'):
                    name = name[:-4]
                return name
            elif self.vcs == 'hg':
                return filename
            else:
                (name, in_branch, _) = filename.rpartition('/branches/')
                if not in_branch and name.endswith('/trunk'):
                    return name[:-6]
                return name
        elif filename.endswith('.whl'):
            return parse_wheel_filename(filename)[0]
        else:
            try:
                return parse_sdist_filename(filename)[0]
            except ValueError:
                match = _egg_info_re.match(filename)
                if match:
                    return match.group(1)
        return None

    @classmethod
    def create(cls: type[T], **kwargs: Any) -> T:
        if False:
            print('Hello World!')
        if kwargs.get('path'):
            kwargs['path'] = Path(kwargs['path'])
        return super().create(**kwargs)

    @property
    def str_path(self) -> str | None:
        if False:
            while True:
                i = 10
        if not self.path:
            return None
        if self.path.is_absolute():
            try:
                result = self.path.relative_to(Path.cwd()).as_posix()
            except ValueError:
                return self.path.as_posix()
        else:
            result = self.path.as_posix()
        result = posixpath.normpath(result)
        if not result.startswith(('./', '../')):
            result = './' + result
        if result.startswith('./../'):
            result = result[2:]
        return result

    def _parse_url(self) -> None:
        if False:
            i = 10
            return i + 15
        if not self.url and self.path and self.path.is_absolute():
            self.url = path_to_url(self.path.as_posix())
        if not self.path:
            path = get_relative_path(self.url)
            if path is None:
                try:
                    self.path = path_without_fragments(url_to_path(self.url))
                except AssertionError:
                    pass
            else:
                self.path = path_without_fragments(path)
        if self.url:
            self._parse_name_from_url()

    def relocate(self, backend: BuildBackend) -> None:
        if False:
            return 10
        'Change the project root to the given path'
        if self.path is None or self.path.is_absolute():
            return
        self.path = path_without_fragments(os.path.relpath(self.path, backend.root))
        path = self.path.as_posix()
        if path == '.':
            path = ''
        self.url = backend.relative_path_to_url(path)

    @property
    def is_local(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.path and self.path.exists() or False

    @property
    def is_local_dir(self) -> bool:
        if False:
            while True:
                i = 10
        return self.is_local and cast(Path, self.path).is_dir()

    def as_file_link(self) -> Link:
        if False:
            return 10
        from unearth import Link
        url = self.get_full_url()
        if self.subdirectory:
            url += f'#subdirectory={self.subdirectory}'
        return Link(url)

    def get_full_url(self) -> str:
        if False:
            return 10
        return url_without_fragments(self.url)

    def as_line(self) -> str:
        if False:
            print('Hello World!')
        project_name = f'{self.project_name}' if self.project_name else ''
        extras = f"[{','.join(sorted(self.extras))}]" if self.extras and self.project_name else ''
        marker = self._format_marker()
        if marker:
            marker = f' {marker}'
        url = self.get_full_url()
        fragments = []
        if self.subdirectory:
            fragments.append(f'subdirectory={self.subdirectory}')
        if self.editable:
            if project_name:
                fragments.insert(0, f'egg={project_name}{extras}')
            fragment_str = '#' + '&'.join(fragments) if fragments else ''
            return f'-e {url}{fragment_str}{marker}'
        delimiter = ' @ ' if project_name else ''
        fragment_str = '#' + '&'.join(fragments) if fragments else ''
        return f'{project_name}{extras}{delimiter}{url}{fragment_str}{marker}'

    def _parse_name_from_url(self) -> None:
        if False:
            print('Hello World!')
        parsed = urlparse.urlparse(self.url)
        fragments = dict(urlparse.parse_qsl(parsed.fragment))
        if 'egg' in fragments:
            egg_info = urlparse.unquote(fragments['egg'])
            (name, extras) = strip_extras(egg_info)
            self.name = name
            if not self.extras:
                self.extras = extras
        if not self.name and (not self.is_vcs):
            self.name = self.guess_name()

    def _check_installable(self) -> None:
        if False:
            print('Hello World!')
        assert self.path
        if not (self.path.joinpath('setup.py').exists() or self.path.joinpath('pyproject.toml').exists()):
            raise RequirementError(f"The local path '{self.path}' is not installable.")
        result = Setup.from_directory(self.path.absolute())
        if result.name:
            self.name = result.name

@dataclasses.dataclass(eq=False)
class VcsRequirement(FileRequirement):
    vcs: str = ''
    ref: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__post_init__()
        if not self.vcs:
            self.vcs = self.url.split('+', 1)[0]

    def get_full_url(self) -> str:
        if False:
            print('Hello World!')
        url = super().get_full_url()
        if self.revision and (not self.editable):
            url += f'@{self.revision}'
        elif self.ref:
            url += f'@{self.ref}'
        return url

    def _parse_url(self) -> None:
        if False:
            while True:
                i = 10
        (vcs, url_no_vcs) = self.url.split('+', 1)
        if url_no_vcs.startswith('git@'):
            url_no_vcs = add_ssh_scheme_to_git_uri(url_no_vcs)
        if not self.name:
            self._parse_name_from_url()
        ref = self.ref
        parsed = urlparse.urlparse(url_no_vcs)
        path = parsed.path
        fragments = dict(urlparse.parse_qsl(parsed.fragment))
        if 'subdirectory' in fragments:
            self.subdirectory = fragments['subdirectory']
        if '@' in parsed.path:
            (path, ref) = parsed.path.split('@', 1)
        repo = urlparse.urlunparse(parsed._replace(path=path, fragment=''))
        self.url = f'{vcs}+{repo}'
        (self.repo, self.ref) = (repo, ref)

def filter_requirements_with_extras(project_name: str, requirement_lines: list[str], extras: Sequence[str], include_default: bool=False) -> list[str]:
    if False:
        return 10
    'Filter the requirements with extras.\n    If extras are given, return those with matching extra markers.\n    Otherwise, return those without extra markers.\n    '
    extras = [normalize_name(e) for e in extras]
    result: list[str] = []
    extras_in_meta: set[str] = set()
    for req in requirement_lines:
        _r = parse_requirement(req)
        if _r.marker:
            (req_extras, rest) = split_marker_extras(str(_r.marker))
            if req_extras:
                extras_in_meta.update(req_extras)
                _r.marker = Marker(rest) if rest else None
        else:
            req_extras = set()
        if req_extras and (not req_extras.isdisjoint(extras)) or (not req_extras and (include_default or not extras)):
            result.append(_r.as_line())
    extras_not_found = [e for e in extras if e not in extras_in_meta]
    if extras_not_found:
        warnings.warn(ExtrasWarning(project_name, extras_not_found), stacklevel=2)
    return result

def parse_as_pkg_requirement(line: str) -> PackageRequirement:
    if False:
        return 10
    'Parse a requirement line as packaging.requirement.Requirement'
    try:
        return PackageRequirement(line)
    except InvalidRequirement:
        if not PACKAGING_22:
            raise
        new_line = fix_legacy_specifier(line)
        return PackageRequirement(new_line)

def parse_requirement(line: str, editable: bool=False) -> Requirement:
    if False:
        while True:
            i = 10
    m = _vcs_req_re.match(line)
    r: Requirement
    if m is not None:
        r = VcsRequirement.create(**m.groupdict())
    else:
        root_url = path_to_url(Path().as_posix())
        replaced = '{root:uri}' in line
        if replaced:
            line = line.replace('{root:uri}', root_url)
        try:
            pkg_req = parse_as_pkg_requirement(line)
        except InvalidRequirement as e:
            m = _file_req_re.match(line)
            if m is None:
                raise RequirementError(str(e)) from None
            args = m.groupdict()
            if not line.startswith('.') and (not args['url']) and args['path'] and (not os.path.exists(args['path'])):
                raise RequirementError(str(e)) from None
            r = FileRequirement.create(**args)
        else:
            r = Requirement.from_pkg_requirement(pkg_req)
        if replaced:
            assert isinstance(r, FileRequirement)
            r.url = r.url.replace(root_url, '{root:uri}')
            r.path = Path(get_relative_path(r.url) or '')
    if editable:
        if r.is_vcs or (r.is_file_or_url and r.is_local_dir):
            assert isinstance(r, FileRequirement)
            r.editable = True
        else:
            raise RequirementError('Editable requirement is only supported for VCS link or local directory.')
    return r