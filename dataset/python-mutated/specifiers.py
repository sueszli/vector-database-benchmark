from __future__ import annotations
import json
import re
from functools import lru_cache
from operator import attrgetter
from typing import Any, Iterable, Match, cast
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from pdm.exceptions import InvalidPyVersion
from pdm.models.versions import Version

def _read_max_versions() -> dict[Version, int]:
    if False:
        return 10
    from pdm.compat import resources_open_binary
    with resources_open_binary('pdm.models', 'python_max_versions.json') as fp:
        return {Version(k): v for (k, v) in json.load(fp).items()}

@lru_cache()
def get_specifier(version_str: SpecifierSet | str) -> SpecifierSet:
    if False:
        while True:
            i = 10
    if isinstance(version_str, SpecifierSet):
        return version_str
    if not version_str or version_str == '*':
        return SpecifierSet()
    return SpecifierSet(version_str)
_legacy_specifier_re = re.compile('(==|!=|<=|>=|<|>)(\\s*)([^,;\\s)]*)')

@lru_cache()
def fix_legacy_specifier(specifier: str) -> str:
    if False:
        return 10
    "Since packaging 22.0, legacy specifiers like '>=4.*' are no longer\n    supported. We try to normalize them to the new format.\n    "
    from pdm.utils import deprecation_warning

    def fix_wildcard(match: Match[str]) -> str:
        if False:
            while True:
                i = 10
        (operator, _, version) = match.groups()
        if operator in ('==', '!='):
            return match.group(0)
        if '.*' in version:
            deprecation_warning('.* suffix can only be used with `==` or `!=` operators', stacklevel=4)
            version = version.replace('.*', '.0')
            if operator in ('<', '<='):
                operator = '<'
            elif operator in ('>', '>='):
                operator = '>='
        elif '+' in version:
            deprecation_warning('Local version label can only be used with `==` or `!=` operators', stacklevel=4)
            version = version.split('+')[0]
        return f'{operator}{version}'
    return _legacy_specifier_re.sub(fix_wildcard, specifier)

def _normalize_op_specifier(op: str, version_str: str) -> tuple[str, Version]:
    if False:
        return 10
    version = Version(version_str)
    if version.is_wildcard:
        if op == '==':
            op = '~='
            version[-1] = 0
        elif op == '>':
            op = '>='
            version = version.bump(-2)
        elif op in ('<', '>=', '<='):
            version[-1] = 0
            if op == '<=':
                op = '<'
        elif op != '!=':
            raise InvalidPyVersion(f'Unsupported version specifier: {op}{version}')
    if op != '~=' and (not (op == '!=' and version.is_wildcard)):
        version = version.complete()
    return (op, version)

class PySpecSet(SpecifierSet):
    """A custom SpecifierSet that supports merging with logic operators (&, |)."""
    PY_MAX_MINOR_VERSION = _read_max_versions()
    MAX_MAJOR_VERSION = max(PY_MAX_MINOR_VERSION)[:1].bump()

    def __init__(self, specifiers: str='', analyze: bool=True) -> None:
        if False:
            print('Hello World!')
        if specifiers == '*':
            specifiers = ''
        try:
            super().__init__(fix_legacy_specifier(specifiers))
        except InvalidSpecifier as e:
            raise InvalidPyVersion(str(e)) from e
        self._lower_bound = Version.MIN
        self._upper_bound = Version.MAX
        self._excludes: list[Version] = []
        if specifiers and analyze:
            self._analyze_specifiers()

    def _analyze_specifiers(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (lower_bound, upper_bound) = (Version.MIN, Version.MAX)
        excludes: set[Version] = set()
        for spec in self:
            (op, version) = _normalize_op_specifier(spec.operator, spec.version)
            if op in ('==', '==='):
                lower_bound = version
                upper_bound = version.bump()
                break
            if op == '!=':
                excludes.add(version)
            elif op[0] == '>':
                lower_bound = max(lower_bound, version if op == '>=' else version.bump())
            elif op[0] == '<':
                upper_bound = min(upper_bound, version.bump() if op == '<=' else version)
            elif op == '~=':
                new_lower = version.complete()
                new_upper = version.bump(-2)
                if new_upper < upper_bound:
                    upper_bound = new_upper
                if new_lower > lower_bound:
                    lower_bound = new_lower
            else:
                raise InvalidPyVersion(f'Unsupported version specifier: {op}{version}')
        self._rearrange(lower_bound, upper_bound, excludes)

    @classmethod
    def _merge_bounds_and_excludes(cls, lower: Version, upper: Version, excludes: Iterable[Version]) -> tuple[Version, Version, list[Version]]:
        if False:
            for i in range(10):
                print('nop')
        sorted_excludes = sorted(excludes)
        wildcard_excludes = {version[:-1] for version in sorted_excludes if version.is_wildcard}
        sorted_excludes = [version for version in sorted_excludes if version.is_wildcard or not any((version.startswith(wv) for wv in wildcard_excludes))]
        if lower == Version.MIN and upper == Version.MAX:
            return (lower, upper, sorted_excludes)
        for version in list(sorted_excludes):
            if version >= upper:
                sorted_excludes[:] = []
                break
            if version.is_wildcard:
                valid_length = len(version._version) - 1
                valid_version = version[:valid_length]
                if valid_version < lower[:valid_length]:
                    sorted_excludes.remove(version)
                elif lower.startswith(valid_version):
                    lower = version.bump(-2)
                    sorted_excludes.remove(version)
                else:
                    break
            elif version < lower:
                sorted_excludes.remove(version)
            elif version == lower:
                lower = version.bump()
                sorted_excludes.remove(version)
            else:
                break
        for version in reversed(sorted_excludes):
            if version >= upper:
                sorted_excludes.remove(version)
                continue
            if not version.is_wildcard:
                break
            valid_length = len(version._version) - 1
            valid_version = version[:valid_length]
            if upper.startswith(valid_version) or version.bump(-2) == upper:
                upper = valid_version.complete()
                sorted_excludes.remove(version)
            else:
                break
        return (lower, upper, sorted_excludes)

    def _rearrange(self, lower_bound: Version, upper_bound: Version, excludes: Iterable[Version]) -> None:
        if False:
            return 10
        'Rearrange the version bounds with the given inputs.'
        (self._lower_bound, self._upper_bound, self._excludes) = self._merge_bounds_and_excludes(lower_bound, upper_bound, excludes)
        if not self.is_impossible:
            super().__init__(str(self))

    def _comp_key(self) -> tuple[Version, Version, tuple[Version, ...]]:
        if False:
            while True:
                i = 10
        return (self._lower_bound, self._upper_bound, tuple(self._excludes))

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        return hash(self._comp_key())

    def __eq__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        if not isinstance(other, PySpecSet):
            return False
        return self._comp_key() == other._comp_key()

    @property
    def is_impossible(self) -> bool:
        if False:
            return 10
        'Check whether the specifierset contains any valid versions.'
        if self._lower_bound == Version.MIN or self._upper_bound == Version.MAX:
            return False
        return self._lower_bound >= self._upper_bound

    @property
    def is_allow_all(self) -> bool:
        if False:
            print('Hello World!')
        'Return True if the specifierset accepts all versions.'
        if self.is_impossible:
            return False
        return self._lower_bound == Version.MIN and self._upper_bound == Version.MAX and (not self._excludes)

    def __bool__(self) -> bool:
        if False:
            while True:
                i = 10
        return not self.is_allow_all

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        if self.is_impossible:
            return 'impossible'
        if self.is_allow_all:
            return ''
        lower = self._lower_bound
        upper = self._upper_bound
        if lower[-1] == 0 and (not lower.is_prerelease):
            lower = lower[:-1]
        if upper[-1] == 0 and (not upper.is_prerelease):
            upper = upper[:-1]
        lower_str = '' if lower == Version.MIN else f'>={lower}'
        upper_str = '' if upper == Version.MAX else f'<{upper}'
        excludes_str = ','.join((f'!={version}' for version in self._excludes))
        return ','.join(filter(None, [lower_str, upper_str, excludes_str]))

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'<PySpecSet {self}>'

    def copy(self) -> PySpecSet:
        if False:
            i = 10
            return i + 15
        'Create a new specifierset that is same as the original one.'
        if self.is_impossible:
            return ImpossiblePySpecSet()
        instance = self.__class__(str(self), False)
        instance._lower_bound = self._lower_bound
        instance._upper_bound = self._upper_bound
        instance._excludes = self._excludes[:]
        return instance

    @lru_cache()
    def __and__(self, other: PySpecSet) -> PySpecSet:
        if False:
            for i in range(10):
                print('nop')
        if any((s.is_impossible for s in (self, other))):
            return ImpossiblePySpecSet()
        if self.is_allow_all:
            return other.copy()
        elif other.is_allow_all:
            return self.copy()
        rv = self.copy()
        excludes = set(rv._excludes) | set(other._excludes)
        lower = max(rv._lower_bound, other._lower_bound)
        upper = min(rv._upper_bound, other._upper_bound)
        rv._rearrange(lower, upper, excludes)
        return rv

    @lru_cache()
    def __or__(self, other: PySpecSet) -> PySpecSet:
        if False:
            for i in range(10):
                print('nop')
        if self.is_impossible:
            return other.copy()
        elif other.is_impossible:
            return self.copy()
        if self.is_allow_all:
            return self.copy()
        elif other.is_allow_all:
            return other.copy()
        rv = self.copy()
        (left, right) = sorted([rv, other], key=lambda x: x._lower_bound)
        excludes = set(left._excludes) & set(right._excludes)
        lower = left._lower_bound
        upper = max(left._upper_bound, right._upper_bound)
        if right._lower_bound > left._upper_bound:
            excludes.update(self._populate_version_range(left._upper_bound, right._lower_bound))
        rv._rearrange(lower, upper, excludes)
        return rv

    def _populate_version_range(self, lower: Version, upper: Version) -> Iterable[Version]:
        if False:
            return 10
        'Expand the version range to a collection of versions to exclude,\n        taking the released python versions into consideration.\n        '
        assert lower < upper
        prev = lower
        while prev < upper:
            if prev[-2:] == Version((0, 0)):
                cur = prev.bump(0)
                if cur <= upper:
                    yield Version((prev[0], '*'))
                    prev = cur
                    continue
            if prev[-1] == 0:
                cur = prev.bump(1)
                if cur <= upper:
                    yield prev[:2].complete('*')
                    prev = prev.bump(0) if cur.is_py2 and cast(int, cur[1]) > self.PY_MAX_MINOR_VERSION[cur[:1]] else cur
                    continue
                while prev < upper:
                    yield prev
                    prev = prev.bump()
                break
            cur = prev.bump(1)
            if cur <= upper:
                current_max = self.PY_MAX_MINOR_VERSION[prev[:2]]
                for z in range(cast(int, prev[2]), current_max + 1):
                    yield prev[:2].complete(z)
                prev = prev.bump(0) if cur.is_py2 and cast(int, cur[1]) > self.PY_MAX_MINOR_VERSION[cur[:1]] else cur
            else:
                while prev < upper:
                    yield prev
                    prev = prev.bump()
                break

    @lru_cache()
    def is_superset(self, other: str | SpecifierSet) -> bool:
        if False:
            print('Hello World!')
        if self.is_impossible:
            return False
        if self.is_allow_all:
            return True
        other = type(self)(str(other))
        if other._upper_bound >= self.MAX_MAJOR_VERSION:
            other._upper_bound = self.MAX_MAJOR_VERSION
        (lower, upper, excludes) = self._merge_bounds_and_excludes(other._lower_bound, other._upper_bound, self._excludes)
        if self._lower_bound > other._lower_bound or self._upper_bound < other._upper_bound:
            return False
        return lower <= other._lower_bound and upper >= other._upper_bound and (set(excludes) <= set(other._excludes))

    @lru_cache()
    def is_subset(self, other: str | SpecifierSet) -> bool:
        if False:
            while True:
                i = 10
        if self.is_impossible:
            return False
        other = type(self)(str(other))
        if other._upper_bound >= self.MAX_MAJOR_VERSION:
            other._upper_bound = Version.MAX
        if other.is_allow_all:
            return True
        (lower, upper, excludes) = self._merge_bounds_and_excludes(self._lower_bound, self._upper_bound, other._excludes)
        if self._lower_bound < other._lower_bound or self._upper_bound > other._upper_bound:
            return False
        return lower <= self._lower_bound and upper >= self._upper_bound and (set(self._excludes) >= set(excludes))

    def as_marker_string(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        if self.is_allow_all:
            return ''
        result = []
        excludes = []
        full_excludes = []
        for spec in sorted(self, key=attrgetter('version')):
            (op, version) = (spec.operator, spec.version)
            if len(version.split('.')) < 3:
                key = 'python_version'
            else:
                key = 'python_full_version'
                if version[-2:] == '.*':
                    version = version[:-2]
                    key = 'python_version'
            if op == '!=':
                if key == 'python_version':
                    excludes.append(version)
                else:
                    full_excludes.append(version)
            else:
                result.append(f'{key}{op}{version!r}')
        if excludes:
            result.append('python_version not in {!r}'.format(', '.join(sorted(excludes))))
        if full_excludes:
            result.append('python_full_version not in {!r}'.format(', '.join(sorted(full_excludes))))
        return ' and '.join(result)

    def supports_py2(self) -> bool:
        if False:
            while True:
                i = 10
        return self._lower_bound.is_py2

class ImpossiblePySpecSet(PySpecSet):
    """
    A special subclass of PySpecSet that references to an impossible specifier set.
    """

    def __init__(self, version_str: str='', analyze: bool=True) -> None:
        if False:
            return 10
        super().__init__(specifiers=version_str, analyze=False)
        self._lower_bound = Version.MAX
        self._upper_bound = Version.MIN

    @property
    def is_impossible(self) -> bool:
        if False:
            print('Hello World!')
        return True