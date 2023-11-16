import numbers
import re
from bisect import bisect_left
from typing import List, Optional, Tuple, Union
from spack.util.spack_yaml import syaml_dict
from .common import COMMIT_VERSION, EmptyRangeError, VersionLookupError, infinity_versions, is_git_version, iv_min_len
from .lookup import AbstractRefLookup
VALID_VERSION = re.compile('^[A-Za-z0-9_.-][=A-Za-z0-9_.-]*$')
SEGMENT_REGEX = re.compile('(?:(?P<num>[0-9]+)|(?P<str>[a-zA-Z]+))(?P<sep>[_.-]*)')

class VersionStrComponent:
    __slots__ = ['data']

    def __init__(self, data):
        if False:
            print('Hello World!')
        self.data: Union[int, str] = data

    @staticmethod
    def from_string(string):
        if False:
            return 10
        if len(string) >= iv_min_len:
            try:
                string = infinity_versions.index(string)
            except ValueError:
                pass
        return VersionStrComponent(string)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.data)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return ('infinity' if self.data >= len(infinity_versions) else infinity_versions[self.data]) if isinstance(self.data, int) else self.data

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'VersionStrComponent("{self}")'

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return isinstance(other, VersionStrComponent) and self.data == other.data

    def __lt__(self, other):
        if False:
            print('Hello World!')
        lhs_inf = isinstance(self.data, int)
        if isinstance(other, int):
            return not lhs_inf
        rhs_inf = isinstance(other.data, int)
        return not lhs_inf and rhs_inf if lhs_inf ^ rhs_inf else self.data < other.data

    def __le__(self, other):
        if False:
            i = 10
            return i + 15
        return self < other or self == other

    def __gt__(self, other):
        if False:
            while True:
                i = 10
        lhs_inf = isinstance(self.data, int)
        if isinstance(other, int):
            return lhs_inf
        rhs_inf = isinstance(other.data, int)
        return lhs_inf and (not rhs_inf) if lhs_inf ^ rhs_inf else self.data > other.data

    def __ge__(self, other):
        if False:
            while True:
                i = 10
        return self > other or self == other

def parse_string_components(string: str) -> Tuple[tuple, tuple]:
    if False:
        while True:
            i = 10
    string = string.strip()
    if string and (not VALID_VERSION.match(string)):
        raise ValueError('Bad characters in version string: %s' % string)
    segments = SEGMENT_REGEX.findall(string)
    version = tuple((int(m[0]) if m[0] else VersionStrComponent.from_string(m[1]) for m in segments))
    separators = tuple((m[2] for m in segments))
    return (version, separators)

class ConcreteVersion:
    pass

class StandardVersion(ConcreteVersion):
    """Class to represent versions"""
    __slots__ = ['version', 'string', 'separators']

    def __init__(self, string: Optional[str], version: tuple, separators: tuple):
        if False:
            return 10
        self.string = string
        self.version = version
        self.separators = separators

    @staticmethod
    def from_string(string: str):
        if False:
            for i in range(10):
                print('nop')
        return StandardVersion(string, *parse_string_components(string))

    @staticmethod
    def typemin():
        if False:
            i = 10
            return i + 15
        return StandardVersion('', (), ())

    @staticmethod
    def typemax():
        if False:
            while True:
                i = 10
        return StandardVersion('infinity', (VersionStrComponent(len(infinity_versions)),), ())

    def __bool__(self):
        if False:
            return 10
        return True

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, StandardVersion):
            return self.version == other.version
        return False

    def __ne__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, StandardVersion):
            return self.version != other.version
        return True

    def __lt__(self, other):
        if False:
            return 10
        if isinstance(other, StandardVersion):
            return self.version < other.version
        if isinstance(other, ClosedOpenRange):
            return self <= other.lo
        return NotImplemented

    def __le__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, StandardVersion):
            return self.version <= other.version
        if isinstance(other, ClosedOpenRange):
            return self <= other.lo
        return NotImplemented

    def __ge__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, StandardVersion):
            return self.version >= other.version
        if isinstance(other, ClosedOpenRange):
            return self > other.lo
        return NotImplemented

    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, StandardVersion):
            return self.version > other.version
        if isinstance(other, ClosedOpenRange):
            return self > other.lo
        return NotImplemented

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.version)

    def __len__(self):
        if False:
            return 10
        return len(self.version)

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        cls = type(self)
        if isinstance(idx, numbers.Integral):
            return self.version[idx]
        elif isinstance(idx, slice):
            string_arg = []
            pairs = zip(self.version[idx], self.separators[idx])
            for (token, sep) in pairs:
                string_arg.append(str(token))
                string_arg.append(str(sep))
            if string_arg:
                string_arg.pop()
                string_arg = ''.join(string_arg)
                return cls.from_string(string_arg)
            else:
                return StandardVersion.from_string('')
        message = '{cls.__name__} indices must be integers'
        raise TypeError(message.format(cls=cls))

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.string if isinstance(self.string, str) else '.'.join((str(c) for c in self.version))

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'Version("{str(self)}")'

    def __hash__(self):
        if False:
            return 10
        return hash(self.version)

    def __contains__(rhs, lhs):
        if False:
            i = 10
            return i + 15
        if isinstance(lhs, (StandardVersion, ClosedOpenRange, VersionList)):
            return lhs.satisfies(rhs)
        raise ValueError(lhs)

    def intersects(self, other: Union['StandardVersion', 'GitVersion', 'ClosedOpenRange']) -> bool:
        if False:
            i = 10
            return i + 15
        if isinstance(other, StandardVersion):
            return self == other
        return other.intersects(self)

    def overlaps(self, other) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.intersects(other)

    def satisfies(self, other: Union['ClosedOpenRange', 'StandardVersion', 'GitVersion', 'VersionList']) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, GitVersion):
            return False
        if isinstance(other, StandardVersion):
            return self == other
        if isinstance(other, ClosedOpenRange):
            return other.intersects(self)
        if isinstance(other, VersionList):
            return other.intersects(self)
        return NotImplemented

    def union(self, other: Union['ClosedOpenRange', 'StandardVersion']):
        if False:
            return 10
        if isinstance(other, StandardVersion):
            return self if self == other else VersionList([self, other])
        return other.union(self)

    def intersection(self, other: Union['ClosedOpenRange', 'StandardVersion']):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, StandardVersion):
            return self if self == other else VersionList()
        return other.intersection(self)

    def isdevelop(self):
        if False:
            return 10
        'Triggers on the special case of the `@develop-like` version.'
        return any((isinstance(p, VersionStrComponent) and isinstance(p.data, int) for p in self.version))

    @property
    def dotted(self):
        if False:
            print('Hello World!')
        "The dotted representation of the version.\n\n        Example:\n        >>> version = Version('1-2-3b')\n        >>> version.dotted\n        Version('1.2.3b')\n\n        Returns:\n            Version: The version with separator characters replaced by dots\n        "
        return type(self).from_string(self.string.replace('-', '.').replace('_', '.'))

    @property
    def underscored(self):
        if False:
            while True:
                i = 10
        "The underscored representation of the version.\n\n        Example:\n        >>> version = Version('1.2.3b')\n        >>> version.underscored\n        Version('1_2_3b')\n\n        Returns:\n            Version: The version with separator characters replaced by\n                underscores\n        "
        return type(self).from_string(self.string.replace('.', '_').replace('-', '_'))

    @property
    def dashed(self):
        if False:
            print('Hello World!')
        "The dashed representation of the version.\n\n        Example:\n        >>> version = Version('1.2.3b')\n        >>> version.dashed\n        Version('1-2-3b')\n\n        Returns:\n            Version: The version with separator characters replaced by dashes\n        "
        return type(self).from_string(self.string.replace('.', '-').replace('_', '-'))

    @property
    def joined(self):
        if False:
            while True:
                i = 10
        "The joined representation of the version.\n\n        Example:\n        >>> version = Version('1.2.3b')\n        >>> version.joined\n        Version('123b')\n\n        Returns:\n            Version: The version with separator characters removed\n        "
        return type(self).from_string(self.string.replace('.', '').replace('-', '').replace('_', ''))

    def up_to(self, index):
        if False:
            while True:
                i = 10
        "The version up to the specified component.\n\n        Examples:\n        >>> version = Version('1.23-4b')\n        >>> version.up_to(1)\n        Version('1')\n        >>> version.up_to(2)\n        Version('1.23')\n        >>> version.up_to(3)\n        Version('1.23-4')\n        >>> version.up_to(4)\n        Version('1.23-4b')\n        >>> version.up_to(-1)\n        Version('1.23-4')\n        >>> version.up_to(-2)\n        Version('1.23')\n        >>> version.up_to(-3)\n        Version('1')\n\n        Returns:\n            Version: The first index components of the version\n        "
        return self[:index]

class GitVersion(ConcreteVersion):
    """Class to represent versions interpreted from git refs.

    There are two distinct categories of git versions:

    1) GitVersions instantiated with an associated reference version (e.g. 'git.foo=1.2')
    2) GitVersions requiring commit lookups

    Git ref versions that are not paired with a known version are handled separately from
    all other version comparisons. When Spack identifies a git ref version, it associates a
    ``CommitLookup`` object with the version. This object handles caching of information
    from the git repo. When executing comparisons with a git ref version, Spack queries the
    ``CommitLookup`` for the most recent version previous to this git ref, as well as the
    distance between them expressed as a number of commits. If the previous version is
    ``X.Y.Z`` and the distance is ``D``, the git commit version is represented by the
    tuple ``(X, Y, Z, '', D)``. The component ``''`` cannot be parsed as part of any valid
    version, but is a valid component. This allows a git ref version to be less than (older
    than) every Version newer than its previous version, but still newer than its previous
    version.

    To find the previous version from a git ref version, Spack queries the git repo for its
    tags. Any tag that matches a version known to Spack is associated with that version, as
    is any tag that is a known version prepended with the character ``v`` (i.e., a tag
    ``v1.0`` is associated with the known version ``1.0``). Additionally, any tag that
    represents a semver version (X.Y.Z with X, Y, Z all integers) is associated with the
    version it represents, even if that version is not known to Spack. Each tag is then
    queried in git to see whether it is an ancestor of the git ref in question, and if so
    the distance between the two. The previous version is the version that is an ancestor
    with the least distance from the git ref in question.

    This procedure can be circumvented if the user supplies a known version to associate
    with the GitVersion (e.g. ``[hash]=develop``).  If the user prescribes the version then
    there is no need to do a lookup and the standard version comparison operations are
    sufficient.
    """
    __slots__ = ['ref', 'has_git_prefix', 'is_commit', '_ref_lookup', '_ref_version']

    def __init__(self, string: str):
        if False:
            return 10
        self._ref_lookup: Optional[AbstractRefLookup] = None
        self._ref_version: Optional[StandardVersion]
        self.has_git_prefix = string.startswith('git.')
        normalized_string = string[4:] if self.has_git_prefix else string
        if '=' in normalized_string:
            (self.ref, spack_version) = normalized_string.split('=')
            self._ref_version = StandardVersion(spack_version, *parse_string_components(spack_version))
        else:
            self._ref_version = None
            self.ref = normalized_string
        self.is_commit: bool = len(self.ref) == 40 and bool(COMMIT_VERSION.match(self.ref))

    @property
    def ref_version(self) -> StandardVersion:
        if False:
            while True:
                i = 10
        if self._ref_version is not None:
            return self._ref_version
        if self.ref_lookup is None:
            raise VersionLookupError(f"git ref '{self.ref}' cannot be looked up: call attach_lookup first")
        (version_string, distance) = self.ref_lookup.get(self.ref)
        version_string = version_string or '0'
        if distance > 0:
            version_string += f'-git.{distance}'
        self._ref_version = StandardVersion(version_string, *parse_string_components(version_string))
        return self._ref_version

    def intersects(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, GitVersion):
            return self == other
        if isinstance(other, StandardVersion):
            return False
        if isinstance(other, ClosedOpenRange):
            return self.ref_version.intersects(other)
        if isinstance(other, VersionList):
            return any((self.intersects(rhs) for rhs in other))
        raise ValueError(f'Unexpected type {type(other)}')

    def intersection(self, other):
        if False:
            return 10
        if isinstance(other, ConcreteVersion):
            return self if self == other else VersionList()
        return other.intersection(self)

    def overlaps(self, other) -> bool:
        if False:
            while True:
                i = 10
        return self.intersects(other)

    def satisfies(self, other: Union['GitVersion', StandardVersion, 'ClosedOpenRange', 'VersionList']):
        if False:
            return 10
        if isinstance(other, GitVersion):
            return self == other
        if isinstance(other, StandardVersion):
            return False
        if isinstance(other, ClosedOpenRange):
            return self.ref_version.satisfies(other)
        if isinstance(other, VersionList):
            return any((self.satisfies(rhs) for rhs in other))
        raise ValueError(f'Unexpected type {type(other)}')

    def __str__(self):
        if False:
            while True:
                i = 10
        s = f'git.{self.ref}' if self.has_git_prefix else self.ref
        try:
            s += f'={self.ref_version}'
        except VersionLookupError:
            pass
        return s

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'GitVersion("{self}")'

    def __bool__(self):
        if False:
            return 10
        return True

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, GitVersion) and self.ref == other.ref and (self.ref_version == other.ref_version)

    def __ne__(self, other):
        if False:
            return 10
        return not self == other

    def __lt__(self, other):
        if False:
            return 10
        if isinstance(other, GitVersion):
            return (self.ref_version, self.ref) < (other.ref_version, other.ref)
        if isinstance(other, StandardVersion):
            return self.ref_version < other
        if isinstance(other, ClosedOpenRange):
            return self.ref_version < other
        raise ValueError(f'Unexpected type {type(other)}')

    def __le__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, GitVersion):
            return (self.ref_version, self.ref) <= (other.ref_version, other.ref)
        if isinstance(other, StandardVersion):
            return self.ref_version < other
        if isinstance(other, ClosedOpenRange):
            return self.ref_version < other
        raise ValueError(f'Unexpected type {type(other)}')

    def __ge__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, GitVersion):
            return (self.ref_version, self.ref) >= (other.ref_version, other.ref)
        if isinstance(other, StandardVersion):
            return self.ref_version >= other
        if isinstance(other, ClosedOpenRange):
            return self.ref_version > other
        raise ValueError(f'Unexpected type {type(other)}')

    def __gt__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, GitVersion):
            return (self.ref_version, self.ref) > (other.ref_version, other.ref)
        if isinstance(other, StandardVersion):
            return self.ref_version >= other
        if isinstance(other, ClosedOpenRange):
            return self.ref_version > other
        raise ValueError(f'Unexpected type {type(other)}')

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.ref)

    def __contains__(self, other):
        if False:
            while True:
                i = 10
        raise Exception('Not implemented yet')

    @property
    def ref_lookup(self):
        if False:
            i = 10
            return i + 15
        if self._ref_lookup:
            self._ref_lookup.get(self.ref)
            return self._ref_lookup

    def attach_lookup(self, lookup: AbstractRefLookup):
        if False:
            i = 10
            return i + 15
        '\n        Use the git fetcher to look up a version for a commit.\n\n        Since we want to optimize the clone and lookup, we do the clone once\n        and store it in the user specified git repository cache. We also need\n        context of the package to get known versions, which could be tags if\n        they are linked to Git Releases. If we are unable to determine the\n        context of the version, we cannot continue. This implementation is\n        alongside the GitFetcher because eventually the git repos cache will\n        be one and the same with the source cache.\n        '
        self._ref_lookup = lookup

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return self.ref_version.__iter__()

    def __len__(self):
        if False:
            return 10
        return self.ref_version.__len__()

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        return self.ref_version.__getitem__(idx)

    def isdevelop(self):
        if False:
            print('Hello World!')
        return self.ref_version.isdevelop()

    @property
    def dotted(self) -> StandardVersion:
        if False:
            print('Hello World!')
        return self.ref_version.dotted

    @property
    def underscored(self) -> StandardVersion:
        if False:
            for i in range(10):
                print('nop')
        return self.ref_version.underscored

    @property
    def dashed(self) -> StandardVersion:
        if False:
            while True:
                i = 10
        return self.ref_version.dashed

    @property
    def joined(self) -> StandardVersion:
        if False:
            i = 10
            return i + 15
        return self.ref_version.joined

    def up_to(self, index) -> StandardVersion:
        if False:
            while True:
                i = 10
        return self.ref_version.up_to(index)

class ClosedOpenRange:

    def __init__(self, lo: StandardVersion, hi: StandardVersion):
        if False:
            i = 10
            return i + 15
        if hi < lo:
            raise EmptyRangeError(f'{lo}..{hi} is an empty range')
        self.lo: StandardVersion = lo
        self.hi: StandardVersion = hi

    @classmethod
    def from_version_range(cls, lo: StandardVersion, hi: StandardVersion):
        if False:
            print('Hello World!')
        'Construct ClosedOpenRange from lo:hi range.'
        try:
            return ClosedOpenRange(lo, next_version(hi))
        except EmptyRangeError as e:
            raise EmptyRangeError(f'{lo}:{hi} is an empty range') from e

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        hi_prev = prev_version(self.hi)
        if self.lo != StandardVersion.typemin() and self.lo == hi_prev:
            return str(self.lo)
        lhs = '' if self.lo == StandardVersion.typemin() else str(self.lo)
        rhs = '' if hi_prev == StandardVersion.typemax() else str(hi_prev)
        return f'{lhs}:{rhs}'

    def __repr__(self):
        if False:
            while True:
                i = 10
        return str(self)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((self.lo, prev_version(self.hi)))

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, StandardVersion):
            return False
        if isinstance(other, ClosedOpenRange):
            return (self.lo, self.hi) == (other.lo, other.hi)
        return NotImplemented

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, StandardVersion):
            return True
        if isinstance(other, ClosedOpenRange):
            return (self.lo, self.hi) != (other.lo, other.hi)
        return NotImplemented

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, StandardVersion):
            return other > self
        if isinstance(other, ClosedOpenRange):
            return (self.lo, self.hi) < (other.lo, other.hi)
        return NotImplemented

    def __le__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, StandardVersion):
            return other >= self
        if isinstance(other, ClosedOpenRange):
            return (self.lo, self.hi) <= (other.lo, other.hi)
        return NotImplemented

    def __ge__(self, other):
        if False:
            return 10
        if isinstance(other, StandardVersion):
            return other <= self
        if isinstance(other, ClosedOpenRange):
            return (self.lo, self.hi) >= (other.lo, other.hi)
        return NotImplemented

    def __gt__(self, other):
        if False:
            return 10
        if isinstance(other, StandardVersion):
            return other < self
        if isinstance(other, ClosedOpenRange):
            return (self.lo, self.hi) > (other.lo, other.hi)
        return NotImplemented

    def __contains__(rhs, lhs):
        if False:
            return 10
        if isinstance(lhs, (ConcreteVersion, ClosedOpenRange, VersionList)):
            return lhs.satisfies(rhs)
        raise ValueError(f'Unexpected type {type(lhs)}')

    def intersects(self, other: Union[ConcreteVersion, 'ClosedOpenRange', 'VersionList']):
        if False:
            while True:
                i = 10
        if isinstance(other, StandardVersion):
            return self.lo <= other < self.hi
        if isinstance(other, GitVersion):
            return self.lo <= other.ref_version < self.hi
        if isinstance(other, ClosedOpenRange):
            return self.lo < other.hi and other.lo < self.hi
        if isinstance(other, VersionList):
            return any((self.intersects(rhs) for rhs in other))
        raise ValueError(f'Unexpected type {type(other)}')

    def satisfies(self, other: Union['ClosedOpenRange', ConcreteVersion, 'VersionList']):
        if False:
            i = 10
            return i + 15
        if isinstance(other, ConcreteVersion):
            return False
        if isinstance(other, ClosedOpenRange):
            return not (self.lo < other.lo or other.hi < self.hi)
        if isinstance(other, VersionList):
            return any((self.satisfies(rhs) for rhs in other))
        raise ValueError(other)

    def overlaps(self, other: Union['ClosedOpenRange', ConcreteVersion, 'VersionList']) -> bool:
        if False:
            print('Hello World!')
        return self.intersects(other)

    def union(self, other: Union['ClosedOpenRange', ConcreteVersion, 'VersionList']):
        if False:
            print('Hello World!')
        if isinstance(other, StandardVersion):
            return self if self.lo <= other < self.hi else VersionList([self, other])
        if isinstance(other, GitVersion):
            return self if self.lo <= other.ref_version < self.hi else VersionList([self, other])
        if isinstance(other, ClosedOpenRange):
            if self.lo <= other.hi and other.lo <= self.hi:
                return ClosedOpenRange(min(self.lo, other.lo), max(self.hi, other.hi))
            return VersionList([self, other])
        if isinstance(other, VersionList):
            v = other.copy()
            v.add(self)
            return v
        raise ValueError(f'Unexpected type {type(other)}')

    def intersection(self, other: Union['ClosedOpenRange', ConcreteVersion]):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, ConcreteVersion):
            return other if self.intersects(other) else VersionList()
        max_lo = max(self.lo, other.lo)
        min_hi = min(self.hi, other.hi)
        return ClosedOpenRange(max_lo, min_hi) if max_lo < min_hi else VersionList()

class VersionList:
    """Sorted, non-redundant list of Version and ClosedOpenRange elements."""

    def __init__(self, vlist=None):
        if False:
            print('Hello World!')
        self.versions: List[StandardVersion, GitVersion, ClosedOpenRange] = []
        if vlist is not None:
            if isinstance(vlist, str):
                vlist = from_string(vlist)
                if isinstance(vlist, VersionList):
                    self.versions = vlist.versions
                else:
                    self.versions = [vlist]
            else:
                for v in vlist:
                    self.add(ver(v))

    def add(self, item):
        if False:
            return 10
        if isinstance(item, ConcreteVersion):
            i = bisect_left(self, item)
            if (i == 0 or not item.intersects(self[i - 1])) and (i == len(self) or not item.intersects(self[i])):
                self.versions.insert(i, item)
        elif isinstance(item, ClosedOpenRange):
            i = bisect_left(self, item)
            while i > 0 and item.intersects(self[i - 1]):
                item = item.union(self[i - 1])
                del self.versions[i - 1]
                i -= 1
            while i < len(self) and item.intersects(self[i]):
                item = item.union(self[i])
                del self.versions[i]
            self.versions.insert(i, item)
        elif isinstance(item, VersionList):
            for v in item:
                self.add(v)
        else:
            raise TypeError("Can't add %s to VersionList" % type(item))

    @property
    def concrete(self) -> Optional[ConcreteVersion]:
        if False:
            while True:
                i = 10
        return self[0] if len(self) == 1 and isinstance(self[0], ConcreteVersion) else None

    @property
    def concrete_range_as_version(self) -> Optional[ConcreteVersion]:
        if False:
            for i in range(10):
                print('nop')
        'Like concrete, but collapses VersionRange(x, x) to Version(x).\n        This is just for compatibility with old Spack.'
        if len(self) != 1:
            return None
        v = self[0]
        if isinstance(v, ConcreteVersion):
            return v
        if isinstance(v, ClosedOpenRange) and next_version(v.lo) == v.hi:
            return v.lo
        return None

    def copy(self):
        if False:
            print('Hello World!')
        return VersionList(self)

    def lowest(self) -> Optional[StandardVersion]:
        if False:
            print('Hello World!')
        'Get the lowest version in the list.'
        return None if not self else self[0]

    def highest(self) -> Optional[StandardVersion]:
        if False:
            i = 10
            return i + 15
        'Get the highest version in the list.'
        return None if not self else self[-1]

    def highest_numeric(self) -> Optional[StandardVersion]:
        if False:
            return 10
        'Get the highest numeric version in the list.'
        numeric_versions = list(filter(lambda v: str(v) not in infinity_versions, self.versions))
        return None if not any(numeric_versions) else numeric_versions[-1]

    def preferred(self) -> Optional[StandardVersion]:
        if False:
            for i in range(10):
                print('nop')
        'Get the preferred (latest) version in the list.'
        return self.highest_numeric() or self.highest()

    def satisfies(self, other) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, VersionList):
            return all((any((lhs.satisfies(rhs) for rhs in other)) for lhs in self))
        if isinstance(other, (ConcreteVersion, ClosedOpenRange)):
            return all((lhs.satisfies(other) for lhs in self))
        raise ValueError(f'Unsupported type {type(other)}')

    def intersects(self, other):
        if False:
            return 10
        if isinstance(other, VersionList):
            s = o = 0
            while s < len(self) and o < len(other):
                if self[s].intersects(other[o]):
                    return True
                elif self[s] < other[o]:
                    s += 1
                else:
                    o += 1
            return False
        if isinstance(other, (ClosedOpenRange, StandardVersion)):
            return any((v.intersects(other) for v in self))
        raise ValueError(f'Unsupported type {type(other)}')

    def overlaps(self, other) -> bool:
        if False:
            print('Hello World!')
        return self.intersects(other)

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        'Generate human-readable dict for YAML.'
        if self.concrete:
            return syaml_dict([('version', str(self[0]))])
        return syaml_dict([('versions', [str(v) for v in self])])

    @staticmethod
    def from_dict(dictionary):
        if False:
            print('Hello World!')
        'Parse dict from to_dict.'
        if 'versions' in dictionary:
            return VersionList(dictionary['versions'])
        elif 'version' in dictionary:
            return VersionList([Version(dictionary['version'])])
        raise ValueError("Dict must have 'version' or 'versions' in it.")

    def update(self, other: 'VersionList'):
        if False:
            while True:
                i = 10
        for v in other.versions:
            self.add(v)

    def union(self, other: 'VersionList'):
        if False:
            i = 10
            return i + 15
        result = self.copy()
        result.update(other)
        return result

    def intersection(self, other: 'VersionList') -> 'VersionList':
        if False:
            print('Hello World!')
        result = VersionList()
        for (lhs, rhs) in ((self, other), (other, self)):
            for x in lhs:
                i = bisect_left(rhs.versions, x)
                if i > 0:
                    result.add(rhs[i - 1].intersection(x))
                if i < len(rhs):
                    result.add(rhs[i].intersection(x))
        return result

    def intersect(self, other) -> bool:
        if False:
            return 10
        "Intersect this spec's list with other.\n\n        Return True if the spec changed as a result; False otherwise\n        "
        isection = self.intersection(other)
        changed = isection.versions != self.versions
        self.versions = isection.versions
        return changed

    def __contains__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, (ClosedOpenRange, StandardVersion)):
            i = bisect_left(self, other)
            return i > 0 and other in self[i - 1] or (i < len(self) and other in self[i])
        if isinstance(other, VersionList):
            return all((item in self for item in other))
        return False

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        return self.versions[index]

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self.versions)

    def __reversed__(self):
        if False:
            return 10
        return reversed(self.versions)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.versions)

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.versions)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, VersionList):
            return self.versions == other.versions
        return False

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, VersionList):
            return self.versions != other.versions
        return False

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, VersionList):
            return self.versions < other.versions
        return NotImplemented

    def __le__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, VersionList):
            return self.versions <= other.versions
        return NotImplemented

    def __ge__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, VersionList):
            return self.versions >= other.versions
        return NotImplemented

    def __gt__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, VersionList):
            return self.versions > other.versions
        return NotImplemented

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(tuple(self.versions))

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return ','.join((f'={v}' if isinstance(v, StandardVersion) else str(v) for v in self.versions))

    def __repr__(self):
        if False:
            print('Hello World!')
        return str(self.versions)

def next_str(s: str) -> str:
    if False:
        print('Hello World!')
    'Produce the next string of A-Z and a-z characters'
    return s + 'A' if len(s) == 0 or s[-1] == 'z' else s[:-1] + ('a' if s[-1] == 'Z' else chr(ord(s[-1]) + 1))

def prev_str(s: str) -> str:
    if False:
        while True:
            i = 10
    'Produce the previous string of A-Z and a-z characters'
    return s[:-1] if len(s) == 0 or s[-1] == 'A' else s[:-1] + ('Z' if s[-1] == 'a' else chr(ord(s[-1]) - 1))

def next_version_str_component(v: VersionStrComponent) -> VersionStrComponent:
    if False:
        for i in range(10):
            print('nop')
    '\n    Produce the next VersionStrComponent, where\n    masteq -> mastes\n    master -> main\n    '
    data = v.data
    if isinstance(data, int):
        return VersionStrComponent(data + 1)
    while True:
        data = next_str(data)
        if data not in infinity_versions:
            break
    return VersionStrComponent(data)

def prev_version_str_component(v: VersionStrComponent) -> VersionStrComponent:
    if False:
        print('Hello World!')
    '\n    Produce the previous VersionStrComponent, where\n    mastes -> masteq\n    master -> head\n    '
    data = v.data
    if isinstance(data, int):
        return VersionStrComponent(data - 1)
    while True:
        data = prev_str(data)
        if data not in infinity_versions:
            break
    return VersionStrComponent(data)

def next_version(v: StandardVersion) -> StandardVersion:
    if False:
        return 10
    if len(v.version) == 0:
        nxt = VersionStrComponent('A')
    elif isinstance(v.version[-1], VersionStrComponent):
        nxt = next_version_str_component(v.version[-1])
    else:
        nxt = v.version[-1] + 1
    string_components = []
    for (part, sep) in zip(v.version[:-1], v.separators):
        string_components.append(str(part))
        string_components.append(str(sep))
    string_components.append(str(nxt))
    return StandardVersion(''.join(string_components), v.version[:-1] + (nxt,), v.separators)

def prev_version(v: StandardVersion) -> StandardVersion:
    if False:
        for i in range(10):
            print('nop')
    if len(v.version) == 0:
        return v
    elif isinstance(v.version[-1], VersionStrComponent):
        prev = prev_version_str_component(v.version[-1])
    else:
        prev = v.version[-1] - 1
    string_components = []
    for (part, sep) in zip(v.version[:-1], v.separators):
        string_components.append(str(part))
        string_components.append(str(sep))
    string_components.append(str(prev))
    return StandardVersion(''.join(string_components), v.version[:-1] + (prev,), v.separators)

def Version(string: Union[str, int]) -> Union[GitVersion, StandardVersion]:
    if False:
        i = 10
        return i + 15
    if not isinstance(string, (str, int)):
        raise ValueError(f'Cannot construct a version from {type(string)}')
    string = str(string)
    if is_git_version(string):
        return GitVersion(string)
    return StandardVersion.from_string(str(string))

def VersionRange(lo: Union[str, StandardVersion], hi: Union[str, StandardVersion]):
    if False:
        print('Hello World!')
    lo = lo if isinstance(lo, StandardVersion) else StandardVersion.from_string(lo)
    hi = hi if isinstance(hi, StandardVersion) else StandardVersion.from_string(hi)
    return ClosedOpenRange.from_version_range(lo, hi)

def from_string(string) -> Union[VersionList, ClosedOpenRange, StandardVersion, GitVersion]:
    if False:
        print('Hello World!')
    'Converts a string to a version object. This is private. Client code should use ver().'
    string = string.replace(' ', '')
    if ',' in string:
        return VersionList(list(map(from_string, string.split(','))))
    elif ':' in string:
        (s, e) = string.split(':')
        lo = StandardVersion.typemin() if s == '' else StandardVersion.from_string(s)
        hi = StandardVersion.typemax() if e == '' else StandardVersion.from_string(e)
        return VersionRange(lo, hi)
    elif string.startswith('='):
        return Version(string[1:])
    elif is_git_version(string):
        return GitVersion(string)
    else:
        v = StandardVersion.from_string(string)
        return VersionRange(v, v)

def ver(obj) -> Union[VersionList, ClosedOpenRange, StandardVersion, GitVersion]:
    if False:
        print('Hello World!')
    'Parses a Version, VersionRange, or VersionList from a string\n    or list of strings.\n    '
    if isinstance(obj, (list, tuple)):
        return VersionList(obj)
    elif isinstance(obj, str):
        return from_string(obj)
    elif isinstance(obj, (int, float)):
        return from_string(str(obj))
    elif isinstance(obj, (StandardVersion, GitVersion, ClosedOpenRange, VersionList)):
        return obj
    else:
        raise TypeError("ver() can't convert %s to version!" % type(obj))