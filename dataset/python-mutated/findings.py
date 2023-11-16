from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import IntEnum, auto, unique
from typing import Any, Dict, List, NamedTuple, Optional
from sentry.utils import json

class InstanceID(NamedTuple):
    """Every entry in the generated backup JSON file should have a unique model+ordinal combination,
    which serves as its identifier."""
    model: str
    ordinal: Optional[int] = None

    def pretty(self) -> str:
        if False:
            i = 10
            return i + 15
        out = f'InstanceID(model: {self.model!r}'
        if self.ordinal:
            out += f', ordinal: {self.ordinal}'
        return out + ')'

class FindingKind(IntEnum):
    pass

@unique
class ComparatorFindingKind(FindingKind):
    Unknown = auto()
    UnorderedInput = auto()
    UnequalCounts = auto()
    UnequalJSON = auto()
    AutoSuffixComparator = auto()
    AutoSuffixComparatorExistenceCheck = auto()
    DatetimeEqualityComparator = auto()
    DatetimeEqualityComparatorExistenceCheck = auto()
    DateUpdatedComparator = auto()
    DateUpdatedComparatorExistenceCheck = auto()
    EmailObfuscatingComparator = auto()
    EmailObfuscatingComparatorExistenceCheck = auto()
    HashObfuscatingComparator = auto()
    HashObfuscatingComparatorExistenceCheck = auto()
    ForeignKeyComparator = auto()
    ForeignKeyComparatorExistenceCheck = auto()
    IgnoredComparator = auto()
    IgnoredComparatorExistenceCheck = auto()
    SecretHexComparator = auto()
    SecretHexComparatorExistenceCheck = auto()
    SubscriptionIDComparator = auto()
    SubscriptionIDComparatorExistenceCheck = auto()
    UUID4Comparator = auto()
    UUID4ComparatorExistenceCheck = auto()
    UserPasswordObfuscatingComparator = auto()
    UserPasswordObfuscatingComparatorExistenceCheck = auto()

@dataclass(frozen=True)
class Finding(ABC):
    """
    A JSON serializable and user-reportable finding for an import/export operation. Don't use this
    class directly - inherit from it, set a specific `kind` type, and define your own pretty
    printer!
    """
    on: InstanceID
    left_pk: Optional[int] = None
    right_pk: Optional[int] = None
    reason: str = ''

    def get_finding_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.__class__.__name__

    def _pretty_inner(self) -> str:
        if False:
            return 10
        '\n        Pretty print only the fields on the shared `Finding` portion.\n        '
        out = f'\n    on: {self.on.pretty()}'
        if self.left_pk:
            out += f',\n    left_pk: {self.left_pk}'
        if self.right_pk:
            out += f',\n    right_pk: {self.right_pk}'
        if self.reason:
            out += f',\n    reason: {self.reason}'
        return out

    @abstractmethod
    def pretty(self) -> str:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        if False:
            return 10
        pass

@dataclass(frozen=True)
class ComparatorFinding(Finding):
    """
    Store all information about a single failed matching between expected and actual output.
    """
    kind: ComparatorFindingKind = ComparatorFindingKind.Unknown

    def pretty(self) -> str:
        if False:
            while True:
                i = 10
        return f'ComparatorFinding(\n    kind: {self.kind.name},{self._pretty_inner()}\n)'

    def to_dict(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return asdict(self)

class ComparatorFindings:
    """A wrapper type for a list of 'ComparatorFinding' which enables pretty-printing in asserts."""

    def __init__(self, findings: List[ComparatorFinding]):
        if False:
            for i in range(10):
                print('nop')
        self.findings = findings

    def append(self, finding: ComparatorFinding) -> None:
        if False:
            i = 10
            return i + 15
        self.findings.append(finding)

    def empty(self) -> bool:
        if False:
            while True:
                i = 10
        return not self.findings

    def extend(self, findings: List[ComparatorFinding]) -> None:
        if False:
            return 10
        self.findings += findings

    def pretty(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '\n'.join((f.pretty() for f in self.findings))

class FindingJSONEncoder(json.JSONEncoder):
    """JSON serializer that handles findings properly."""

    def default(self, obj):
        if False:
            return 10
        if isinstance(obj, Finding):
            kind = getattr(obj, 'kind', None)
            d = obj.to_dict()
            d['finding'] = obj.get_finding_name()
            if isinstance(kind, FindingKind):
                d['kind'] = kind.name
            elif isinstance(kind, str):
                d['kind'] = kind
            return d
        return super().default(obj)