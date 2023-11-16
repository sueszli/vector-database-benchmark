from __future__ import annotations
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence, Type, TypedDict, cast
from django.utils.timezone import is_aware
from sentry import nodestore
from sentry.issues.grouptype import GroupType, get_group_type_by_type_id
from sentry.utils.dates import parse_timestamp
DEFAULT_LEVEL = 'info'

class IssueEvidenceData(TypedDict):
    name: str
    value: str
    important: bool

class IssueOccurrenceData(TypedDict):
    id: str
    project_id: int
    event_id: str
    fingerprint: Sequence[str]
    issue_title: str
    subtitle: str
    resource_id: str | None
    evidence_data: Mapping[str, Any]
    evidence_display: Sequence[IssueEvidenceData]
    type: int
    detection_time: float
    level: Optional[str]
    culprit: Optional[str]

@dataclass(frozen=True)
class IssueEvidence:
    name: str
    value: str
    important: bool

    def to_dict(self) -> IssueEvidenceData:
        if False:
            while True:
                i = 10
        return {'name': self.name, 'value': self.value, 'important': self.important}

@dataclass(frozen=True)
class IssueOccurrence:
    """
    A class representing a specific occurrence of an issue. Separate to an `Event`. An `Event` may
    have 0-M `IssueOccurrences` associated with it, and each `IssueOccurrence` is associated with
    one `Event`.

    Longer term, we might change this relationship so that each `IssueOccurrence` is the primary
    piece of data that is passed around. It would have an `Event` associated with it.
    """
    id: str
    project_id: int
    event_id: str
    fingerprint: Sequence[str]
    issue_title: str
    subtitle: str
    resource_id: str | None
    evidence_data: Mapping[str, Any]
    evidence_display: Sequence[IssueEvidence]
    type: Type[GroupType]
    detection_time: datetime
    level: str
    culprit: str

    def __post_init__(self) -> None:
        if False:
            i = 10
            return i + 15
        if not is_aware(self.detection_time):
            raise ValueError('detection_time must be timezone aware')

    def to_dict(self) -> IssueOccurrenceData:
        if False:
            while True:
                i = 10
        return {'id': self.id, 'project_id': self.project_id, 'event_id': self.event_id, 'fingerprint': self.fingerprint, 'issue_title': self.issue_title, 'subtitle': self.subtitle, 'resource_id': self.resource_id, 'evidence_data': self.evidence_data, 'evidence_display': [evidence.to_dict() for evidence in self.evidence_display], 'type': self.type.type_id, 'detection_time': self.detection_time.timestamp(), 'level': self.level, 'culprit': self.culprit}

    @classmethod
    def from_dict(cls, data: IssueOccurrenceData) -> IssueOccurrence:
        if False:
            for i in range(10):
                print('nop')
        level = data.get('level')
        if not level:
            level = DEFAULT_LEVEL
        culprit = data.get('culprit')
        if not culprit:
            culprit = ''
        return cls(data['id'], data['project_id'], data['event_id'], data['fingerprint'], data['issue_title'], data['subtitle'], data['resource_id'], data['evidence_data'], [IssueEvidence(evidence['name'], evidence['value'], evidence['important']) for evidence in data['evidence_display']], get_group_type_by_type_id(data['type']), cast(datetime, parse_timestamp(data['detection_time'])), level, culprit)

    @property
    def important_evidence_display(self) -> Optional[IssueEvidence]:
        if False:
            print('Hello World!')
        '\n        Returns the most important piece of evidence for display in space constrained integrations.\n        If multiple pieces of evidence are marked as important, returns the first one seen.\n        '
        for evidence in self.evidence_display:
            if evidence.important:
                return evidence
        return None

    def __eq__(self, other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, IssueOccurrence):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash(self.id)

    @classmethod
    def build_storage_identifier(cls, id_: str, project_id: int) -> str:
        if False:
            return 10
        identifier = hashlib.md5(f'{id_}::{project_id}'.encode()).hexdigest()
        return f'i-o:{identifier}'

    def save(self) -> None:
        if False:
            while True:
                i = 10
        nodestore.set(self.build_storage_identifier(self.id, self.project_id), self.to_dict())

    @classmethod
    def fetch(cls, id_: str, project_id: int) -> Optional[IssueOccurrence]:
        if False:
            print('Hello World!')
        results = nodestore.get(cls.build_storage_identifier(id_, project_id))
        if results:
            return IssueOccurrence.from_dict(results)
        return None

    @classmethod
    def fetch_multi(cls, ids: Sequence[str], project_id: int) -> Sequence[Optional[IssueOccurrence]]:
        if False:
            for i in range(10):
                print('nop')
        ids = [cls.build_storage_identifier(id, project_id) for id in ids]
        results = nodestore.get_multi(ids)
        return [IssueOccurrence.from_dict(results[_id]) if results.get(_id) else None for _id in ids]