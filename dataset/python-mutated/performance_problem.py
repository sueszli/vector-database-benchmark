from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Type
from sentry.issues.grouptype import GroupType, get_group_type_by_type_id
from sentry.issues.issue_occurrence import IssueEvidence

@dataclass
class PerformanceProblem:
    fingerprint: str
    op: str
    desc: str
    type: Type[GroupType]
    parent_span_ids: Optional[Sequence[str]]
    cause_span_ids: Optional[Sequence[str]]
    offender_span_ids: Sequence[str]
    evidence_data: Optional[Mapping[str, Any]]
    evidence_display: Sequence[IssueEvidence]

    def to_dict(self) -> Mapping[str, Any]:
        if False:
            print('Hello World!')
        return {'fingerprint': self.fingerprint, 'op': self.op, 'desc': self.desc, 'type': self.type.type_id, 'parent_span_ids': self.parent_span_ids, 'cause_span_ids': self.cause_span_ids, 'offender_span_ids': self.offender_span_ids, 'evidence_data': self.evidence_data, 'evidence_display': [evidence.to_dict() for evidence in self.evidence_display]}

    @property
    def title(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.type.description

    @classmethod
    def from_dict(cls, data: dict):
        if False:
            while True:
                i = 10
        return cls(data['fingerprint'], data['op'], data['desc'], get_group_type_by_type_id(data['type']), data['parent_span_ids'], data['cause_span_ids'], data['offender_span_ids'], data.get('evidence_data', {}), [IssueEvidence(evidence['name'], evidence['value'], evidence['important']) for evidence in data.get('evidence_display', [])])

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, PerformanceProblem):
            return NotImplemented
        return self.fingerprint == other.fingerprint and self.offender_span_ids == other.offender_span_ids and (self.type == other.type)

    def __hash__(self):
        if False:
            return 10
        return hash((self.fingerprint, frozenset(self.offender_span_ids), self.type))