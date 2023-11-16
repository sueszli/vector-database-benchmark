from __future__ import annotations
from dataclasses import asdict, dataclass
from enum import auto, unique
from typing import Any, Dict
from sentry.backup.dependencies import get_model_name
from sentry.backup.findings import ComparatorFinding, ComparatorFindingKind, Finding, FindingJSONEncoder, FindingKind, InstanceID
from sentry.models.email import Email
from sentry.services.hybrid_cloud.import_export.model import RpcExportError, RpcExportErrorKind, RpcImportError, RpcImportErrorKind
from sentry.testutils.cases import TestCase
encoder = FindingJSONEncoder(sort_keys=True, ensure_ascii=True, check_circular=True, allow_nan=True, indent=4, encoding='utf-8')

@unique
class TestFindingKind(FindingKind):
    __test__ = False
    Unknown = auto()
    Foo = auto()
    Bar = auto()

@dataclass(frozen=True)
class TestFinding(Finding):
    __test__ = False
    kind: TestFindingKind = TestFindingKind.Unknown

    def pretty(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        out = f'TestFinding(\n    kind: {self.kind.name},\n    on: {self.on.pretty()}'
        if self.left_pk:
            out += f',\n    left_pk: {self.left_pk}'
        if self.right_pk:
            out += f',\n    right_pk: {self.right_pk}'
        if self.reason:
            out += f',\n    reason: {self.reason}'
        return out + '\n)'

    def to_dict(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return asdict(self)

class FindingsTests(TestCase):

    def test_defaults(self):
        if False:
            return 10
        finding = TestFinding(on=InstanceID(model=str(get_model_name(Email))), reason='test reason')
        assert encoder.encode(finding) == '{\n    "finding": "TestFinding",\n    "kind": "Unknown",\n    "left_pk": null,\n    "on": {\n        "model": "sentry.email",\n        "ordinal": null\n    },\n    "reason": "test reason",\n    "right_pk": null\n}'
        assert finding.pretty() == "TestFinding(\n    kind: Unknown,\n    on: InstanceID(model: 'sentry.email'),\n    reason: test reason\n)"

    def test_no_nulls(self):
        if False:
            print('Hello World!')
        finding = TestFinding(kind=TestFindingKind.Foo, on=InstanceID(model=str(get_model_name(Email)), ordinal=1), left_pk=2, right_pk=3, reason='test reason')
        assert encoder.encode(finding) == '{\n    "finding": "TestFinding",\n    "kind": "Foo",\n    "left_pk": 2,\n    "on": {\n        "model": "sentry.email",\n        "ordinal": 1\n    },\n    "reason": "test reason",\n    "right_pk": 3\n}'
        assert finding.pretty() == "TestFinding(\n    kind: Foo,\n    on: InstanceID(model: 'sentry.email', ordinal: 1),\n    left_pk: 2,\n    right_pk: 3,\n    reason: test reason\n)"

    def test_comparator_finding(self):
        if False:
            i = 10
            return i + 15
        finding = ComparatorFinding(kind=ComparatorFindingKind.Unknown, on=InstanceID(model=str(get_model_name(Email)), ordinal=1), left_pk=2, right_pk=3, reason='test reason')
        assert encoder.encode(finding) == '{\n    "finding": "ComparatorFinding",\n    "kind": "Unknown",\n    "left_pk": 2,\n    "on": {\n        "model": "sentry.email",\n        "ordinal": 1\n    },\n    "reason": "test reason",\n    "right_pk": 3\n}'
        assert finding.pretty() == "ComparatorFinding(\n    kind: Unknown,\n    on: InstanceID(model: 'sentry.email', ordinal: 1),\n    left_pk: 2,\n    right_pk: 3,\n    reason: test reason\n)"

    def test_rpc_export_error(self):
        if False:
            for i in range(10):
                print('nop')
        finding = RpcExportError(kind=RpcExportErrorKind.Unknown, on=InstanceID(model=str(get_model_name(Email)), ordinal=1), left_pk=2, right_pk=3, reason='test reason')
        assert encoder.encode(finding) == '{\n    "finding": "RpcExportError",\n    "kind": "Unknown",\n    "left_pk": 2,\n    "on": {\n        "model": "sentry.email",\n        "ordinal": 1\n    },\n    "reason": "test reason",\n    "right_pk": 3\n}'
        assert finding.pretty() == "RpcExportError(\n    kind: Unknown,\n    on: InstanceID(model: 'sentry.email', ordinal: 1),\n    left_pk: 2,\n    right_pk: 3,\n    reason: test reason\n)"

    def test_rpc_import_error(self):
        if False:
            print('Hello World!')
        finding = RpcImportError(kind=RpcImportErrorKind.Unknown, on=InstanceID(model=str(get_model_name(Email)), ordinal=1), left_pk=2, right_pk=3, reason='test reason')
        assert encoder.encode(finding) == '{\n    "finding": "RpcImportError",\n    "kind": "Unknown",\n    "left_pk": 2,\n    "on": {\n        "model": "sentry.email",\n        "ordinal": 1\n    },\n    "reason": "test reason",\n    "right_pk": 3\n}'
        assert finding.pretty() == "RpcImportError(\n    kind: Unknown,\n    on: InstanceID(model: 'sentry.email', ordinal: 1),\n    left_pk: 2,\n    right_pk: 3,\n    reason: test reason\n)"