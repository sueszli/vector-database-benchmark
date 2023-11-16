from __future__ import annotations
import hashlib
import os
from collections import defaultdict
from datetime import timedelta
from typing import List, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlparse
from django.utils.encoding import force_bytes
from sentry import features
from sentry.issues.grouptype import PerformanceNPlusOneAPICallsGroupType
from sentry.issues.issue_occurrence import IssueEvidence
from sentry.models.organization import Organization
from sentry.models.project import Project
from sentry.utils.performance_issues.detectors.utils import get_total_span_duration
from ..base import DETECTOR_TYPE_TO_GROUP_TYPE, DetectorType, PerformanceDetector, fingerprint_http_spans, get_notification_attachment_body, get_span_evidence_value, get_url_from_span, parameterize_url
from ..performance_problem import PerformanceProblem
from ..types import PerformanceProblemsMap, Span

class NPlusOneAPICallsDetector(PerformanceDetector):
    """
    Detect parallel network calls to the same endpoint.

      [-------- transaction -----------]
         [-------- parent span -----------]
          [n0] https://service.io/resources/?id=12443
          [n1] https://service.io/resources/?id=13342
          [n2] https://service.io/resources/?id=13441
          ...
    """
    __slots__ = ['stored_problems']
    type = DetectorType.N_PLUS_ONE_API_CALLS
    settings_key = DetectorType.N_PLUS_ONE_API_CALLS
    HOST_DENYLIST: list[str] = []

    def init(self):
        if False:
            while True:
                i = 10
        self.stored_problems: PerformanceProblemsMap = {}
        self.spans: list[Span] = []
        self.span_hashes = {}

    def visit_span(self, span: Span) -> None:
        if False:
            i = 10
            return i + 15
        if not NPlusOneAPICallsDetector.is_span_eligible(span):
            return
        op = span.get('op', None)
        if op not in self.settings.get('allowed_span_ops', []):
            return
        self.span_hashes[span['span_id']] = get_span_hash(span)
        previous_span = self.spans[-1] if len(self.spans) > 0 else None
        if previous_span is None:
            self.spans.append(span)
        elif self._spans_are_concurrent(previous_span, span) and self._spans_are_similar(previous_span, span):
            self.spans.append(span)
        else:
            self._maybe_store_problem()
            self.spans = [span]

    def is_creation_allowed_for_organization(self, organization: Organization) -> bool:
        if False:
            while True:
                i = 10
        return features.has('organizations:performance-n-plus-one-api-calls-detector', organization, actor=None)

    def is_creation_allowed_for_project(self, project: Project) -> bool:
        if False:
            while True:
                i = 10
        return self.settings['detection_enabled']

    @classmethod
    def is_event_eligible(cls, event, project=None):
        if False:
            print('Hello World!')
        trace_op = event.get('contexts', {}).get('trace', {}).get('op')
        if trace_op and trace_op not in ['navigation', 'pageload', 'ui.load', 'ui.action']:
            return False
        return True

    @classmethod
    def is_span_eligible(cls, span: Span) -> bool:
        if False:
            while True:
                i = 10
        span_id = span.get('span_id', None)
        op = span.get('op', None)
        hash = span.get('hash', None)
        if not span_id or not op or (not hash):
            return False
        description = span.get('description')
        if not description:
            return False
        if description.strip()[:3].upper() != 'GET':
            return False
        url = get_url_from_span(span)
        if 'graphql' in url:
            return False
        if '_next/data' in url:
            return False
        if '__nextjs_original-stack-frame' in url:
            return False
        if not url:
            return False
        parsed_url = urlparse(str(url))
        if parsed_url.netloc in cls.HOST_DENYLIST:
            return False
        (_pathname, extension) = os.path.splitext(parsed_url.path)
        if extension and extension in ['.js', '.css', '.svg', '.png', '.mp3', '.jpg', '.jpeg']:
            return False
        return True

    def on_complete(self):
        if False:
            while True:
                i = 10
        self._maybe_store_problem()
        self.spans = []

    def _maybe_store_problem(self):
        if False:
            return 10
        if len(self.spans) < 1:
            return
        if len(self.spans) < self.settings['count']:
            return
        total_duration = get_total_span_duration(self.spans)
        if total_duration < self.settings['total_duration']:
            return
        last_span = self.spans[-1]
        fingerprint = self._fingerprint()
        if not fingerprint:
            return
        offender_span_ids = [span['span_id'] for span in self.spans]
        self.stored_problems[fingerprint] = PerformanceProblem(fingerprint=fingerprint, op=last_span['op'], desc=os.path.commonprefix([span.get('description', '') or '' for span in self.spans]), type=DETECTOR_TYPE_TO_GROUP_TYPE[self.settings_key], cause_span_ids=[], parent_span_ids=[last_span.get('parent_span_id', None)], offender_span_ids=offender_span_ids, evidence_data={'op': last_span['op'], 'cause_span_ids': [], 'parent_span_ids': [last_span.get('parent_span_id', None)], 'offender_span_ids': offender_span_ids, 'transaction_name': self._event.get('transaction', ''), 'num_repeating_spans': str(len(offender_span_ids)) if offender_span_ids else '', 'repeating_spans': self._get_path_prefix(self.spans[0]), 'repeating_spans_compact': get_span_evidence_value(self.spans[0], include_op=False), 'parameters': self._get_parameters()}, evidence_display=[IssueEvidence(name='Offending Spans', value=get_notification_attachment_body(last_span['op'], os.path.commonprefix([span.get('description', '') or '' for span in self.spans])), important=True)])

    def _get_parameters(self) -> List[str]:
        if False:
            return 10
        if not self.spans or len(self.spans) == 0:
            return []
        urls = [get_url_from_span(span) for span in self.spans]
        all_parameters: Mapping[str, List[str]] = defaultdict(list)
        for url in urls:
            parsed_url = urlparse(url)
            parameters = parse_qs(parsed_url.query)
            for (key, value) in parameters.items():
                all_parameters[key] += value
        return ['{{{}: {}}}'.format(key, ','.join(values)) for (key, values) in all_parameters.items()]

    def _get_path_prefix(self, repeating_span) -> str:
        if False:
            for i in range(10):
                print('nop')
        if not repeating_span:
            return ''
        url = get_url_from_span(repeating_span)
        parsed_url = urlparse(url)
        return parsed_url.path or ''

    def _fingerprint(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        first_url = get_url_from_span(self.spans[0])
        parameterized_first_url = parameterize_url(first_url)
        if without_query_params(parameterized_first_url) == without_query_params(first_url):
            return None
        fingerprint = fingerprint_http_spans([self.spans[0]])
        return f'1-{PerformanceNPlusOneAPICallsGroupType.type_id}-{fingerprint}'

    def _spans_are_concurrent(self, span_a: Span, span_b: Span) -> bool:
        if False:
            while True:
                i = 10
        span_a_start: int = span_a.get('start_timestamp', 0) or 0
        span_b_start: int = span_b.get('start_timestamp', 0) or 0
        return timedelta(seconds=abs(span_a_start - span_b_start)) < timedelta(milliseconds=self.settings['concurrency_threshold'])

    def _spans_are_similar(self, span_a: Span, span_b: Span) -> bool:
        if False:
            while True:
                i = 10
        return self.span_hashes[span_a['span_id']] == self.span_hashes[span_b['span_id']] and span_a['parent_span_id'] == span_b['parent_span_id']
HTTP_METHODS = {'GET', 'HEAD', 'POST', 'PUT', 'DELETE', 'CONNECT', 'OPTIONS', 'TRACE', 'PATCH'}

def get_span_hash(span: Span) -> Optional[str]:
    if False:
        return 10
    if span.get('op') != 'http.client':
        return span.get('hash')
    parts = remove_http_client_query_string_strategy(span)
    if not parts:
        return None
    hash = hashlib.md5()
    for part in parts:
        hash.update(force_bytes(part, errors='replace'))
    return hash.hexdigest()[:16]

def remove_http_client_query_string_strategy(span: Span) -> Optional[Sequence[str]]:
    if False:
        i = 10
        return i + 15
    '\n    This is an inline version of the `http.client` parameterization code in\n    `"default:2022-10-27"`, the default span grouping strategy at time of\n    writing. It\'s inlined here to insulate this detector from changes in the\n    strategy, which are coming soon.\n    '
    description = span.get('description') or ''
    parts = description.split(' ', 1)
    if len(parts) != 2:
        return None
    (method, url_str) = parts
    method = method.upper()
    if method not in HTTP_METHODS:
        return None
    url = urlparse(url_str)
    return [method, url.scheme, url.netloc, url.path]

def without_query_params(url: str) -> str:
    if False:
        return 10
    return urlparse(url)._replace(query='').geturl()