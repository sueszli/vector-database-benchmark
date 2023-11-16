from __future__ import annotations
from typing import Any
import pytest
from sentry.issues.grouptype import PerformanceRenderBlockingAssetSpanGroupType
from sentry.models.options.project_option import ProjectOption
from sentry.testutils.cases import TestCase
from sentry.testutils.performance_issues.event_generators import PROJECT_ID, create_span, modify_span_start
from sentry.testutils.silo import region_silo_test
from sentry.utils.performance_issues.detectors.render_blocking_asset_span_detector import RenderBlockingAssetSpanDetector
from sentry.utils.performance_issues.performance_detection import get_detection_settings, run_detector_on_data
from sentry.utils.performance_issues.performance_problem import PerformanceProblem

def _valid_render_blocking_asset_event(url: str) -> dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    return {'event_id': 'a' * 16, 'project': PROJECT_ID, 'measurements': {'fcp': {'value': 2500.0, 'unit': 'millisecond'}}, 'spans': [create_span('resource.script', desc=url, duration=1000.0, data={'http.response_transfer_size': 1200000, 'http.response_content_length': 1200000, 'http.decoded_response_content_length': 2000000, 'resource.render_blocking_status': 'blocking'})], 'contexts': {'trace': {'span_id': 'c' * 16}}, 'transaction': '/'}

def find_problems(settings, event: dict[str, Any]) -> list[PerformanceProblem]:
    if False:
        while True:
            i = 10
    detector = RenderBlockingAssetSpanDetector(settings, event)
    run_detector_on_data(detector, event)
    return list(detector.stored_problems.values())

@region_silo_test(stable=True)
@pytest.mark.django_db
class RenderBlockingAssetDetectorTest(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self._settings = get_detection_settings()

    def find_problems(self, event):
        if False:
            for i in range(10):
                print('nop')
        return find_problems(self._settings, event)

    def test_detects_render_blocking_asset(self):
        if False:
            while True:
                i = 10
        event = _valid_render_blocking_asset_event('https://example.com/a.js')
        assert self.find_problems(event) == [PerformanceProblem(fingerprint='1-1004-ba43281143a88ba902029356cb543dd0bff8f41c', op='resource.script', desc='https://example.com/a.js', type=PerformanceRenderBlockingAssetSpanGroupType, parent_span_ids=[], cause_span_ids=[], offender_span_ids=['bbbbbbbbbbbbbbbb'], evidence_data={'op': 'resource.script', 'parent_span_ids': [], 'cause_span_ids': [], 'offender_span_ids': ['bbbbbbbbbbbbbbbb']}, evidence_display=[])]

    def test_respects_project_option(self):
        if False:
            while True:
                i = 10
        project = self.create_project()
        event = _valid_render_blocking_asset_event('https://example.com/a.js')
        event['project_id'] = project.id
        settings = get_detection_settings(project.id)
        detector = RenderBlockingAssetSpanDetector(settings, event)
        assert detector.is_creation_allowed_for_project(project)
        ProjectOption.objects.set_value(project=project, key='sentry:performance_issue_settings', value={'large_render_blocking_asset_detection_enabled': False})
        settings = get_detection_settings(project.id)
        detector = RenderBlockingAssetSpanDetector(settings, event)
        assert not detector.is_creation_allowed_for_project(project)

    def test_does_not_detect_if_resource_overlaps_fcp(self):
        if False:
            return 10
        event = _valid_render_blocking_asset_event('https://example.com/a.js')
        for span in event['spans']:
            if span['op'] == 'resource.script':
                modify_span_start(span, 2000.0)
        assert self.find_problems(event) == []

    def test_does_not_detect_with_no_fcp(self):
        if False:
            print('Hello World!')
        event = _valid_render_blocking_asset_event('https://example.com/a.js')
        event['measurements']['fcp']['value'] = None
        assert self.find_problems(event) == []

    def test_does_not_detect_with_no_measurements(self):
        if False:
            return 10
        event = _valid_render_blocking_asset_event('https://example.com/a.js')
        event['measurements'] = None
        assert self.find_problems(event) == []

    def test_does_not_detect_with_short_render_blocking_asset(self):
        if False:
            for i in range(10):
                print('nop')
        event = _valid_render_blocking_asset_event('https://example.com/a.js')
        for span in event['spans']:
            if span['op'] == 'resource.script':
                span['timestamp'] = 0.1
        assert self.find_problems(event) == []

    def test_does_not_detect_if_too_small(self):
        if False:
            return 10
        event = _valid_render_blocking_asset_event('https://example.com/a.js')
        for span in event['spans']:
            if span['op'] == 'resource.script':
                span['data']['http.response_content_length'] = 400000
        assert self.find_problems(event) == []

    def test_does_not_detect_if_missing_size(self):
        if False:
            return 10
        event = _valid_render_blocking_asset_event('https://example.com/a.js')
        for span in event['spans']:
            if span['op'] == 'resource.script':
                del span['data']
        assert self.find_problems(event) == []

    def test_does_not_detect_if_too_large(self):
        if False:
            for i in range(10):
                print('nop')
        event = _valid_render_blocking_asset_event('https://example.com/a.js')
        for span in event['spans']:
            if span['op'] == 'resource.script':
                span['data']['http.response_content_length'] = 18446744073709552000
        assert self.find_problems(event) == []

    def test_detects_if_render_blocking_status_is_missing(self):
        if False:
            print('Hello World!')
        event = _valid_render_blocking_asset_event('https://example.com/a.js')
        for span in event['spans']:
            del span['data']['resource.render_blocking_status']
        assert self.find_problems(event) == [PerformanceProblem(fingerprint='1-1004-ba43281143a88ba902029356cb543dd0bff8f41c', op='resource.script', desc='https://example.com/a.js', type=PerformanceRenderBlockingAssetSpanGroupType, parent_span_ids=[], cause_span_ids=[], offender_span_ids=['bbbbbbbbbbbbbbbb'], evidence_data={'op': 'resource.script', 'parent_span_ids': [], 'cause_span_ids': [], 'offender_span_ids': ['bbbbbbbbbbbbbbbb']}, evidence_display=[])]

    def test_does_not_detect_if_render_blocking_status_is_non_blocking(self):
        if False:
            i = 10
            return i + 15
        event = _valid_render_blocking_asset_event('https://example.com/a.js')
        for span in event['spans']:
            span['data']['resource.render_blocking_status'] = 'non-blocking'
        assert self.find_problems(event) == []

@pytest.mark.django_db
@pytest.mark.parametrize('expected,first_url,second_url', [(True, '/foo.js', '/foo.js'), (False, '/foo.js', '/bar.js'), (True, '/foo.js?bar', '/foo.js?baz'), (True, '/foo.6a7a65d8.chunk.js', '/foo.9aa723de.chunk.js'), (False, '/foo.6a7a65d8.chunk.js', '/bar.9aa723de.chunk.js'), (True, '/2.6a7a65d8.chunk.js', '/3.9aa723de.chunk.js'), (True, '/foo.6a7a65d8.js', '/foo.9aa723de.js'), (True, '/foo-6a7a65d8bf641868d8683022a5b62f54.js', '/foo-9aa723de2aa141eeb2e61a2c6bbf0d53.js'), (True, '/foo.6a7a65d8.woff2', '/foo.9aa723de.woff2'), (False, '/foo.6a7a65d8.js', '/bar.9aa723de.js'), (False, '/foo-6a7a65d8bf641868d8683022a5b62f54.js', '/bar-9aa723de2aa141eeb2e61a2c6bbf0d53.js'), (True, '/foo-6dbbbd06.cfdb8c53.js', '/foo-7753eadb.362e3029.js'), (False, '/foo-6dbbbd06.cfdb8c53.js', '/bar-7753eadb.362e3029.js'), (True, '/6a7a65d8bf641868d868.js', '/9aa723de2aa141eeb2e6.js'), (False, '/6a7a65d8.js', '/9aa723de.js'), (True, '/6a7a65d8bf641868d868/foo.js', '/9aa723de2aa141eeb2e6/foo.js'), (False, '/6a7a65d8/foo.js', '/9aa723de/foo.js'), (True, '/6a7a65d8-bf64-1868-d868-3022a5b62f54.js', '/9aa723de-2aa1-41ee-b2e6-1a2c6bbf0d53.js'), (True, '/v7.7.19.1.2/foo.js', '/v8.10/foo.js'), (False, '/v7.7.19.1.2/foo.js', '/v8.10/bar.js'), (True, '/v1/foo.js', '/v20220301115713/foo.js'), (False, '/v1/foo.js', '/v20220301115713/bar.js'), (True, '/foo/1.js', '/foo/23.js'), (False, '/foo/1.css', '/foo/23.js'), (False, '/foo/1.js', '/bar/23.js'), (False, '/foo/bar1.js', '/foo/bar2.js')])
def test_fingerprint_similarity(expected, first_url, second_url):
    if False:
        for i in range(10):
            print('nop')
    first_event = _valid_render_blocking_asset_event(first_url)
    second_event = _valid_render_blocking_asset_event(second_url)
    settings = get_detection_settings()
    first_problems = find_problems(settings, first_event)
    second_problems = find_problems(settings, second_event)
    assert len(first_problems) == 1
    assert len(second_problems) == 1
    if expected:
        assert first_problems[0].fingerprint == second_problems[0].fingerprint
    else:
        assert first_problems[0].fingerprint != second_problems[0].fingerprint