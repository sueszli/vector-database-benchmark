import copy
import time
from unittest.mock import patch
import pytest
from sentry.sentry_metrics import indexer
from sentry.sentry_metrics.use_case_id_registry import UseCaseID
from sentry.snuba.metrics import SingularEntityDerivedMetric
from sentry.snuba.metrics.fields.snql import complement, division_float
from sentry.snuba.metrics.naming_layer.mapping import get_mri, get_public_name_from_mri
from sentry.snuba.metrics.naming_layer.mri import SessionMRI
from sentry.snuba.metrics.naming_layer.public import SessionMetricKey
from sentry.testutils.cases import OrganizationMetricMetaIntegrationTestCase
from sentry.testutils.silo import region_silo_test
from tests.sentry.api.endpoints.test_organization_metrics import MOCKED_DERIVED_METRICS, mocked_mri_resolver
MOCKED_DERIVED_METRICS_2 = copy.deepcopy(MOCKED_DERIVED_METRICS)
MOCKED_DERIVED_METRICS_2.update({'derived_metric.multiple_metrics': SingularEntityDerivedMetric(metric_mri='derived_metric.multiple_metrics', metrics=['metric_foo_doe', SessionMRI.ALL.value], unit='percentage', snql=lambda *args, metric_ids, alias=None: complement(division_float(*args), alias=SessionMetricKey.CRASH_FREE_RATE.value))})
pytestmark = pytest.mark.sentry_metrics

def _indexer_record(org_id: int, string: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    indexer.record(use_case_id=UseCaseID.SESSIONS, org_id=org_id, string=string)

@region_silo_test(stable=True)
class OrganizationMetricDetailsIntegrationTest(OrganizationMetricMetaIntegrationTestCase):
    endpoint = 'sentry-api-0-organization-metric-details'

    @patch('sentry.snuba.metrics.datasource.get_mri', mocked_mri_resolver(['metric1', 'metric2', 'metric3'], get_mri))
    @patch('sentry.snuba.metrics.get_public_name_from_mri', mocked_mri_resolver(['metric1', 'metric2', 'metric3'], get_public_name_from_mri))
    def test_metric_details(self):
        if False:
            print('Hello World!')
        response = self.get_success_response(self.organization.slug, 'metric1')
        assert response.data == {'name': 'metric1', 'type': 'counter', 'operations': ['max_timestamp', 'min_timestamp', 'sum'], 'unit': None, 'tags': [{'key': 'tag1'}, {'key': 'tag2'}, {'key': 'tag3'}]}
        response = self.get_success_response(self.organization.slug, 'metric2')
        assert response.data == {'name': 'metric2', 'type': 'set', 'operations': ['count_unique', 'max_timestamp', 'min_timestamp'], 'unit': None, 'tags': [{'key': 'tag1'}, {'key': 'tag2'}, {'key': 'tag4'}]}
        response = self.get_success_response(self.organization.slug, 'metric3')
        assert response.data == {'name': 'metric3', 'type': 'set', 'operations': ['count_unique', 'max_timestamp', 'min_timestamp'], 'unit': None, 'tags': []}

    @patch('sentry.snuba.metrics.datasource.get_mri', mocked_mri_resolver(['foo.bar'], get_mri))
    def test_metric_details_metric_does_not_exist_in_indexer(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get_response(self.organization.slug, 'foo.bar')
        assert response.status_code == 404
        assert response.data['detail'] == "Some or all of the metric names in ['foo.bar'] do not exist in the indexer"

    @patch('sentry.snuba.metrics.datasource.get_mri', mocked_mri_resolver(['foo.bar'], get_mri))
    @patch('sentry.snuba.metrics.get_public_name_from_mri', mocked_mri_resolver(['foo.bar'], get_public_name_from_mri))
    def test_metric_details_metric_does_not_have_data(self):
        if False:
            while True:
                i = 10
        _indexer_record(self.organization.id, 'foo.bar')
        response = self.get_response(self.organization.slug, 'foo.bar')
        assert response.status_code == 404
        _indexer_record(self.organization.id, SessionMRI.RAW_SESSION.value)
        response = self.get_response(self.organization.slug, SessionMetricKey.CRASH_FREE_RATE.value)
        assert response.status_code == 404
        assert response.data['detail'] == f"The following metrics ['{SessionMRI.CRASH_FREE_RATE.value}'] do not exist in the dataset"

    def test_derived_metric_details(self):
        if False:
            while True:
                i = 10
        self.store_session(self.build_session(project_id=self.project.id, started=time.time() // 60 * 60, status='ok', release='foobar@2.0'))
        response = self.get_success_response(self.organization.slug, SessionMetricKey.CRASH_FREE_RATE.value)
        assert response.data == {'name': SessionMetricKey.CRASH_FREE_RATE.value, 'type': 'numeric', 'operations': [], 'unit': 'percentage', 'tags': [{'key': 'environment'}, {'key': 'release'}]}

    @patch('sentry.snuba.metrics.fields.base.DERIVED_METRICS', MOCKED_DERIVED_METRICS_2)
    @patch('sentry.snuba.metrics.datasource.get_mri')
    @patch('sentry.snuba.metrics.datasource.get_derived_metrics')
    def test_incorrectly_setup_derived_metric(self, mocked_derived_metrics, mocked_get_mri):
        if False:
            while True:
                i = 10
        mocked_get_mri.return_value = 'crash_free_fake'
        mocked_derived_metrics.return_value = MOCKED_DERIVED_METRICS_2
        self.store_session(self.build_session(project_id=self.project.id, started=time.time() // 60 * 60, status='ok', release='foobar@2.0', errors=2))
        response = self.get_response(self.organization.slug, 'crash_free_fake')
        assert response.status_code == 400
        assert response.json()['detail'] == "The following metrics {'crash_free_fake'} cannot be computed from single entities. Please revise the definition of these singular entity derived metrics"

    @patch('sentry.snuba.metrics.fields.base.DERIVED_METRICS', MOCKED_DERIVED_METRICS_2)
    @patch('sentry.snuba.metrics.datasource.get_mri', mocked_mri_resolver(['metric_foo_doe', 'derived_metric.multiple_metrics'], get_mri))
    @patch('sentry.snuba.metrics.get_public_name_from_mri', mocked_mri_resolver(['metric_foo_doe', 'derived_metric.multiple_metrics'], get_public_name_from_mri))
    @patch('sentry.snuba.metrics.datasource.get_derived_metrics')
    def test_same_entity_multiple_metric_ids_missing_data(self, mocked_derived_metrics):
        if False:
            i = 10
            return i + 15
        '\n        Test when not requested metrics have data in the dataset\n        '
        mocked_derived_metrics.return_value = MOCKED_DERIVED_METRICS_2
        _indexer_record(self.organization.id, 'metric_foo_doe')
        self.store_session(self.build_session(project_id=self.project.id, started=time.time() // 60 * 60, status='ok', release='foobar@2.0', errors=2))
        response = self.get_response(self.organization.slug, 'derived_metric.multiple_metrics')
        assert response.status_code == 404
        assert response.json()['detail'] == "Not all the requested metrics or the constituent metrics in ['derived_metric.multiple_metrics'] have data in the dataset"

    @patch('sentry.snuba.metrics.fields.base.DERIVED_METRICS', MOCKED_DERIVED_METRICS_2)
    @patch('sentry.snuba.metrics.datasource.get_mri', mocked_mri_resolver(['metric_foo_doe', 'derived_metric.multiple_metrics'], get_mri))
    @patch('sentry.snuba.metrics.get_public_name_from_mri', mocked_mri_resolver(['metric_foo_doe', 'derived_metric.multiple_metrics'], get_public_name_from_mri))
    @patch('sentry.snuba.metrics.datasource.get_derived_metrics')
    def test_same_entity_multiple_metric_ids(self, mocked_derived_metrics):
        if False:
            while True:
                i = 10
        '\n        Test that ensures that if a derived metric is defined with constituent metrics that\n        belong to the same entity but have different ids, then we are able to correctly return\n        its detail info\n        '
        mocked_derived_metrics.return_value = MOCKED_DERIVED_METRICS_2
        org_id = self.project.organization.id
        self.store_session(self.build_session(project_id=self.project.id, started=time.time() // 60 * 60, status='ok', release='foobar@2.0', errors=2))
        self.store_metric(org_id=org_id, project_id=self.project.id, type='counter', name='metric_foo_doe', timestamp=int(time.time() // 60 - 2) * 60, tags={'release': 'foow'}, value=5, use_case_id=UseCaseID.SESSIONS)
        response = self.get_success_response(self.organization.slug, 'derived_metric.multiple_metrics')
        assert response.data == {'name': 'derived_metric.multiple_metrics', 'type': 'numeric', 'operations': [], 'unit': 'percentage', 'tags': [{'key': 'release'}]}