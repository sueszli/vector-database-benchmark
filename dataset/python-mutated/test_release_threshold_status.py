from datetime import datetime, timedelta
from sentry.api.endpoints.release_thresholds.release_threshold_status_index import EnrichedThreshold, is_error_count_healthy
from sentry.api.serializers import serialize
from sentry.models.environment import Environment
from sentry.models.release import Release
from sentry.models.release_threshold.constants import ReleaseThresholdType, TriggerType
from sentry.models.release_threshold.release_threshold import ReleaseThreshold
from sentry.models.releaseenvironment import ReleaseEnvironment
from sentry.models.releaseprojectenvironment import ReleaseProjectEnvironment
from sentry.testutils.cases import APITestCase, TestCase

class ReleaseThresholdStatusTest(APITestCase):
    endpoint = 'sentry-api-0-organization-release-threshold-statuses'
    method = 'get'

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.user = self.create_user(is_staff=True, is_superuser=True)
        self.project1 = self.create_project(name='foo', organization=self.organization)
        self.project2 = self.create_project(name='bar', organization=self.organization)
        self.project3 = self.create_project(name='biz', organization=self.organization)
        self.canary_environment = Environment.objects.create(organization_id=self.organization.id, name='canary')
        self.production_environment = Environment.objects.create(organization_id=self.organization.id, name='production')
        self.release1 = Release.objects.create(version='v1', organization=self.organization)
        self.release1.add_project(self.project1)
        self.release1.add_project(self.project2)
        self.release2 = Release.objects.create(version='v2', organization=self.organization)
        self.release2.add_project(self.project1)
        self.release3 = Release.objects.create(version='v3', organization=self.organization)
        self.release3.add_project(self.project3)
        ReleaseEnvironment.objects.create(organization_id=self.organization.id, release_id=self.release1.id, environment_id=self.canary_environment.id)
        ReleaseProjectEnvironment.objects.create(release_id=self.release1.id, project_id=self.project1.id, environment_id=self.canary_environment.id)
        ReleaseProjectEnvironment.objects.create(release_id=self.release1.id, project_id=self.project2.id, environment_id=self.canary_environment.id)
        ReleaseEnvironment.objects.create(organization_id=self.organization.id, release_id=self.release2.id, environment_id=self.production_environment.id)
        ReleaseProjectEnvironment.objects.create(release_id=self.release2.id, project_id=self.project1.id, environment_id=self.production_environment.id)
        ReleaseThreshold.objects.create(threshold_type=ReleaseThresholdType.TOTAL_ERROR_COUNT, trigger_type=1, value=100, window_in_seconds=100, project=self.project1, environment=self.canary_environment)
        ReleaseThreshold.objects.create(threshold_type=ReleaseThresholdType.NEW_ISSUE_COUNT, trigger_type=1, value=100, window_in_seconds=100, project=self.project1, environment=self.canary_environment)
        ReleaseThreshold.objects.create(threshold_type=ReleaseThresholdType.TOTAL_ERROR_COUNT, trigger_type=1, value=100, window_in_seconds=100, project=self.project1, environment=self.production_environment)
        ReleaseThreshold.objects.create(threshold_type=ReleaseThresholdType.TOTAL_ERROR_COUNT, trigger_type=1, value=100, window_in_seconds=100, project=self.project2, environment=self.canary_environment)
        ReleaseThreshold.objects.create(threshold_type=ReleaseThresholdType.TOTAL_ERROR_COUNT, trigger_type=1, value=100, window_in_seconds=100, project=self.project3)
        self.login_as(user=self.user)

    def test_get_success(self):
        if False:
            print('Hello World!')
        '\n        Tests fetching all thresholds (env+project agnostic) within the past 24hrs.\n\n        Set up creates\n        - 3 releases\n            - release1 - canary # env only matters for how we filter releases\n                - r1-proj1-canary # NOTE: is it possible to have a ReleaseProjectEnvironment without a corresponding ReleaseEnvironment??\n                - r1-proj2-canary\n            - release2 - prod # env only matters for how we filter releases\n                - r2-proj1-prod\n            - release3 - None\n        - 4 thresholds\n            - project1 canary error_counts\n            - project1 canary new_issues\n            - project1 prod error_counts\n            - project2 canary error_counts\n            - project3 no environment error_counts\n\n        so response should look like\n        {\n            {p1.slug}-{canary}-{release1.version}: [threshold1, threshold2]\n            {p1.slug}-{prod}-{release1.version}: [threshold]\n            {p2.slug}-{canary}-{release1.version}: [threshold]\n            {p1.slug}-{prod}-{release2.version}: [threshold, threshold]\n            {p1.slug}-{prod}-{release2.version}: [threshold]\n            {p1.slug}-None-{release3.version}: [threshold]\n        }\n        '
        now = str(datetime.now())
        yesterday = str(datetime.now() - timedelta(hours=24))
        last_week = str(datetime.now() - timedelta(days=7))
        release_old = Release.objects.create(version='old_version', organization=self.organization, date_added=last_week)
        response = self.get_success_response(self.organization.slug, start=yesterday, end=now)
        assert len(response.data.keys()) == 6
        for key in response.data.keys():
            assert release_old.version not in key
        data = response.data
        r1_keys = [k for (k, v) in data.items() if k.split('-')[2] == self.release1.version]
        assert len(r1_keys) == 3
        temp_key = f'{self.project1.slug}-{self.canary_environment.name}-{self.release1.version}'
        assert temp_key in r1_keys
        assert len(data[temp_key]) == 2
        temp_key = f'{self.project2.slug}-{self.canary_environment.name}-{self.release1.version}'
        assert temp_key in r1_keys
        assert len(data[temp_key]) == 1
        temp_key = f'{self.project1.slug}-{self.production_environment.name}-{self.release1.version}'
        assert temp_key in r1_keys
        assert len(data[temp_key]) == 1
        r2_keys = [k for (k, v) in data.items() if k.split('-')[2] == self.release2.version]
        assert len(r2_keys) == 2
        temp_key = f'{self.project1.slug}-{self.canary_environment.name}-{self.release2.version}'
        assert temp_key in r2_keys
        assert len(data[temp_key]) == 2
        temp_key = f'{self.project1.slug}-{self.production_environment.name}-{self.release2.version}'
        assert temp_key in r2_keys
        assert len(data[temp_key]) == 1
        r3_keys = [k for (k, v) in data.items() if k.split('-')[2] == self.release3.version]
        assert len(r3_keys) == 1
        temp_key = f'{self.project3.slug}-None-{self.release3.version}'
        assert temp_key in r3_keys
        assert len(data[temp_key]) == 1

    def test_get_success_environment_filter(self):
        if False:
            while True:
                i = 10
        "\n        Tests fetching thresholds within the past 24hrs filtered on environment\n\n        Set up creates\n        - 2 releases\n            - release1 - canary # env only matters for how we filter releases\n                - r1-proj1-canary\n                - r1-proj2-canary\n            - release2 - prod # env only matters for how we filter releases\n                - r2-proj1-prod\n        - 4 thresholds\n            - project1 canary error_counts\n            - project1 canary new_issues\n            - project1 prod error_counts\n            - project2 canary error_counts\n\n        We'll filter for _only_ canary releases, so the response should look like\n        {\n            {p1.slug}-{canary}-{release1.version}: [threshold1, threshold2]\n            {p2.slug}-{canary}-{release1.version}: [threshold]\n        }\n        "
        now = str(datetime.now())
        yesterday = str(datetime.now() - timedelta(hours=24))
        response = self.get_success_response(self.organization.slug, start=yesterday, end=now, environment=['canary'])
        assert len(response.data.keys()) == 2
        data = response.data
        r1_keys = [k for (k, v) in data.items() if k.split('-')[2] == self.release1.version]
        assert len(r1_keys) == 2
        temp_key = f'{self.project1.slug}-{self.canary_environment.name}-{self.release1.version}'
        assert temp_key in r1_keys
        assert len(data[temp_key]) == 2
        temp_key = f'{self.project2.slug}-{self.canary_environment.name}-{self.release1.version}'
        assert temp_key in r1_keys
        assert len(data[temp_key]) == 1

    def test_get_success_release_filter(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Tests fetching thresholds within the past 24hrs filtered on release versions\n\n        Set up creates\n        - 2 releases\n            - release1 - canary # env only matters for how we filter releases\n                - r1-proj1-canary\n                - r1-proj2-canary\n            - release2 - prod # env only matters for how we filter releases\n                - r2-proj1-prod\n        - 4 thresholds\n            - project1 canary error_counts\n            - project1 canary new_issues\n            - project1 prod error_counts\n            - project2 canary error_counts\n\n        We'll filter for _only_ release1, so the response should look like\n        {\n            {p1.slug}-{canary}-{release1.version}: [threshold1, threshold2]\n            {p1.slug}-{prod}-{release1.version}: [threshold]\n            {p2.slug}-{canary}-{release1.version}: [threshold]\n        }\n        "
        now = str(datetime.now())
        yesterday = str(datetime.now() - timedelta(hours=24))
        response = self.get_success_response(self.organization.slug, start=yesterday, end=now, release=[self.release1.version])
        assert len(response.data.keys()) == 3
        data = response.data
        r1_keys = [k for (k, v) in data.items() if k.split('-')[2] == self.release1.version]
        assert len(r1_keys) == 3
        temp_key = f'{self.project1.slug}-{self.canary_environment.name}-{self.release1.version}'
        assert temp_key in r1_keys
        assert len(data[temp_key]) == 2
        temp_key = f'{self.project2.slug}-{self.canary_environment.name}-{self.release1.version}'
        assert temp_key in r1_keys
        assert len(data[temp_key]) == 1
        temp_key = f'{self.project1.slug}-{self.production_environment.name}-{self.release1.version}'
        assert temp_key in r1_keys
        assert len(data[temp_key]) == 1
        r2_keys = [k for (k, v) in data.items() if k.split('-')[2] == self.release2.version]
        assert len(r2_keys) == 0

    def test_get_success_project_slug_filter(self):
        if False:
            print('Hello World!')
        "\n        Tests fetching thresholds within the past 24hrs filtered on project_slug's\n        NOTE: Because releases may have multiple projects, filtering by project is _not_ adequate to\n        return accurate release health\n        So - filtering on project will give us all the releases associated with that project\n        but we still need all the other projects associated with the release to determine health status\n\n        Set up creates\n        - 2 releases\n            - release1 - canary # env only matters for how we filter releases\n                - r1-proj1-canary\n                - r1-proj2-canary\n            - release2 - prod # env only matters for how we filter releases\n                - r2-proj1-prod\n        - 4 thresholds\n            - project1 canary error_counts\n            - project1 canary new_issues\n            - project1 prod error_counts\n            - project2 canary error_counts\n\n\n        We'll filter for _only_ project2, so the response should look like\n        since project2 was only ever added to release1\n        {\n            {p2.slug}-{canary}-{release1.version}: [threshold]\n        }\n        "
        now = str(datetime.now())
        yesterday = str(datetime.now() - timedelta(hours=24))
        response = self.get_success_response(self.organization.slug, start=yesterday, end=now, project=[self.project2.slug])
        assert len(response.data.keys()) == 1
        data = response.data
        r1_keys = [k for (k, v) in data.items() if k.split('-')[2] == self.release1.version]
        assert len(r1_keys) == 1
        temp_key = f'{self.project1.slug}-{self.canary_environment.name}-{self.release1.version}'
        assert temp_key not in r1_keys
        temp_key = f'{self.project2.slug}-{self.canary_environment.name}-{self.release1.version}'
        assert temp_key in r1_keys
        assert len(data[temp_key]) == 1
        temp_key = f'{self.project1.slug}-{self.production_environment.name}-{self.release1.version}'
        assert temp_key not in r1_keys
        r2_keys = [k for (k, v) in data.items() if k.split('-')[2] == self.release2.version]
        assert len(r2_keys) == 0

class ErrorCountThresholdCheckTest(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.project1 = self.create_project(name='foo', organization=self.organization)
        self.project2 = self.create_project(name='bar', organization=self.organization)
        self.canary_environment = Environment.objects.create(organization_id=self.organization.id, name='canary')
        self.release1 = Release.objects.create(version='v1', organization=self.organization)
        self.release1.add_project(self.project1)
        self.release1.add_project(self.project2)
        self.release2 = Release.objects.create(version='v2', organization=self.organization)
        self.release2.add_project(self.project1)

    def test_threshold_within_timeseries(self):
        if False:
            print('Hello World!')
        '\n        construct a timeseries with:\n        - a single release\n        - a single project\n        - no environment\n        - multiple timestamps both before and after our threshold window\n        '
        now = datetime.utcnow()
        timeseries = [{'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=3)).isoformat(), 'environment': None, 'count()': 1}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=2)).isoformat(), 'environment': None, 'count()': 1}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=1)).isoformat(), 'environment': None, 'count()': 1}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': now.isoformat(), 'environment': None, 'count()': 1}]
        current_threshold_healthy: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=1), 'end': now, 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.OVER_STR, 'value': 4, 'window_in_seconds': 60}
        assert is_error_count_healthy(ethreshold=current_threshold_healthy, timeseries=timeseries)
        threshold_at_limit_healthy: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=1), 'end': now, 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.OVER_STR, 'value': 1, 'window_in_seconds': 60}
        assert is_error_count_healthy(ethreshold=threshold_at_limit_healthy, timeseries=timeseries)
        past_threshold_healthy: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=2), 'end': now - timedelta(minutes=1), 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.OVER_STR, 'value': 2, 'window_in_seconds': 60}
        assert is_error_count_healthy(ethreshold=past_threshold_healthy, timeseries=timeseries)
        threshold_under_unhealthy: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=1), 'end': now, 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.UNDER_STR, 'value': 4, 'window_in_seconds': 60}
        assert not is_error_count_healthy(ethreshold=threshold_under_unhealthy, timeseries=timeseries)
        threshold_unfinished: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=1), 'end': now + timedelta(minutes=5), 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.OVER_STR, 'value': 4, 'window_in_seconds': 60}
        assert is_error_count_healthy(ethreshold=threshold_unfinished, timeseries=timeseries)

    def test_multiple_releases_within_timeseries(self):
        if False:
            while True:
                i = 10
        now = datetime.utcnow()
        timeseries = [{'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=3)).isoformat(), 'environment': None, 'count()': 1}, {'release': self.release2.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=3)).isoformat(), 'environment': None, 'count()': 2}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=2)).isoformat(), 'environment': None, 'count()': 1}, {'release': self.release2.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=2)).isoformat(), 'environment': None, 'count()': 2}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=1)).isoformat(), 'environment': None, 'count()': 1}, {'release': self.release2.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=1)).isoformat(), 'environment': None, 'count()': 2}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': now.isoformat(), 'environment': None, 'count()': 1}, {'release': self.release2.version, 'project_id': self.project1.id, 'time': now.isoformat(), 'environment': None, 'count()': 2}]
        threshold_healthy: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=1), 'end': now, 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.OVER_STR, 'value': 4, 'window_in_seconds': 60}
        assert is_error_count_healthy(ethreshold=threshold_healthy, timeseries=timeseries)
        threshold_unhealthy: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=1), 'end': now, 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release2.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.OVER_STR, 'value': 1, 'window_in_seconds': 60}
        assert not is_error_count_healthy(ethreshold=threshold_unhealthy, timeseries=timeseries)

    def test_multiple_projects_within_timeseries(self):
        if False:
            while True:
                i = 10
        now = datetime.utcnow()
        timeseries = [{'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=3)).isoformat(), 'environment': None, 'count()': 1}, {'release': self.release1.version, 'project_id': self.project2.id, 'time': (now - timedelta(minutes=3)).isoformat(), 'environment': None, 'count()': 2}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=2)).isoformat(), 'environment': None, 'count()': 1}, {'release': self.release1.version, 'project_id': self.project2.id, 'time': (now - timedelta(minutes=2)).isoformat(), 'environment': None, 'count()': 2}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=1)).isoformat(), 'environment': None, 'count()': 1}, {'release': self.release1.version, 'project_id': self.project2.id, 'time': (now - timedelta(minutes=1)).isoformat(), 'environment': None, 'count()': 2}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': now.isoformat(), 'environment': None, 'count()': 1}, {'release': self.release1.version, 'project_id': self.project2.id, 'time': now.isoformat(), 'environment': None, 'count()': 2}]
        threshold_healthy: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=1), 'end': now, 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.OVER_STR, 'value': 4, 'window_in_seconds': 60}
        assert is_error_count_healthy(ethreshold=threshold_healthy, timeseries=timeseries)
        threshold_unhealthy: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=1), 'end': now, 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project2), 'project_id': self.project2.id, 'project_slug': self.project2.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.OVER_STR, 'value': 1, 'window_in_seconds': 60}
        assert not is_error_count_healthy(ethreshold=threshold_unhealthy, timeseries=timeseries)

    def test_multiple_environments_within_timeseries(self):
        if False:
            return 10
        now = datetime.utcnow()
        timeseries = [{'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=3)).isoformat(), 'environment': None, 'count()': 1}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=3)).isoformat(), 'environment': 'canary', 'count()': 2}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=2)).isoformat(), 'environment': None, 'count()': 1}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=2)).isoformat(), 'environment': 'canary', 'count()': 2}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=1)).isoformat(), 'environment': None, 'count()': 1}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=1)).isoformat(), 'environment': 'canary', 'count()': 2}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': now.isoformat(), 'environment': None, 'count()': 1}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': now.isoformat(), 'environment': 'canary', 'count()': 2}]
        threshold_healthy: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=1), 'end': now, 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.OVER_STR, 'value': 2, 'window_in_seconds': 60}
        assert is_error_count_healthy(ethreshold=threshold_healthy, timeseries=timeseries)
        threshold_unhealthy: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=1), 'end': now, 'environment': serialize(self.canary_environment), 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.OVER_STR, 'value': 1, 'window_in_seconds': 60}
        assert not is_error_count_healthy(ethreshold=threshold_unhealthy, timeseries=timeseries)

    def test_unordered_timeseries(self):
        if False:
            return 10
        '\n        construct a timeseries with:\n        - a single release\n        - a single project\n        - no environment\n        - multiple timestamps both before and after our threshold window\n        - all disorganized\n        '
        now = datetime.utcnow()
        timeseries = [{'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=3)).isoformat(), 'environment': None, 'count()': 1}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': now.isoformat(), 'environment': None, 'count()': 1}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=1)).isoformat(), 'environment': None, 'count()': 1}, {'release': self.release1.version, 'project_id': self.project1.id, 'time': (now - timedelta(minutes=2)).isoformat(), 'environment': None, 'count()': 1}]
        current_threshold_healthy: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=1), 'end': now, 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.OVER_STR, 'value': 4, 'window_in_seconds': 60}
        assert is_error_count_healthy(ethreshold=current_threshold_healthy, timeseries=timeseries)
        threshold_at_limit_healthy: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=1), 'end': now, 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.OVER_STR, 'value': 1, 'window_in_seconds': 60}
        assert is_error_count_healthy(ethreshold=threshold_at_limit_healthy, timeseries=timeseries)
        past_threshold_healthy: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=2), 'end': now - timedelta(minutes=1), 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.OVER_STR, 'value': 2, 'window_in_seconds': 60}
        assert is_error_count_healthy(ethreshold=past_threshold_healthy, timeseries=timeseries)
        threshold_under_unhealthy: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=1), 'end': now, 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.UNDER_STR, 'value': 4, 'window_in_seconds': 60}
        assert not is_error_count_healthy(ethreshold=threshold_under_unhealthy, timeseries=timeseries)
        threshold_unfinished: EnrichedThreshold = {'date': now, 'start': now - timedelta(minutes=1), 'end': now + timedelta(minutes=5), 'environment': None, 'is_healthy': False, 'key': '', 'project': serialize(self.project1), 'project_id': self.project1.id, 'project_slug': self.project1.slug, 'release': self.release1.version, 'threshold_type': ReleaseThresholdType.TOTAL_ERROR_COUNT, 'trigger_type': TriggerType.OVER_STR, 'value': 4, 'window_in_seconds': 60}
        assert is_error_count_healthy(ethreshold=threshold_unfinished, timeseries=timeseries)