from __future__ import annotations
from datetime import datetime
from unittest.mock import patch
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import region_silo_test

@region_silo_test(stable=True)
class SampleScheduleDataTest(APITestCase):
    endpoint = 'sentry-api-0-organization-monitors-schedule-sample-data'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.login_as(self.user)

    @patch('django.utils.timezone.now')
    def test_simple_crontab(self, mock_now):
        if False:
            return 10
        mock_now.return_value = datetime(2023, 10, 26, 12, 32)
        expected_ticks = [int(datetime(2023, 10, 26, 13, 0).timestamp()), int(datetime(2023, 10, 26, 14, 0).timestamp()), int(datetime(2023, 10, 26, 15, 0).timestamp()), int(datetime(2023, 10, 26, 16, 0).timestamp()), int(datetime(2023, 10, 26, 17, 0).timestamp())]
        response = self.get_success_response(self.organization.slug, qs_params={'num_ticks': 5, 'schedule_type': 'crontab', 'schedule': '0 * * * *'})
        assert response.data == expected_ticks

    @patch('django.utils.timezone.now')
    def test_simple_interval(self, mock_now):
        if False:
            for i in range(10):
                print('nop')
        mock_now.return_value = datetime(2023, 10, 26, 12, 32)
        expected_ticks = [int(datetime(2023, 10, 26, 12, 0).timestamp()), int(datetime(2023, 10, 26, 13, 0).timestamp()), int(datetime(2023, 10, 26, 14, 0).timestamp()), int(datetime(2023, 10, 26, 15, 0).timestamp()), int(datetime(2023, 10, 26, 16, 0).timestamp())]
        response = self.get_success_response(self.organization.slug, qs_params={'num_ticks': 5, 'schedule_type': 'interval', 'schedule': [1, 'hour']})
        assert response.data == expected_ticks

    def test_missing_params(self):
        if False:
            i = 10
            return i + 15
        self.get_error_response(self.organization.slug, qs_params={}, status_code=400)
        self.get_error_response(self.organization.slug, qs_params={'schedule_type': 'crontab', 'schedule': '0 * * * *'}, status_code=400)
        self.get_error_response(self.organization.slug, qs_params={'schedule_type': 'interval', 'schedule': '* * * * *'}, status_code=400)
        self.get_error_response(self.organization.slug, qs_params={'schedule_type': 'crontab'}, status_code=400)

    def test_bad_params(self):
        if False:
            print('Hello World!')
        self.get_error_response(self.organization.slug, qs_params={'num_ticks': 5, 'schedule_type': 'crontab', 'schedule': '0 * * *'}, status_code=400)
        self.get_error_response(self.organization.slug, qs_params={'numTicks': 5, 'scheduleType': 'interval', 'intervalUnit': 'second', 'intervalFrequency': '1'}, status_code=400)
        self.get_error_response(self.organization.slug, qs_params={'numTicks': 5, 'scheduleType': 'interval', 'intervalUnit': 'month', 'intervalFrequency': '-1'}, status_code=400)