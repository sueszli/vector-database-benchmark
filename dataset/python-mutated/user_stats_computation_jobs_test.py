"""Unit tests for jobs.batch_jobs.user_stats_computation_jobs."""
from __future__ import annotations
import datetime
from core import feconf
from core.jobs import job_test_utils
from core.jobs.batch_jobs import user_stats_computation_jobs
from core.jobs.types import job_run_result
from core.platform import models
from typing import Final, Type
MYPY = False
if MYPY:
    from mypy_imports import user_models
(user_models,) = models.Registry.import_models([models.Names.USER])

class CollectWeeklyDashboardStatsJobTests(job_test_utils.JobTestBase):
    JOB_CLASS: Type[user_stats_computation_jobs.CollectWeeklyDashboardStatsJob] = user_stats_computation_jobs.CollectWeeklyDashboardStatsJob
    VALID_USER_ID_1: Final = 'uid_%s' % ('a' * feconf.USER_ID_RANDOM_PART_LENGTH)
    VALID_USER_ID_2: Final = 'uid_%s' % ('b' * feconf.USER_ID_RANDOM_PART_LENGTH)

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.formated_datetime = datetime.datetime.utcnow().strftime(feconf.DASHBOARD_STATS_DATETIME_STRING_FORMAT)

    def test_empty_storage(self) -> None:
        if False:
            return 10
        self.assert_job_output_is_empty()

    def test_updates_existing_stats_model_when_no_values_are_provided(self) -> None:
        if False:
            while True:
                i = 10
        user_settings_model = self.create_model(user_models.UserSettingsModel, id=self.VALID_USER_ID_1, email='a@a.com')
        user_stats_model = self.create_model(user_models.UserStatsModel, id=self.VALID_USER_ID_1)
        self.put_multi([user_settings_model, user_stats_model])
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='OLD MODELS SUCCESS: 1')])
        new_user_stats_model = user_models.UserStatsModel.get(self.VALID_USER_ID_1)
        assert new_user_stats_model is not None
        self.assertEqual(new_user_stats_model.weekly_creator_stats_list, [{self.formated_datetime: {'num_ratings': 0, 'average_ratings': None, 'total_plays': 0}}])

    def test_fails_when_existing_stats_has_wrong_schema_version(self) -> None:
        if False:
            print('Hello World!')
        user_settings_model = self.create_model(user_models.UserSettingsModel, id=self.VALID_USER_ID_1, email='a@a.com')
        user_stats_model = self.create_model(user_models.UserStatsModel, id=self.VALID_USER_ID_1, schema_version=0)
        self.put_multi([user_settings_model, user_stats_model])
        with self.assertRaisesRegex(Exception, 'Sorry, we can only process v1-v%d dashboard stats schemas at present.' % feconf.CURRENT_DASHBOARD_STATS_SCHEMA_VERSION):
            self.assert_job_output_is([job_run_result.JobRunResult(stdout='OLD MODELS SUCCESS: 1')])
        new_user_stats_model = user_models.UserStatsModel.get(self.VALID_USER_ID_1)
        assert new_user_stats_model is not None
        self.assertEqual(new_user_stats_model.weekly_creator_stats_list, [])

    def test_updates_existing_stats_model_when_values_are_provided(self) -> None:
        if False:
            while True:
                i = 10
        user_settings_model = self.create_model(user_models.UserSettingsModel, id=self.VALID_USER_ID_1, email='a@a.com')
        user_stats_model = self.create_model(user_models.UserStatsModel, id=self.VALID_USER_ID_1, num_ratings=10, average_ratings=4.5, total_plays=22)
        self.put_multi([user_settings_model, user_stats_model])
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='OLD MODELS SUCCESS: 1')])
        new_user_stats_model = user_models.UserStatsModel.get(self.VALID_USER_ID_1)
        assert new_user_stats_model is not None
        self.assertEqual(new_user_stats_model.weekly_creator_stats_list, [{self.formated_datetime: {'num_ratings': 10, 'average_ratings': 4.5, 'total_plays': 22}}])

    def test_creates_new_stats_model_if_not_existing(self) -> None:
        if False:
            return 10
        user_settings_model = self.create_model(user_models.UserSettingsModel, id=self.VALID_USER_ID_1, email='a@a.com')
        user_settings_model.update_timestamps()
        user_settings_model.put()
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='NEW MODELS SUCCESS: 1')])
        user_stats_model = user_models.UserStatsModel.get(self.VALID_USER_ID_1)
        assert user_stats_model is not None
        self.assertEqual(user_stats_model.weekly_creator_stats_list, [{self.formated_datetime: {'num_ratings': 0, 'average_ratings': None, 'total_plays': 0}}])

    def test_handles_multiple_models(self) -> None:
        if False:
            return 10
        user_settings_model_1 = self.create_model(user_models.UserSettingsModel, id=self.VALID_USER_ID_1, email='a@a.com')
        user_settings_model_2 = self.create_model(user_models.UserSettingsModel, id=self.VALID_USER_ID_2, email='b@b.com')
        user_stats_model_1 = self.create_model(user_models.UserStatsModel, id=self.VALID_USER_ID_1)
        self.put_multi([user_settings_model_1, user_settings_model_2, user_stats_model_1])
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='OLD MODELS SUCCESS: 1'), job_run_result.JobRunResult(stdout='NEW MODELS SUCCESS: 1')])
        user_stats_model = user_models.UserStatsModel.get(self.VALID_USER_ID_2)
        self.assertIsNotNone(user_stats_model)