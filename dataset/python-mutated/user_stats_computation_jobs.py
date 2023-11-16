"""Jobs that are run by CRON scheduler."""
from __future__ import annotations
from core import feconf
from core.domain import user_services
from core.jobs import base_jobs
from core.jobs import job_utils
from core.jobs.io import ndb_io
from core.jobs.transforms import job_result_transforms
from core.jobs.types import job_run_result
from core.platform import models
import apache_beam as beam
from typing import Iterable
MYPY = False
if MYPY:
    from mypy_imports import datastore_services
    from mypy_imports import user_models
(user_models,) = models.Registry.import_models([models.Names.USER])
datastore_services = models.Registry.import_datastore_services()

class CollectWeeklyDashboardStatsJob(base_jobs.JobBase):
    """One-off job for populating weekly dashboard stats for all registered
    users who have a non-None value of UserStatsModel.
    """

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            while True:
                i = 10
        user_settings_models = self.pipeline | 'Get all UserSettingsModels' >> ndb_io.GetModels(user_models.UserSettingsModel.get_all())
        old_user_stats_models = self.pipeline | 'Get all UserStatsModels' >> ndb_io.GetModels(user_models.UserStatsModel.get_all())
        new_user_stats_models = (user_settings_models, old_user_stats_models) | 'Merge models' >> beam.Flatten() | 'Group models with same ID' >> beam.GroupBy(lambda m: m.id) | 'Get rid of key' >> beam.Values() | 'Filter pairs of models' >> beam.Filter(lambda models: len(list(models)) == 1 and isinstance(list(models)[0], user_models.UserSettingsModel)) | 'Transform tuples into models' >> beam.Map(lambda models: list(models)[0]) | 'Create new user stat models' >> beam.ParDo(CreateUserStatsModel())
        unused_put_result = (new_user_stats_models, old_user_stats_models) | 'Merge new and old models together' >> beam.Flatten() | 'Update the dashboard stats' >> beam.ParDo(UpdateWeeklyCreatorStats()) | 'Put models into the datastore' >> ndb_io.PutModels()
        new_user_stats_job_result = new_user_stats_models | 'Create new job run result' >> job_result_transforms.CountObjectsToJobRunResult('NEW MODELS')
        old_user_stats_job_result = old_user_stats_models | 'Create old job run result' >> job_result_transforms.CountObjectsToJobRunResult('OLD MODELS')
        return (new_user_stats_job_result, old_user_stats_job_result) | 'Merge new and old results together' >> beam.Flatten()

class CreateUserStatsModel(beam.DoFn):
    """DoFn to create empty user stats model."""

    def process(self, user_settings_model: user_models.UserSettingsModel) -> Iterable[user_models.UserStatsModel]:
        if False:
            print('Hello World!')
        'Creates empty user stats model with id.\n\n        Args:\n            user_settings_model: UserSettingsModel. Model from which to\n                create the user stats model.\n\n        Yields:\n            UserStatsModel. The created user stats model.\n        '
        with datastore_services.get_ndb_context():
            user_stats_model = user_models.UserStatsModel(id=user_settings_model.id)
        user_stats_model.update_timestamps()
        yield user_stats_model

class UpdateWeeklyCreatorStats(beam.DoFn):
    """DoFn to update weekly dashboard stats in the user stats model."""

    def process(self, user_stats_model: user_models.UserStatsModel) -> Iterable[user_models.UserStatsModel]:
        if False:
            i = 10
            return i + 15
        'Updates weekly dashboard stats with the current values.\n\n        Args:\n            user_stats_model: UserStatsModel. Model for which to update\n                the weekly dashboard stats.\n\n        Yields:\n            UserStatsModel. The updated user stats model.\n        '
        model = job_utils.clone_model(user_stats_model)
        schema_version = model.schema_version
        if schema_version != feconf.CURRENT_DASHBOARD_STATS_SCHEMA_VERSION:
            user_services.migrate_dashboard_stats_to_latest_schema(model)
        weekly_creator_stats = {user_services.get_current_date_as_string(): {'num_ratings': model.num_ratings or 0, 'average_ratings': model.average_ratings, 'total_plays': model.total_plays or 0}}
        model.weekly_creator_stats_list.append(weekly_creator_stats)
        model.update_timestamps()
        yield model