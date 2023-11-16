"""Remove profile_picture_data_url field from UserSettingsModel."""
from __future__ import annotations
from core.jobs import base_jobs
from core.jobs.io import ndb_io
from core.jobs.transforms import job_result_transforms
from core.jobs.types import job_run_result
from core.platform import models
import apache_beam as beam
MYPY = False
if MYPY:
    from mypy_imports import user_models
(user_models,) = models.Registry.import_models([models.Names.USER])

class RemoveProfilePictureFieldJob(base_jobs.JobBase):
    """Remove profile_picture_data_url from UserSettingsModel."""

    def _remove_profile_field(self, user_model: user_models.UserSettingsModel) -> user_models.UserSettingsModel:
        if False:
            for i in range(10):
                print('nop')
        'Remove profile_picture_data_url field from the model.\n\n        Args:\n            user_model: UserSettingsModel. The user settings model.\n\n        Returns:\n            user_model: UserSettingsModel. The updated user settings model.\n        '
        if 'profile_picture_data_url' in user_model._properties:
            del user_model._properties['profile_picture_data_url']
        return user_model

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            i = 10
            return i + 15
        users_with_updated_fields = self.pipeline | 'Get all non-deleted UserSettingsModel' >> ndb_io.GetModels(user_models.UserSettingsModel.get_all(include_deleted=True)) | 'Remove the profile_picture_data_url field' >> beam.Map(self._remove_profile_field)
        count_user_models_updated = users_with_updated_fields | 'Total count for user models' >> job_result_transforms.CountObjectsToJobRunResult('USER MODELS ITERATED OR UPDATED')
        unused_put_results = users_with_updated_fields | 'Put models into the datastore' >> ndb_io.PutModels()
        return count_user_models_updated