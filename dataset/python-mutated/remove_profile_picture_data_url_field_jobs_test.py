"""Remove profile_picture_data_url field from UserSettingsModel."""
from __future__ import annotations
from core import feconf
from core.domain import user_services
from core.jobs import job_test_utils
from core.jobs.batch_jobs import remove_profile_picture_data_url_field_jobs
from core.jobs.types import job_run_result
from core.platform import models
MYPY = False
if MYPY:
    from mypy_imports import datastore_services
    from mypy_imports import user_models
(user_models,) = models.Registry.import_models([models.Names.USER])
datastore_services = models.Registry.import_datastore_services()

class MockUserSettingsModelWithProfilePicture(user_models.UserSettingsModel):
    """Mock UserSettingsModel so that it allows to set
    profile_picture_data_url.
    """
    profile_picture_data_url = datastore_services.TextProperty(default=None, indexed=False)

class RemoveProfilePictureFieldJobTests(job_test_utils.JobTestBase):
    """Tests for remove_profile_picture_data_url_field_jobs."""
    JOB_CLASS = remove_profile_picture_data_url_field_jobs.RemoveProfilePictureFieldJob

    def test_run_with_no_models(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_job_output_is([])

    def test_model_without_profile_picture_field_works(self) -> None:
        if False:
            while True:
                i = 10
        user_1 = self.create_model(user_models.UserSettingsModel, id='test_id_1', email='test_1@example.com', username='test_1', roles=[feconf.ROLE_ID_FULL_USER, feconf.ROLE_ID_CURRICULUM_ADMIN])
        self.assertNotIn('profile_picture_data_url', user_1.to_dict())
        self.assertNotIn('profile_picture_data_url', user_1._values)
        self.assertNotIn('profile_picture_data_url', user_1._properties)
        self.put_multi([user_1])
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='USER MODELS ITERATED OR UPDATED SUCCESS: 1')])
        migrated_setting_model = user_models.UserSettingsModel.get_by_id(user_1.id)
        self.assertNotIn('profile_picture_data_url', migrated_setting_model.to_dict())
        self.assertNotIn('profile_picture_data_url', migrated_setting_model._values)
        self.assertNotIn('profile_picture_data_url', migrated_setting_model._properties)

    def test_removal_of_profile_field(self) -> None:
        if False:
            while True:
                i = 10
        with self.swap(user_models, 'UserSettingsModel', MockUserSettingsModelWithProfilePicture):
            user_1 = self.create_model(user_models.UserSettingsModel, id='test_id_1', email='test_1@example.com', username='test_1', roles=[feconf.ROLE_ID_FULL_USER, feconf.ROLE_ID_CURRICULUM_ADMIN], profile_picture_data_url=user_services.DEFAULT_IDENTICON_DATA_URL)
            user_2 = self.create_model(user_models.UserSettingsModel, id='test_id_2', email='test_2@example.com', username='test_2', roles=[feconf.ROLE_ID_FULL_USER, feconf.ROLE_ID_CURRICULUM_ADMIN], profile_picture_data_url=None)
            self.put_multi([user_1, user_2])
            self.assertIn('profile_picture_data_url', user_1._values)
            self.assertIn('profile_picture_data_url', user_1._properties)
            self.assertIn('profile_picture_data_url', user_2._values)
            self.assertIn('profile_picture_data_url', user_2._properties)
            self.assert_job_output_is([job_run_result.JobRunResult(stdout='USER MODELS ITERATED OR UPDATED SUCCESS: 2')])
            migrated_setting_model_user_1 = user_models.UserSettingsModel.get_by_id(user_1.id)
            self.assertNotIn('profile_picture_data_url', migrated_setting_model_user_1.to_dict())
            self.assertNotIn('profile_picture_data_url', migrated_setting_model_user_1._values)
            self.assertNotIn('profile_picture_data_url', migrated_setting_model_user_1._properties)
            migrated_setting_model_user_2 = user_models.UserSettingsModel.get_by_id(user_2.id)
            self.assertNotIn('profile_picture_data_url', migrated_setting_model_user_2.to_dict())
            self.assertNotIn('profile_picture_data_url', migrated_setting_model_user_2._values)
            self.assertNotIn('profile_picture_data_url', migrated_setting_model_user_2._properties)