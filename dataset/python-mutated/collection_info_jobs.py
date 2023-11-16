"""Jobs that extract Collection models information."""
from __future__ import annotations
from core.jobs import base_jobs
from core.jobs.io import ndb_io
from core.jobs.transforms import job_result_transforms
from core.jobs.types import job_run_result
from core.platform import models
import apache_beam as beam
from typing import Iterable, Tuple
MYPY = False
if MYPY:
    from mypy_imports import collection_models
    from mypy_imports import feedback_models
    from mypy_imports import user_models
(collection_models, feedback_models, user_models) = models.Registry.import_models([models.Names.COLLECTION, models.Names.FEEDBACK, models.Names.USER])

class GetCollectionOwnersEmailsJob(base_jobs.JobBase):
    """Job that extracts collection id and user email from datastore."""

    @staticmethod
    def _extract_user_and_collection_ids(collection_rights_model: collection_models.CollectionRightsModel) -> Iterable[Tuple[str, str]]:
        if False:
            print('Hello World!')
        'Extracts user id and collection id.\n\n        Args:\n            collection_rights_model: datastore_services.Model.\n                The collection rights model to extract user id and\n                collection id from.\n\n        Yields:\n            (str,str). Tuple containing user id and collection id.\n        '
        for user_id in collection_rights_model.owner_ids:
            yield (user_id, collection_rights_model.id)

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            return 10
        collection_pairs = self.pipeline | 'get collection models ' >> ndb_io.GetModels(collection_models.CollectionRightsModel.get_all()) | 'Flatten owner_ids and format' >> beam.FlatMap(self._extract_user_and_collection_ids)
        user_pairs = self.pipeline | 'Get all user settings models' >> ndb_io.GetModels(user_models.UserSettingsModel.get_all()) | 'Extract id and email' >> beam.Map(lambda user_setting: (user_setting.id, user_setting.email))
        collection_ids_to_email_mapping = (collection_pairs, user_pairs) | 'Group by user_id' >> beam.CoGroupByKey() | 'Drop user id' >> beam.Values() | 'Filter out results without any collection' >> beam.Filter(lambda collection_ids_and_email: len(collection_ids_and_email[0]) > 0)
        return collection_ids_to_email_mapping | 'Get final result' >> beam.MapTuple(lambda collection, email: job_run_result.JobRunResult.as_stdout('collection_ids: %s, email: %s' % (collection, email)))

class MatchEntityTypeCollectionJob(base_jobs.JobBase):
    """Job that match entity_type as collection."""

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            print('Hello World!')
        "Returns a PCollection of 'SUCCESS' or 'FAILURE' results from\n        matching entity_type as collection.\n\n        Returns:\n            PCollection. A PCollection of 'SUCCESS' or 'FAILURE' results from\n            matching entity_type as collection.\n        "
        feedback_model_matched_as_collection = self.pipeline | 'Get all GeneralFeedbackThread models' >> ndb_io.GetModels(feedback_models.GeneralFeedbackThreadModel.get_all()) | 'Extract entity_type' >> beam.Map(lambda feeback_model: feeback_model.entity_type) | 'Match entity_type' >> beam.Filter(lambda entity_type: entity_type == 'collection')
        return feedback_model_matched_as_collection | 'Count the output' >> job_result_transforms.CountObjectsToJobRunResult()