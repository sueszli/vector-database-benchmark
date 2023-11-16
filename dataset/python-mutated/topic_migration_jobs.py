"""Jobs used for migrating the topic models."""
from __future__ import annotations
import logging
from core import feconf
from core.domain import topic_domain
from core.domain import topic_fetchers
from core.domain import topic_services
from core.jobs import base_jobs
from core.jobs.io import ndb_io
from core.jobs.transforms import job_result_transforms
from core.jobs.transforms import results_transforms
from core.jobs.types import job_run_result
from core.platform import models
import apache_beam as beam
import result
from typing import Iterable, Sequence, Tuple
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import datastore_services
    from mypy_imports import topic_models
(base_models, topic_models) = models.Registry.import_models([models.Names.BASE_MODEL, models.Names.TOPIC])
datastore_services = models.Registry.import_datastore_services()

class MigrateTopicModels(beam.PTransform):
    """Transform that gets all Topic models, performs migration
      and filters any error results.
    """

    @staticmethod
    def _migrate_topic(topic_id: str, topic_model: topic_models.TopicModel) -> result.Result[Tuple[str, topic_domain.Topic], Tuple[str, Exception]]:
        if False:
            i = 10
            return i + 15
        'Migrates topic and transform topic model into topic object.\n\n        Args:\n            topic_id: str. The id of the topic.\n            topic_model: TopicModel. The topic model to migrate.\n\n        Returns:\n            Result((str, Topic), (str, Exception)). Result containing tuple that\n            consist of topic ID and either topic object or Exception. Topic\n            object is returned when the migration was successful and Exception\n            is returned otherwise.\n        '
        try:
            topic = topic_fetchers.get_topic_from_model(topic_model)
            topic.validate()
        except Exception as e:
            logging.exception(e)
            return result.Err((topic_id, e))
        return result.Ok((topic_id, topic))

    @staticmethod
    def _generate_topic_changes(topic_id: str, topic_model: topic_models.TopicModel) -> Iterable[Tuple[str, topic_domain.TopicChange]]:
        if False:
            i = 10
            return i + 15
        'Generates topic change objects. Topic change object is generated when\n        schema version for some field is lower than the latest schema version.\n\n        Args:\n            topic_id: str. The ID of the topic.\n            topic_model: TopicModel. The topic for which to generate the change\n                objects.\n\n        Yields:\n            (str, TopicChange). Tuple containing Topic ID and topic change\n            object.\n        '
        subtopic_version = topic_model.subtopic_schema_version
        if subtopic_version < feconf.CURRENT_SUBTOPIC_SCHEMA_VERSION:
            topic_change = topic_domain.TopicChange({'cmd': topic_domain.CMD_MIGRATE_SUBTOPIC_SCHEMA_TO_LATEST_VERSION, 'from_version': subtopic_version, 'to_version': feconf.CURRENT_SUBTOPIC_SCHEMA_VERSION})
            yield (topic_id, topic_change)
        story_version = topic_model.story_reference_schema_version
        if story_version < feconf.CURRENT_STORY_REFERENCE_SCHEMA_VERSION:
            topic_change = topic_domain.TopicChange({'cmd': topic_domain.CMD_MIGRATE_STORY_REFERENCE_SCHEMA_TO_LATEST_VERSION, 'from_version': story_version, 'to_version': feconf.CURRENT_STORY_REFERENCE_SCHEMA_VERSION})
            yield (topic_id, topic_change)

    def expand(self, pipeline: beam.Pipeline) -> Tuple[beam.PCollection[base_models.BaseModel], beam.PCollection[job_run_result.JobRunResult]]:
        if False:
            while True:
                i = 10
        'Migrate topic objects and flush the input\n            in case of errors.\n\n        Args:\n            pipeline: Pipeline. Input beam pipeline.\n\n        Returns:\n            (PCollection, PCollection). Tuple containing\n            PCollection of models which should be put into the datastore and\n            a PCollection of results from the topic migration.\n        '
        unmigrated_topic_models = pipeline | 'Get all non-deleted topic models' >> ndb_io.GetModels(topic_models.TopicModel.get_all()) | 'Add topic keys' >> beam.WithKeys(lambda topic_model: topic_model.id)
        topic_summary_models = self.pipeline | 'Get all non-deleted topic summary models' >> ndb_io.GetModels(topic_models.TopicSummaryModel.get_all()) | 'Add topic summary keys' >> beam.WithKeys(lambda topic_summary_model: topic_summary_model.id)
        all_migrated_topic_results = unmigrated_topic_models | 'Transform and migrate model' >> beam.MapTuple(self._migrate_topic)
        migrated_topic_job_run_results = all_migrated_topic_results | 'Generates results for migration' >> job_result_transforms.ResultsToJobRunResults('TOPIC PROCESSED')
        filtered_migrated_exp = all_migrated_topic_results | 'Filter migration results' >> results_transforms.DrainResultsOnError()
        migrated_topics = filtered_migrated_exp | 'Unwrap ok' >> beam.Map(lambda result_item: result_item.unwrap())
        topic_changes = unmigrated_topic_models | 'Generates topic changes' >> beam.FlatMapTuple(self._generate_topic_changes)
        topic_objects_list = {'topic_model': unmigrated_topic_models, 'topic_summary_model': topic_summary_models, 'topic': migrated_topics, 'topic_changes': topic_changes} | 'Merge objects' >> beam.CoGroupByKey() | 'Get rid of ID' >> beam.Values()
        transformed_topic_objects_list = topic_objects_list | 'Remove unmigrated topics' >> beam.Filter(lambda x: len(x['topic_changes']) > 0 and len(x['topic']) > 0) | 'Reorganize the topic objects' >> beam.Map(lambda objects: {'topic_model': objects['topic_model'][0], 'topic_summary_model': objects['topic_summary_model'][0], 'topic': objects['topic'][0], 'topic_changes': objects['topic_changes']})
        already_migrated_job_run_results = topic_objects_list | 'Remove migrated jobs' >> beam.Filter(lambda x: len(x['topic_changes']) == 0 and len(x['topic']) > 0) | 'Transform previously migrated topics into job run results' >> job_result_transforms.CountObjectsToJobRunResult('TOPIC PREVIOUSLY MIGRATED')
        topic_objects_list_job_run_results = transformed_topic_objects_list | 'Transform topic objects into job run results' >> job_result_transforms.CountObjectsToJobRunResult('TOPIC MIGRATED')
        job_run_results = (migrated_topic_job_run_results, already_migrated_job_run_results, topic_objects_list_job_run_results) | 'Flatten job run results' >> beam.Flatten()
        return (transformed_topic_objects_list, job_run_results)

class MigrateTopicJob(base_jobs.JobBase):
    """Job that migrates Topic models."""

    @staticmethod
    def _update_topic(topic_model: topic_models.TopicModel, migrated_topic: topic_domain.Topic, topic_changes: Sequence[topic_domain.TopicChange]) -> Sequence[base_models.BaseModel]:
        if False:
            for i in range(10):
                print('nop')
        'Generates newly updated topic models.\n\n        Args:\n            topic_model: TopicModel. The topic which should be updated.\n            migrated_topic: Topic. The migrated topic domain object.\n            topic_changes: TopicChange. The topic changes to apply.\n\n        Returns:\n            sequence(BaseModel). Sequence of models which should be put into\n            the datastore.\n        '
        updated_topic_model = topic_services.populate_topic_model_fields(topic_model, migrated_topic)
        topic_rights_model = topic_models.TopicRightsModel.get(migrated_topic.id)
        change_dicts = [change.to_dict() for change in topic_changes]
        with datastore_services.get_ndb_context():
            models_to_put = updated_topic_model.compute_models_to_commit(feconf.MIGRATION_BOT_USER_ID, feconf.COMMIT_TYPE_EDIT, 'Update subtopic contents schema version to %d.' % feconf.CURRENT_SUBTOPIC_SCHEMA_VERSION, change_dicts, additional_models={'rights_model': topic_rights_model})
        models_to_put_values = []
        for model in models_to_put.values():
            assert isinstance(model, base_models.BaseModel)
            models_to_put_values.append(model)
        datastore_services.update_timestamps_multi(list(models_to_put_values))
        return models_to_put_values

    @staticmethod
    def _update_topic_summary(migrated_topic: topic_domain.Topic, topic_summary_model: topic_models.TopicSummaryModel) -> topic_models.TopicSummaryModel:
        if False:
            print('Hello World!')
        'Generates newly updated topic summary model.\n\n        Args:\n            migrated_topic: Topic. The migrated topic domain object.\n            topic_summary_model: TopicSummaryModel. The topic summary model to\n                update.\n\n        Returns:\n            TopicSummaryModel. The updated topic summary model to put into the\n            datastore.\n        '
        topic_summary = topic_services.compute_summary_of_topic(migrated_topic)
        topic_summary.version += 1
        updated_topic_summary_model = topic_services.populate_topic_summary_model_fields(topic_summary_model, topic_summary)
        return updated_topic_summary_model

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            while True:
                i = 10
        'Returns a PCollection of results from the topic migration.\n\n        Returns:\n            PCollection. A PCollection of results from the topic\n            migration.\n        '
        (transformed_topic_objects_list, job_run_results) = self.pipeline | 'Perform migration and filter migration results' >> MigrateTopicModels()
        topic_models_to_put = transformed_topic_objects_list | 'Generate topic models to put' >> beam.FlatMap(lambda topic_objects: self._update_topic(topic_objects['topic_model'], topic_objects['topic'], topic_objects['topic_changes']))
        topic_summary_model_to_put = transformed_topic_objects_list | 'Generate topic summary to put' >> beam.Map(lambda topic_objects: self._update_topic_summary(topic_objects['topic'], topic_objects['topic_summary_model']))
        unused_put_results = (topic_models_to_put, topic_summary_model_to_put) | 'Merge models' >> beam.Flatten() | 'Put models into datastore' >> ndb_io.PutModels()
        return job_run_results

class AuditTopicMigrateJob(base_jobs.JobBase):
    """Job that migrates Topic models."""

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            i = 10
            return i + 15
        'Returns a PCollection of results from the audit of topic\n        migration.\n\n        Returns:\n            PCollection. A PCollection of results from the topic\n            migration.\n        '
        (unused_transformed_topic_objects_list, job_run_results) = self.pipeline | 'Perform migration and filter migration results' >> MigrateTopicModels()
        return job_run_results