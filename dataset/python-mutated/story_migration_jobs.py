"""Jobs used for migrating the story models."""
from __future__ import annotations
import logging
from core import feconf
from core.domain import story_domain
from core.domain import story_fetchers
from core.domain import story_services
from core.domain import topic_domain
from core.domain import topic_fetchers
from core.jobs import base_jobs
from core.jobs.io import ndb_io
from core.jobs.transforms import job_result_transforms
from core.jobs.transforms import results_transforms
from core.jobs.types import job_run_result
from core.platform import models
import apache_beam as beam
import result
from typing import Dict, Iterable, Optional, Sequence, Tuple
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import datastore_services
    from mypy_imports import story_models
    from mypy_imports import topic_models
(base_models, story_models, topic_models) = models.Registry.import_models([models.Names.BASE_MODEL, models.Names.STORY, models.Names.TOPIC])
datastore_services = models.Registry.import_datastore_services()

class MigrateStoryModels(beam.PTransform):
    """Transform that gets all Story models, performs migration and filters any
    error results.
    """

    @staticmethod
    def _migrate_story(story_id: str, story_model: story_models.StoryModel, topic_id_to_topic: Optional[Dict[str, topic_domain.Topic]]=None) -> result.Result[Tuple[str, story_domain.Story], Tuple[str, Exception]]:
        if False:
            while True:
                i = 10
        'Migrates story and transform story model into story object.\n\n        Args:\n            story_id: str. The id of the story.\n            story_model: StoryModel. The story model to migrate.\n            topic_id_to_topic: dict(str, Topic). The mapping from topic ID to\n                topic.\n\n        Returns:\n            Result((str, Story), (str, Exception)). Result containing tuple that\n            consists of story ID and either story object or Exception. Story\n            object is returned when the migration was successful and Exception\n            is returned otherwise.\n        '
        try:
            story = story_fetchers.get_story_from_model(story_model)
            story.validate()
            assert topic_id_to_topic is not None
            corresponding_topic = topic_id_to_topic[story.corresponding_topic_id]
            story_services.validate_prerequisite_skills_in_story_contents(corresponding_topic.get_all_skill_ids(), story.story_contents)
        except Exception as e:
            logging.exception(e)
            return result.Err((story_id, e))
        return result.Ok((story_id, story))

    @staticmethod
    def _generate_story_changes(story_id: str, story_model: story_models.StoryModel) -> Iterable[Tuple[str, story_domain.StoryChange]]:
        if False:
            return 10
        'Generates story change objects. Story change object is generated when\n        schema version for some field is lower than the latest schema version.\n\n        Args:\n            story_id: str. The id of the story.\n            story_model: StoryModel. The story for which to generate the change\n                objects.\n\n        Yields:\n            (str, StoryChange). Tuple containing story ID and story change\n            object.\n        '
        schema_version = story_model.story_contents_schema_version
        if schema_version < feconf.CURRENT_STORY_CONTENTS_SCHEMA_VERSION:
            story_change = story_domain.StoryChange({'cmd': story_domain.CMD_MIGRATE_SCHEMA_TO_LATEST_VERSION, 'from_version': story_model.story_contents_schema_version, 'to_version': feconf.CURRENT_STORY_CONTENTS_SCHEMA_VERSION})
            yield (story_id, story_change)

    def expand(self, pipeline: beam.Pipeline) -> Tuple[beam.PCollection[base_models.BaseModel], beam.PCollection[job_run_result.JobRunResult]]:
        if False:
            while True:
                i = 10
        'Migrate story objects and flush the input in case of errors.\n\n        Args:\n            pipeline: Pipeline. Input beam pipeline.\n\n        Returns:\n            (PCollection, PCollection). Tuple containing\n            PCollection of models which should be put into the datastore and\n            a PCollection of results from the story migration.\n        '
        unmigrated_story_models = pipeline | 'Get all non-deleted story models' >> ndb_io.GetModels(story_models.StoryModel.get_all()) | 'Add story keys' >> beam.WithKeys(lambda story_model: story_model.id)
        story_summary_models = pipeline | 'Get all non-deleted story summary models' >> ndb_io.GetModels(story_models.StorySummaryModel.get_all()) | 'Add story summary keys' >> beam.WithKeys(lambda story_summary_model: story_summary_model.id)
        topics = self.pipeline | 'Get all non-deleted topic models' >> ndb_io.GetModels(topic_models.TopicModel.get_all()) | 'Transform model into domain object' >> beam.Map(topic_fetchers.get_topic_from_model) | 'Add topic keys' >> beam.WithKeys(lambda topic: topic.id)
        topic_id_to_topic = beam.pvalue.AsDict(topics)
        all_migrated_story_results = unmigrated_story_models | 'Transform and migrate model' >> beam.MapTuple(self._migrate_story, topic_id_to_topic=topic_id_to_topic)
        migrated_story_job_run_results = all_migrated_story_results | 'Generate results for migration' >> job_result_transforms.ResultsToJobRunResults('STORY PROCESSED')
        filtered_migrated_stories = all_migrated_story_results | 'Filter migration results' >> results_transforms.DrainResultsOnError()
        migrated_stories = filtered_migrated_stories | 'Unwrap ok' >> beam.Map(lambda result_item: result_item.unwrap())
        story_changes = unmigrated_story_models | 'Generate story changes' >> beam.FlatMapTuple(self._generate_story_changes)
        story_objects_list = {'story_model': unmigrated_story_models, 'story_summary_model': story_summary_models, 'story': migrated_stories, 'story_change': story_changes} | 'Merge objects' >> beam.CoGroupByKey() | 'Get rid of ID' >> beam.Values()
        transformed_story_objects_list = story_objects_list | 'Remove unmigrated stories' >> beam.Filter(lambda x: len(x['story_change']) > 0 and len(x['story']) > 0) | 'Reorganize the story objects' >> beam.Map(lambda objects: {'story_model': objects['story_model'][0], 'story_summary_model': objects['story_summary_model'][0], 'story': objects['story'][0], 'story_change': objects['story_change'][0]})
        already_migrated_job_run_results = story_objects_list | 'Remove migrated stories' >> beam.Filter(lambda x: len(x['story_change']) == 0 and len(x['story']) > 0) | 'Transform previously migrated stories into job run results' >> job_result_transforms.CountObjectsToJobRunResult('STORY PREVIOUSLY MIGRATED')
        story_objects_list_job_run_results = transformed_story_objects_list | 'Transform story objects into job run results' >> job_result_transforms.CountObjectsToJobRunResult('STORY MIGRATED')
        job_run_results = (migrated_story_job_run_results, already_migrated_job_run_results, story_objects_list_job_run_results) | 'Flatten job run results' >> beam.Flatten()
        return (transformed_story_objects_list, job_run_results)

class MigrateStoryJob(base_jobs.JobBase):
    """Job that migrates story models."""

    @staticmethod
    def _update_story(story_model: story_models.StoryModel, migrated_story: story_domain.Story, story_change: story_domain.StoryChange) -> Sequence[base_models.BaseModel]:
        if False:
            while True:
                i = 10
        'Generates newly updated story models.\n\n        Args:\n            story_model: StoryModel. The story which should be updated.\n            migrated_story: Story. The migrated story domain object.\n            story_change: StoryChange. The story change to apply.\n\n        Returns:\n            sequence(BaseModel). Sequence of models which should be put into\n            the datastore.\n        '
        updated_story_model = story_services.populate_story_model_fields(story_model, migrated_story)
        change_dicts = [story_change.to_dict()]
        with datastore_services.get_ndb_context():
            models_to_put = updated_story_model.compute_models_to_commit(feconf.MIGRATION_BOT_USERNAME, feconf.COMMIT_TYPE_EDIT, 'Update story contents schema version to %d.' % feconf.CURRENT_STORY_CONTENTS_SCHEMA_VERSION, change_dicts, additional_models={})
        models_to_put_values = []
        for model in models_to_put.values():
            assert isinstance(model, base_models.BaseModel)
            models_to_put_values.append(model)
        datastore_services.update_timestamps_multi(models_to_put_values)
        return models_to_put_values

    @staticmethod
    def _update_story_summary(migrated_story: story_domain.Story, story_summary_model: story_models.StorySummaryModel) -> story_models.StorySummaryModel:
        if False:
            return 10
        'Generates newly updated story summary model.\n\n        Args:\n            migrated_story: Story. The migrated story domain object.\n            story_summary_model: StorySummaryModel. The story summary model to\n                update.\n\n        Returns:\n            StorySummaryModel. The updated story summary model to put into the\n            datastore.\n        '
        story_summary = story_services.compute_summary_of_story(migrated_story)
        story_summary.version += 1
        updated_story_summary_model = story_services.populate_story_summary_model_fields(story_summary_model, story_summary)
        return updated_story_summary_model

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            while True:
                i = 10
        'Returns a PCollection of results from the story migration.\n\n        Returns:\n            PCollection. A PCollection of results from the story migration.\n        '
        (transformed_story_objects_list, job_run_results) = self.pipeline | 'Perform migration and filter migration results' >> MigrateStoryModels()
        story_models_to_put = transformed_story_objects_list | 'Generate story models to put' >> beam.FlatMap(lambda story_objects: self._update_story(story_objects['story_model'], story_objects['story'], story_objects['story_change']))
        story_summary_models_to_put = transformed_story_objects_list | 'Generate story summary models to put' >> beam.Map(lambda story_objects: self._update_story_summary(story_objects['story'], story_objects['story_summary_model']))
        unused_put_results = (story_models_to_put, story_summary_models_to_put) | 'Merge models' >> beam.Flatten() | 'Put models into the datastore' >> ndb_io.PutModels()
        return job_run_results

class AuditStoryMigrationJob(base_jobs.JobBase):
    """Job that audits migrated Story models."""

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            print('Hello World!')
        'Returns a PCollection of results from the audit of story migration.\n\n        Returns:\n            PCollection. A PCollection of results from the story migration.\n        '
        (unused_transformed_story_objects_list, job_run_results) = self.pipeline | 'Perform migration and filter migration results' >> MigrateStoryModels()
        return job_run_results