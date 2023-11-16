"""Jobs used for migrating the question models."""
from __future__ import annotations
import logging
from core import feconf
from core.domain import question_domain
from core.domain import question_fetchers
from core.domain import question_services
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
    from mypy_imports import question_models
(base_models, question_models) = models.Registry.import_models([models.Names.BASE_MODEL, models.Names.QUESTION])
datastore_services = models.Registry.import_datastore_services()

class PopulateQuestionSummaryVersionOneOffJob(base_jobs.JobBase):
    """Job that adds a version field to QuestionSummary models."""

    @staticmethod
    def _regenerate_question_summary(question_id: str, question_model: question_models.QuestionModel) -> result.Result[Tuple[str, question_models.QuestionSummaryModel], Tuple[str, Exception]]:
        if False:
            while True:
                i = 10
        'Validates question and regenerates the question summary model.\n\n        Args:\n            question_id: str. The id of the question.\n            question_model: QuestionModel. The question model.\n\n        Returns:\n            Result((str, QuestionSummaryModel), (str, Exception)). Result\n            containing tuple which consist of question ID and either question\n            summary model or Exception. Question summary model is returned when\n            the validation was successful and Exception is returned otherwise.\n        '
        try:
            with datastore_services.get_ndb_context():
                question = question_fetchers.get_question_from_model(question_model)
            question.validate()
        except Exception as e:
            logging.exception(e)
            return result.Err((question_id, e))
        question_summary = question_services.compute_summary_of_question(question)
        with datastore_services.get_ndb_context():
            question_summary_model = question_models.QuestionSummaryModel(id=question_summary.id, question_model_last_updated=question_summary.last_updated, question_model_created_on=question_summary.created_on, question_content=question_summary.question_content, misconception_ids=question_summary.misconception_ids, interaction_id=question_summary.interaction_id, version=question_summary.version)
        question_summary_model.update_timestamps()
        return result.Ok((question_id, question_summary_model))

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            i = 10
            return i + 15
        'Returns a PCollection of results from the question summary\n        migration.\n\n        Returns:\n            PCollection. A PCollection of results from the\n            question summary migration.\n        '
        all_question_models = self.pipeline | 'Get all non-deleted question models' >> ndb_io.GetModels(question_models.QuestionModel.get_all()) | 'Add question keys' >> beam.WithKeys(lambda model: model.id)
        question_summary_models = all_question_models | 'Regenerate question summaries' >> beam.MapTuple(self._regenerate_question_summary)
        regenerated_question_summary_results = question_summary_models | 'Generates results' >> job_result_transforms.ResultsToJobRunResults('QUESTION SUMMARY PROCESSED')
        question_summary_models_to_put = question_summary_models | 'Filter oks' >> beam.Filter(lambda result_item: result_item.is_ok()) | 'Unwrap ok' >> beam.Map(lambda result_item: result_item.unwrap()) | 'Get rid of ID' >> beam.Values()
        unused_put_results = question_summary_models_to_put | 'Put models into datastore' >> ndb_io.PutModels()
        return regenerated_question_summary_results

class AuditPopulateQuestionSummaryVersionOneOffJob(base_jobs.JobBase):
    """Job that audits PopulateQuestionSummaryVersionOneOffJob."""

    @staticmethod
    def _regenerate_question_summary(question_id: str, question_model: question_models.QuestionModel) -> result.Result[Tuple[str, question_models.QuestionSummaryModel], Tuple[str, Exception]]:
        if False:
            while True:
                i = 10
        'Validates question and regenerates the question summary model.\n\n        Args:\n            question_id: str. The id of the question.\n            question_model: QuestionModel. The question model.\n\n        Returns:\n            Result((str, QuestionSummaryModel), (str, Exception)). Result\n            containing tuple which consist of question ID and either question\n            ummary model or Exception. Question summary model is returned when\n            the validation was successful and Exception is returned otherwise.\n        '
        try:
            with datastore_services.get_ndb_context():
                question = question_fetchers.get_question_from_model(question_model)
            question.validate()
        except Exception as e:
            logging.exception(e)
            return result.Err((question_id, e))
        question_summary = question_services.compute_summary_of_question(question)
        with datastore_services.get_ndb_context():
            question_summary_model = question_models.QuestionSummaryModel(id=question_summary.id, question_model_last_updated=question_summary.last_updated, question_model_created_on=question_summary.created_on, question_content=question_summary.question_content, misconception_ids=question_summary.misconception_ids, interaction_id=question_summary.interaction_id, version=question_summary.version)
        question_summary_model.update_timestamps()
        return result.Ok((question_id, question_summary_model))

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            i = 10
            return i + 15
        'Returns a PCollection of results from the question migration.\n\n        Returns:\n            PCollection. A PCollection of results from the question\n            migration.\n        '
        all_question_models = self.pipeline | 'Get all non-deleted question models' >> ndb_io.GetModels(question_models.QuestionModel.get_all()) | 'Add question keys' >> beam.WithKeys(lambda model: model.id)
        question_summary_models = all_question_models | 'Regenerate question summaries' >> beam.MapTuple(self._regenerate_question_summary)
        regenerated_question_summary_results = question_summary_models | 'Generates results' >> job_result_transforms.ResultsToJobRunResults('QUESTION SUMMARY PROCESSED')
        unused_updated_question_summary = question_summary_models | 'Filter oks' >> beam.Filter(lambda result_item: result_item.is_ok()) | 'Unwrap ok' >> beam.Map(lambda result_item: result_item.unwrap()) | 'Get rid of ID' >> beam.Values()
        return regenerated_question_summary_results

class MigrateQuestionModels(beam.PTransform):
    """Transform that gets all Question models, performs migration
      and filters any error results.
    """

    @staticmethod
    def _migrate_question(question_id: str, question_model: question_models.QuestionModel) -> result.Result[Tuple[str, question_domain.Question], Tuple[str, Exception]]:
        if False:
            while True:
                i = 10
        'Migrates question and transform question model into question object.\n\n        Args:\n            question_id: str. The id of the question.\n            question_model: QuestionModel. The question model to migrate.\n\n        Returns:\n            Result((str, Question), (str, Exception)). Result containing tuple\n            which consist of question ID and either question object or\n            Exception. Question object is returned when the migration was\n            successful and Exception is returned otherwise.\n        '
        try:
            question = question_fetchers.get_question_from_model(question_model)
            question.validate()
        except Exception as e:
            logging.exception(e)
            return result.Err((question_id, e))
        return result.Ok((question_id, question))

    @staticmethod
    def _generate_question_changes(question_id: str, question_model: question_models.QuestionModel) -> Iterable[Tuple[str, question_domain.QuestionChange]]:
        if False:
            return 10
        'Generates question change objects. Question change object is\n        generated when schema version for some field is lower than the latest\n        schema version.\n\n        Args:\n            question_id: str. The ID of the question.\n            question_model: QuestionModel. The question for which to generate\n                the change objects.\n\n        Yields:\n            (str, QuestionChange). Tuple containing question ID and question\n            change object.\n        '
        schema_version = question_model.question_state_data_schema_version
        if schema_version < feconf.CURRENT_STATE_SCHEMA_VERSION:
            question_change = question_domain.QuestionChange({'cmd': question_domain.CMD_MIGRATE_STATE_SCHEMA_TO_LATEST_VERSION, 'from_version': schema_version, 'to_version': feconf.CURRENT_STATE_SCHEMA_VERSION})
            yield (question_id, question_change)

    def expand(self, pipeline: beam.Pipeline) -> Tuple[beam.PCollection[base_models.BaseModel], beam.PCollection[job_run_result.JobRunResult]]:
        if False:
            i = 10
            return i + 15
        'Migrate question objects and flush the input\n            in case of errors.\n\n        Args:\n            pipeline: Pipeline. Input beam pipeline.\n\n        Returns:\n            (PCollection, PCollection). Tuple containing\n            PCollection of models which should be put into the datastore and\n            a PCollection of results from the question migration.\n        '
        unmigrated_question_models = pipeline | 'Get all non-deleted question models' >> ndb_io.GetModels(question_models.QuestionModel.get_all()) | 'Add question keys' >> beam.WithKeys(lambda question_model: question_model.id)
        question_summary_models = self.pipeline | 'Get all non-deleted question summary models' >> ndb_io.GetModels(question_models.QuestionSummaryModel.get_all()) | 'Add question summary keys' >> beam.WithKeys(lambda question_summary_model: question_summary_model.id)
        all_migrated_question_results = unmigrated_question_models | 'Transform and migrate model' >> beam.MapTuple(self._migrate_question)
        migrated_question_job_run_results = all_migrated_question_results | 'Generates results for migration' >> job_result_transforms.ResultsToJobRunResults('QUESTION PROCESSED')
        filtered_migrated_exp = all_migrated_question_results | 'Filter migration results' >> results_transforms.DrainResultsOnError()
        migrated_questions = filtered_migrated_exp | 'Unwrap ok' >> beam.Map(lambda result_item: result_item.unwrap())
        question_changes = unmigrated_question_models | 'Generates question changes' >> beam.FlatMapTuple(self._generate_question_changes)
        question_objects_list = {'question_model': unmigrated_question_models, 'question_summary_model': question_summary_models, 'question': migrated_questions, 'question_changes': question_changes} | 'Merge objects' >> beam.CoGroupByKey() | 'Get rid of ID' >> beam.Values()
        transformed_question_objects_list = question_objects_list | 'Remove unmigrated questions' >> beam.Filter(lambda x: len(x['question_changes']) > 0 and len(x['question']) > 0) | 'Reorganize the question objects' >> beam.Map(lambda objects: {'question_model': objects['question_model'][0], 'question_summary_model': objects['question_summary_model'][0], 'question': objects['question'][0], 'question_changes': objects['question_changes']})
        already_migrated_job_run_results = question_objects_list | 'Remove migrated questions' >> beam.Filter(lambda x: len(x['question_changes']) == 0 and len(x['question']) > 0) | 'Transform already migrated question into job run results' >> job_result_transforms.CountObjectsToJobRunResult('QUESTION PREVIOUSLY MIGRATED')
        question_objects_list_job_run_results = transformed_question_objects_list | 'Transform question objects into job run results' >> job_result_transforms.CountObjectsToJobRunResult('QUESTION MIGRATED')
        job_run_results = (migrated_question_job_run_results, already_migrated_job_run_results, question_objects_list_job_run_results) | 'Flatten job run results' >> beam.Flatten()
        return (transformed_question_objects_list, job_run_results)

class MigrateQuestionJob(base_jobs.JobBase):
    """Job that migrates Question models."""

    @staticmethod
    def _update_question(question_model: question_models.QuestionModel, migrated_question: question_domain.Question, question_changes: Sequence[question_domain.QuestionChange]) -> Sequence[base_models.BaseModel]:
        if False:
            while True:
                i = 10
        'Generates newly updated question models.\n\n        Args:\n            question_model: QuestionModel. The question which to be updated.\n            migrated_question: Question. The migrated question domain object.\n            question_changes: QuestionChange. The question changes to apply.\n\n        Returns:\n            sequence(BaseModel). Sequence of models which should be put into\n            the datastore.\n        '
        updated_question_model = question_services.populate_question_model_fields(question_model, migrated_question)
        change_dicts = [change.to_dict() for change in question_changes]
        with datastore_services.get_ndb_context():
            models_to_put = updated_question_model.compute_models_to_commit(feconf.MIGRATION_BOT_USER_ID, feconf.COMMIT_TYPE_EDIT, 'Update state data contents schema version to %d.' % feconf.CURRENT_STATE_SCHEMA_VERSION, change_dicts, additional_models={})
        models_to_put_values = []
        for model in models_to_put.values():
            assert isinstance(model, base_models.BaseModel)
            models_to_put_values.append(model)
        datastore_services.update_timestamps_multi(list(models_to_put_values))
        return models_to_put_values

    @staticmethod
    def _update_question_summary(migrated_question: question_domain.Question, question_summary_model: question_models.QuestionSummaryModel) -> question_models.QuestionSummaryModel:
        if False:
            i = 10
            return i + 15
        'Generates newly updated question summary model.\n\n        Args:\n            migrated_question: Question. The migrated question domain object.\n            question_summary_model: QuestionSummaryModel. The question summary\n                model to update.\n\n        Returns:\n            QuestionSummaryModel. The updated question summary model to put\n            into the datastore.\n        '
        question_summary = question_services.compute_summary_of_question(migrated_question)
        question_summary.version += 1
        updated_question_summary_model = question_services.populate_question_summary_model_fields(question_summary_model, question_summary)
        return updated_question_summary_model

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            return 10
        'Returns a PCollection of results from the question migration.\n\n        Returns:\n            PCollection. A PCollection of results from the question\n            migration.\n        '
        (transformed_question_objects_list, job_run_results) = self.pipeline | 'Perform migration and filter migration results' >> MigrateQuestionModels()
        question_models_to_put = transformed_question_objects_list | 'Generate question models to put' >> beam.FlatMap(lambda question_objects: self._update_question(question_objects['question_model'], question_objects['question'], question_objects['question_changes']))
        question_summary_model_to_put = transformed_question_objects_list | 'Generate question summary to put' >> beam.Map(lambda question_objects: self._update_question_summary(question_objects['question'], question_objects['question_summary_model']))
        unused_put_results = (question_models_to_put, question_summary_model_to_put) | 'Merge models' >> beam.Flatten() | 'Put models into datastore' >> ndb_io.PutModels()
        return job_run_results

class AuditQuestionMigrationJob(base_jobs.JobBase):
    """Job that audits question migration."""

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a PCollection of results from the audit of question\n        migration.\n\n        Returns:\n            PCollection. A PCollection of results from the question\n            migration.\n        '
        (unused_transformed_question_objects_list, job_run_results) = self.pipeline | 'Perform migration and filter migration results' >> MigrateQuestionModels()
        return job_run_results