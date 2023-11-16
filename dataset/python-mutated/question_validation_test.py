"""Unit tests for jobs.transforms.question_validation."""
from __future__ import annotations
from core.jobs import job_test_utils
from core.jobs.decorators import validation_decorators
from core.jobs.transforms.validation import question_validation
from core.jobs.types import base_validation_errors
from core.platform import models
from core.tests import test_utils
import apache_beam as beam
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import question_models
(base_models, question_models) = models.Registry.import_models([models.Names.BASE_MODEL, models.Names.QUESTION])

class ValidateQuestionSnapshotMetadataModelTests(job_test_utils.PipelinedTestBase):

    def test_validate_change_domain_implemented(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        invalid_commit_cmd_model = question_models.QuestionSnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='commiter-id', commit_type='delete', commit_cmds=[{'cmd': base_models.VersionedModel.CMD_DELETE_COMMIT}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(question_validation.ValidateQuestionSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [])

    def test_change_dict_without_cmd(self) -> None:
        if False:
            return 10
        invalid_commit_cmd_model = question_models.QuestionSnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='commiter-id', commit_type='delete', commit_cmds=[{'invalid': 'data'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(question_validation.ValidateQuestionSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'invalid': 'data'}, 'Missing cmd key in change dict')])

    def test_change_dict_with_invalid_cmd(self) -> None:
        if False:
            while True:
                i = 10
        invalid_commit_cmd_model = question_models.QuestionSnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='commiter-id', commit_type='delete', commit_cmds=[{'cmd': 'invalid'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(question_validation.ValidateQuestionSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'invalid'}, 'Command invalid is not allowed')])

    def test_change_dict_with_missing_attributes_in_cmd(self) -> None:
        if False:
            while True:
                i = 10
        commit_dict = {'cmd': 'update_question_property', 'property_name': 'question_state_data', 'old_value': 'old_value'}
        invalid_commit_cmd_model = question_models.QuestionSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='commiter-id', commit_type='edit', commit_cmds=[commit_dict])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(question_validation.ValidateQuestionSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, commit_dict, 'The following required attributes are missing: new_value')])

    def test_change_dict_with_extra_attributes_in_cmd(self) -> None:
        if False:
            return 10
        invalid_commit_cmd_model = question_models.QuestionSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='commiter-id', commit_type='create', commit_cmds=[{'cmd': 'create_new', 'invalid': 'invalid'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(question_validation.ValidateQuestionSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'create_new', 'invalid': 'invalid'}, 'The following extra attributes are present: invalid')])

    def test_update_question_property_with_wrong_property_name(self) -> None:
        if False:
            i = 10
            return i + 15
        commit_dict = {'cmd': 'update_question_property', 'property_name': 'wrong', 'new_value': 'new_value', 'old_value': 'old_value'}
        invalid_commit_cmd_model = question_models.QuestionSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='commiter-id', commit_type='edit', commit_cmds=[commit_dict])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(question_validation.ValidateQuestionSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, commit_dict, 'Value for property_name in cmd update_question_property: wrong is not allowed')])

class RelationshipsOfTests(test_utils.TestBase):

    def test_question_skill_link_model_relationships(self) -> None:
        if False:
            while True:
                i = 10
        self.assertItemsEqual(validation_decorators.RelationshipsOf.get_model_kind_references('QuestionSkillLinkModel', 'id'), ['QuestionModel'])
        self.assertItemsEqual(validation_decorators.RelationshipsOf.get_model_kind_references('QuestionSkillLinkModel', 'skill_id'), ['SkillModel'])

    def test_question_commit_log_entry_model_relationships(self) -> None:
        if False:
            print('Hello World!')
        self.assertItemsEqual(validation_decorators.RelationshipsOf.get_model_kind_references('QuestionCommitLogEntryModel', 'question_id'), ['QuestionModel'])

    def test_question_summary_model_relationships(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertItemsEqual(validation_decorators.RelationshipsOf.get_model_kind_references('QuestionSummaryModel', 'id'), ['QuestionModel'])

class ValidateQuestionCommitLogEntryModelTests(job_test_utils.PipelinedTestBase):

    def test_validate_question_model(self) -> None:
        if False:
            return 10
        invalid_commit_cmd_model = question_models.QuestionCommitLogEntryModel(id='question_123', created_on=self.YEAR_AGO, last_updated=self.NOW, question_id='123', user_id='', commit_type='delete', post_commit_status='private', commit_cmds=[{'cmd': base_models.VersionedModel.CMD_DELETE_COMMIT}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(question_validation.ValidateQuestionCommitLogEntryModel())
        self.assert_pcoll_equal(output, [])

    def test_raises_commit_cmd_none_error(self) -> None:
        if False:
            while True:
                i = 10
        invalid_commit_cmd_model = question_models.QuestionCommitLogEntryModel(id='model_123', created_on=self.YEAR_AGO, last_updated=self.NOW, question_id='123', user_id='', commit_type='delete', post_commit_status='private', commit_cmds=[{'cmd': base_models.VersionedModel.CMD_DELETE_COMMIT}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(question_validation.ValidateQuestionCommitLogEntryModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsNoneError(invalid_commit_cmd_model)])