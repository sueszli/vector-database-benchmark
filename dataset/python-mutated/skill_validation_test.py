"""Unit tests for jobs.transforms.skill_validation."""
from __future__ import annotations
from core.jobs import job_test_utils
from core.jobs.transforms.validation import skill_validation
from core.jobs.types import base_validation_errors
from core.platform import models
import apache_beam as beam
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import skill_models
(base_models, skill_models) = models.Registry.import_models([models.Names.BASE_MODEL, models.Names.SKILL])

class ValidateSkillSnapshotMetadataModelTests(job_test_utils.PipelinedTestBase):

    def test_validate_change_domain_implemented(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        valid_commit_cmd_model = skill_models.SkillSnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer-id', commit_type='delete', commit_cmds=[{'cmd': base_models.VersionedModel.CMD_DELETE_COMMIT}])
        output = self.pipeline | beam.Create([valid_commit_cmd_model]) | beam.ParDo(skill_validation.ValidateSkillSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [])

    def test_skill_change_object_with_missing_cmd(self) -> None:
        if False:
            i = 10
            return i + 15
        invalid_commit_cmd_model = skill_models.SkillSnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer-id', commit_type='delete', commit_cmds=[{'invalid': 'data'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(skill_validation.ValidateSkillSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'invalid': 'data'}, 'Missing cmd key in change dict')])

    def test_skill_change_object_with_invalid_cmd(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        invalid_commit_cmd_model = skill_models.SkillSnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer-id', commit_type='delete', commit_cmds=[{'cmd': 'invalid'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(skill_validation.ValidateSkillSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'invalid'}, 'Command invalid is not allowed')])

    def test_skill_change_object_with_missing_attribute_in_cmd(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        commit_dict = {'cmd': 'update_skill_property', 'property_name': 'name'}
        invalid_commit_cmd_model = skill_models.SkillSnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer-id', commit_type='edit', commit_cmds=[commit_dict])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(skill_validation.ValidateSkillSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, commit_dict, 'The following required attributes are missing: new_value, old_value')])

    def test_skill_change_object_with_extra_attribute_in_cmd(self) -> None:
        if False:
            i = 10
            return i + 15
        commit_dict = {'cmd': 'add_skill_misconception', 'new_misconception_dict': "{u'id': 0, u'notes': u'<p>notes</p>', u'feedback': u'<p>default_feedback</p>', u'name': u'name'}", 'invalid': 'invalid'}
        invalid_commit_cmd_model = skill_models.SkillSnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer-id', commit_type='create', commit_cmds=[commit_dict])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(skill_validation.ValidateSkillSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, commit_dict, 'The following extra attributes are present: invalid')])

    def test_skill_change_object_with_invalid_skill_property(self) -> None:
        if False:
            return 10
        commit_dict = {'cmd': 'update_skill_property', 'property_name': 'invalid', 'old_value': 'old_value', 'new_value': 'new_value'}
        invalid_commit_cmd_model = skill_models.SkillSnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer-id', commit_type='edit', commit_cmds=[commit_dict])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(skill_validation.ValidateSkillSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, commit_dict, 'Value for property_name in cmd update_skill_property: invalid is not allowed')])

    def test_skill_change_object_with_invalid_skill_misconceptions(self) -> None:
        if False:
            i = 10
            return i + 15
        commit_dict = {'cmd': 'update_skill_misconceptions_property', 'misconception_id': 'id', 'property_name': 'invalid', 'old_value': 'old_value', 'new_value': 'new_value'}
        invalid_commit_cmd_model = skill_models.SkillSnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer-id', commit_type='create', commit_cmds_user_ids=['commit_cmds_user_1_id', 'commit_cmds_user_2_id'], content_user_ids=['content_user_1_id', 'content_user_2_id'], commit_cmds=[commit_dict])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(skill_validation.ValidateSkillSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, commit_dict, 'Value for property_name in cmd update_skill_misconceptions_property: invalid is not allowed')])

    def test_skill_change_object_with_invalid_skill_contents_property(self) -> None:
        if False:
            print('Hello World!')
        commit_dict = {'cmd': 'update_skill_contents_property', 'property_name': 'invalid', 'old_value': 'old_value', 'new_value': 'new_value'}
        invalid_commit_cmd_model = skill_models.SkillSnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer-id', commit_type='create', commit_cmds_user_ids=['commit_cmds_user_1_id', 'commit_cmds_user_2_id'], content_user_ids=['content_user_1_id', 'content_user_2_id'], commit_cmds=[commit_dict])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(skill_validation.ValidateSkillSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, commit_dict, 'Value for property_name in cmd update_skill_contents_property: invalid is not allowed')])

class ValidateSkillCommitLogEntryModelTests(job_test_utils.PipelinedTestBase):

    def test_validate_skill_model(self) -> None:
        if False:
            i = 10
            return i + 15
        valid_commit_cmd_model = skill_models.SkillCommitLogEntryModel(id='skill_id123', created_on=self.YEAR_AGO, last_updated=self.NOW, skill_id='skill-id', user_id='user-id', commit_type='test-type', post_commit_status='private', commit_cmds=[{'cmd': 'create_new'}])
        output = self.pipeline | beam.Create([valid_commit_cmd_model]) | beam.ParDo(skill_validation.ValidateSkillCommitLogEntryModel())
        self.assert_pcoll_equal(output, [])

    def test_raises_commit_cmd_none_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        invalid_commit_cmd_model = skill_models.SkillCommitLogEntryModel(id='model_id123', created_on=self.YEAR_AGO, last_updated=self.NOW, skill_id='skill-id', user_id='user-id', commit_type='test-type', post_commit_status='private', commit_cmds=[{'cmd': 'create_new'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(skill_validation.ValidateSkillCommitLogEntryModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsNoneError(invalid_commit_cmd_model)])