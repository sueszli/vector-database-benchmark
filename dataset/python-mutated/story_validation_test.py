"""Unit tests for jobs.transforms.story_validation."""
from __future__ import annotations
from core.jobs import job_test_utils
from core.jobs.transforms.validation import story_validation
from core.jobs.types import base_validation_errors
from core.platform import models
import apache_beam as beam
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import story_models
(base_models, story_models) = models.Registry.import_models([models.Names.BASE_MODEL, models.Names.STORY])

class ValidateStorySnapshotMetadataModelTests(job_test_utils.PipelinedTestBase):

    def test_validate_change_domain_implemented(self) -> None:
        if False:
            while True:
                i = 10
        valid_commit_cmd_model = story_models.StorySnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='delete', commit_cmds=[{'cmd': base_models.VersionedModel.CMD_DELETE_COMMIT}])
        output = self.pipeline | beam.Create([valid_commit_cmd_model]) | beam.ParDo(story_validation.ValidateStorySnapshotMetadataModel())
        self.assert_pcoll_equal(output, [])

    def test_story_change_object_with_missing_cmd(self) -> None:
        if False:
            while True:
                i = 10
        invalid_commit_cmd_model = story_models.StorySnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds=[{'invalid': 'data'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(story_validation.ValidateStorySnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'invalid': 'data'}, 'Missing cmd key in change dict')])

    def test_story_change_object_with_invalid_cmd(self) -> None:
        if False:
            print('Hello World!')
        invalid_commit_cmd_model = story_models.StorySnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds=[{'invalid': 'data'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(story_validation.ValidateStorySnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'invalid': 'data'}, 'Missing cmd key in change dict')])

    def test_story_change_object_with_missing_attribute_in_cmd(self) -> None:
        if False:
            return 10
        commit_dict = {'cmd': 'update_story_property', 'property_name': 'title'}
        invalid_commit_cmd_model = story_models.StorySnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds=[commit_dict])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(story_validation.ValidateStorySnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, commit_dict, 'The following required attributes are missing: new_value, old_value')])

    def test_story_change_object_with_extra_attribute_in_cmd(self) -> None:
        if False:
            while True:
                i = 10
        invalid_commit_cmd_model = story_models.StorySnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds=[{'cmd': 'add_story_node', 'node_id': 'node_id', 'invalid': 'invalid'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(story_validation.ValidateStorySnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'add_story_node', 'node_id': 'node_id', 'invalid': 'invalid'}, 'The following required attributes are missing: title, The following extra attributes are present: invalid')])

    def test_story_change_object_with_invalid_story_property(self) -> None:
        if False:
            print('Hello World!')
        commit_dict = {'cmd': 'update_story_property', 'property_name': 'invalid', 'old_value': 'old_value', 'new_value': 'new_value'}
        invalid_commit_cmd_model = story_models.StorySnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='edit', commit_cmds=[commit_dict])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(story_validation.ValidateStorySnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, commit_dict, 'Value for property_name in cmd update_story_property: invalid is not allowed')])

    def test_story_change_object_with_invalid_story_node_property(self) -> None:
        if False:
            i = 10
            return i + 15
        invalid_commit_cmd_model = story_models.StorySnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='edit', commit_cmds=[{'cmd': 'update_story_node_property', 'node_id': 'node_id', 'property_name': 'invalid', 'old_value': 'old_value', 'new_value': 'new_value'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(story_validation.ValidateStorySnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'update_story_node_property', 'node_id': 'node_id', 'property_name': 'invalid', 'old_value': 'old_value', 'new_value': 'new_value'}, 'Value for property_name in cmd update_story_node_property: invalid is not allowed')])

    def test_story_change_object_with_invalid_story_contents_property(self) -> None:
        if False:
            while True:
                i = 10
        commit_dict = {'cmd': 'update_story_contents_property', 'property_name': 'invalid', 'old_value': 'old_value', 'new_value': 'new_value'}
        invalid_commit_cmd_model = story_models.StorySnapshotMetadataModel(id='123', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='edit', commit_cmds=[commit_dict])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(story_validation.ValidateStorySnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, commit_dict, 'Value for property_name in cmd update_story_contents_property: invalid is not allowed')])

class ValidateStoryCommitLogEntryModelTests(job_test_utils.PipelinedTestBase):

    def test_validate_story_model(self) -> None:
        if False:
            while True:
                i = 10
        valid_commit_cmd_model = story_models.StoryCommitLogEntryModel(id='story_id123', created_on=self.YEAR_AGO, last_updated=self.NOW, story_id='story-id', user_id='user-id', commit_type='test-type', post_commit_status='private', commit_cmds=[{'cmd': base_models.VersionedModel.CMD_DELETE_COMMIT}])
        output = self.pipeline | beam.Create([valid_commit_cmd_model]) | beam.ParDo(story_validation.ValidateStoryCommitLogEntryModel())
        self.assert_pcoll_equal(output, [])

    def test_raises_commit_cmd_none_error(self) -> None:
        if False:
            print('Hello World!')
        invalid_commit_cmd_model = story_models.StoryCommitLogEntryModel(id='model_id123', created_on=self.YEAR_AGO, last_updated=self.NOW, story_id='story-id', user_id='user-id', commit_type='test-type', post_commit_status='private', commit_cmds=[{'cmd': base_models.VersionedModel.CMD_DELETE_COMMIT}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(story_validation.ValidateStoryCommitLogEntryModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsNoneError(invalid_commit_cmd_model)])