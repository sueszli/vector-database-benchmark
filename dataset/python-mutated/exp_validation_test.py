"""Unit tests for jobs.transforms.exp_validation."""
from __future__ import annotations
from core.domain import rights_domain
from core.jobs import job_test_utils
from core.jobs.decorators import validation_decorators
from core.jobs.transforms.validation import exp_validation
from core.jobs.types import base_validation_errors
from core.platform import models
from core.tests import test_utils
import apache_beam as beam
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import exp_models
(base_models, exp_models) = models.Registry.import_models([models.Names.BASE_MODEL, models.Names.EXPLORATION])

class ValidateExplorationSnapshotMetadataModelTests(job_test_utils.PipelinedTestBase):

    def test_validate_change_domain_implemented(self) -> None:
        if False:
            while True:
                i = 10
        invalid_commit_cmd_model = exp_models.ExplorationSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds_user_ids=['commit_cmds_user_1_id', 'commit_cmds_user_2_id'], content_user_ids=['content_user_1_id', 'content_user_2_id'], commit_cmds=[{'cmd': base_models.VersionedModel.CMD_DELETE_COMMIT}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [])

    def test_validate_exp_model_object_with_missing_cmd(self) -> None:
        if False:
            i = 10
            return i + 15
        invalid_commit_cmd_model = exp_models.ExplorationSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds_user_ids=['commit_cmds_user_1_id', 'commit_cmds_user_2_id'], content_user_ids=['content_user_1_id', 'content_user_2_id'], commit_cmds=[{'invalid': 'data'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'invalid': 'data'}, 'Missing cmd key in change dict')])

    def test_validate_exp_model_object_with_invalid_cmd(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        invalid_commit_cmd_model = exp_models.ExplorationSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds_user_ids=['commit_cmds_user_1_id', 'commit_cmds_user_2_id'], content_user_ids=['content_user_1_id', 'content_user_2_id'], commit_cmds=[{'cmd': 'invalid'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'invalid'}, 'Command invalid is not allowed')])

    def test_validate_exp_model_object_with_missing_attribute_in_cmd(self) -> None:
        if False:
            print('Hello World!')
        invalid_commit_cmd_model = exp_models.ExplorationSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds_user_ids=['commit_cmds_user_1_id', 'commit_cmds_user_2_id'], content_user_ids=['content_user_1_id', 'content_user_2_id'], commit_cmds=[{'cmd': 'edit_state_property', 'property_name': 'content', 'old_value': 'old_value'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'edit_state_property', 'property_name': 'content', 'old_value': 'old_value'}, 'The following required attributes are missing: new_value, state_name')])

    def test_validate_exp_model_object_with_extra_attribute_in_cmd(self) -> None:
        if False:
            i = 10
            return i + 15
        invalid_commit_cmd_model = exp_models.ExplorationSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds_user_ids=['commit_cmds_user_1_id', 'commit_cmds_user_2_id'], content_user_ids=['content_user_1_id', 'content_user_2_id'], commit_cmds=[{'cmd': 'rename_state', 'old_state_name': 'old_state_name', 'new_state_name': 'new_state_name', 'invalid': 'invalid'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'rename_state', 'old_state_name': 'old_state_name', 'new_state_name': 'new_state_name', 'invalid': 'invalid'}, 'The following extra attributes are present: invalid')])

    def test_validate_exp_model_object_with_invalid_exploration_property(self) -> None:
        if False:
            while True:
                i = 10
        invalid_commit_cmd_model = exp_models.ExplorationSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds_user_ids=['commit_cmds_user_1_id', 'commit_cmds_user_2_id'], content_user_ids=['content_user_1_id', 'content_user_2_id'], commit_cmds=[{'cmd': 'edit_exploration_property', 'property_name': 'invalid', 'old_value': 'old_value', 'new_value': 'new_value'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'edit_exploration_property', 'property_name': 'invalid', 'old_value': 'old_value', 'new_value': 'new_value'}, 'Value for property_name in cmd edit_exploration_property: invalid is not allowed')])

    def test_validate_exp_model_object_with_invalid_state_property(self) -> None:
        if False:
            return 10
        invalid_commit_cmd_model = exp_models.ExplorationSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds_user_ids=['commit_cmds_user_1_id', 'commit_cmds_user_2_id'], content_user_ids=['content_user_1_id', 'content_user_2_id'], commit_cmds=[{'cmd': 'edit_state_property', 'state_name': 'state_name', 'property_name': 'invalid', 'old_value': 'old_value', 'new_value': 'new_value'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'edit_state_property', 'state_name': 'state_name', 'property_name': 'invalid', 'old_value': 'old_value', 'new_value': 'new_value'}, 'Value for property_name in cmd edit_state_property: invalid is not allowed')])

class RelationshipsOfTests(test_utils.TestBase):

    def test_exploration_context_model_relationships(self) -> None:
        if False:
            print('Hello World!')
        self.assertItemsEqual(validation_decorators.RelationshipsOf.get_model_kind_references('ExplorationContextModel', 'story_id'), ['StoryModel'])
        self.assertItemsEqual(validation_decorators.RelationshipsOf.get_model_kind_references('ExplorationContextModel', 'id'), ['ExplorationModel'])

    def test_exp_summary_model_relationships(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertItemsEqual(validation_decorators.RelationshipsOf.get_model_kind_references('ExpSummaryModel', 'id'), ['ExplorationRightsModel', 'ExplorationModel'])

class ValidateExplorationRightsSnapshotMetadataModelTests(job_test_utils.PipelinedTestBase):

    def test_exploration_rights_change_object_with_missing_cmd(self) -> None:
        if False:
            while True:
                i = 10
        invalid_commit_cmd_model = exp_models.ExplorationRightsSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds=[{'invalid': 'data'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationRightsSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'invalid': 'data'}, 'Missing cmd key in change dict')])

    def test_exploration_rights_change_object_with_invalid_cmd(self) -> None:
        if False:
            print('Hello World!')
        invalid_commit_cmd_model = exp_models.ExplorationRightsSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds=[{'cmd': 'invalid'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationRightsSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'invalid'}, 'Command invalid is not allowed')])

    def test_exploration_rights_change_object_with_missing_attribute_in_cmd(self) -> None:
        if False:
            while True:
                i = 10
        invalid_commit_cmd_model = exp_models.ExplorationRightsSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds=[{'cmd': 'change_role', 'assignee_id': 'assignee_id'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationRightsSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'change_role', 'assignee_id': 'assignee_id'}, 'The following required attributes are missing: new_role, old_role')])

    def test_exploration_rights_change_object_with_extra_attribute_in_cmd(self) -> None:
        if False:
            i = 10
            return i + 15
        invalid_commit_cmd_model = exp_models.ExplorationRightsSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds=[{'cmd': 'change_private_viewability', 'old_viewable_if_private': 'old_viewable_if_private', 'new_viewable_if_private': 'new_viewable_if_private', 'invalid': 'invalid'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationRightsSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'change_private_viewability', 'old_viewable_if_private': 'old_viewable_if_private', 'new_viewable_if_private': 'new_viewable_if_private', 'invalid': 'invalid'}, 'The following extra attributes are present: invalid')])

    def test_exploration_rights_change_object_with_invalid_role(self) -> None:
        if False:
            i = 10
            return i + 15
        invalid_commit_cmd_model = exp_models.ExplorationRightsSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds=[{'cmd': 'change_role', 'assignee_id': 'assignee_id', 'old_role': rights_domain.ROLE_OWNER, 'new_role': 'invalid'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationRightsSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'change_role', 'assignee_id': 'assignee_id', 'old_role': rights_domain.ROLE_OWNER, 'new_role': 'invalid'}, 'Value for new_role in cmd change_role: invalid is not allowed')])

    def test_exploration_rights_change_object_with_invalid_status(self) -> None:
        if False:
            while True:
                i = 10
        invalid_commit_cmd_model = exp_models.ExplorationRightsSnapshotMetadataModel(id='model_id-1', created_on=self.YEAR_AGO, last_updated=self.NOW, committer_id='committer_id', commit_type='create', commit_cmds=[{'cmd': 'change_exploration_status', 'old_status': rights_domain.ACTIVITY_STATUS_PRIVATE, 'new_status': 'invalid'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationRightsSnapshotMetadataModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsValidateError(invalid_commit_cmd_model, {'cmd': 'change_exploration_status', 'old_status': rights_domain.ACTIVITY_STATUS_PRIVATE, 'new_status': 'invalid'}, 'Value for new_status in cmd change_exploration_status: invalid is not allowed')])

class ValidateExplorationCommitLogEntryModelTests(job_test_utils.PipelinedTestBase):

    def test_validate_rights_model(self) -> None:
        if False:
            i = 10
            return i + 15
        invalid_commit_cmd_model = exp_models.ExplorationCommitLogEntryModel(id='rights_id123', created_on=self.YEAR_AGO, last_updated=self.NOW, commit_type='test-type', user_id='', exploration_id='123', post_commit_status='private', commit_cmds=[{'cmd': 'create_new'}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationCommitLogEntryModel())
        self.assert_pcoll_equal(output, [])

    def test_validate_exploration_model(self) -> None:
        if False:
            i = 10
            return i + 15
        invalid_commit_cmd_model = exp_models.ExplorationCommitLogEntryModel(id='exploration_id123', created_on=self.YEAR_AGO, last_updated=self.NOW, commit_type='test-type', user_id='', exploration_id='123', post_commit_status='private', commit_cmds=[{'cmd': base_models.VersionedModel.CMD_DELETE_COMMIT}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationCommitLogEntryModel())
        self.assert_pcoll_equal(output, [])

    def test_raises_commit_cmd_none_error(self) -> None:
        if False:
            while True:
                i = 10
        invalid_commit_cmd_model = exp_models.ExplorationCommitLogEntryModel(id='model_id123', created_on=self.YEAR_AGO, last_updated=self.NOW, commit_type='test-type', user_id='', exploration_id='123', post_commit_status='private', commit_cmds=[{'cmd': base_models.VersionedModel.CMD_DELETE_COMMIT}])
        output = self.pipeline | beam.Create([invalid_commit_cmd_model]) | beam.ParDo(exp_validation.ValidateExplorationCommitLogEntryModel())
        self.assert_pcoll_equal(output, [base_validation_errors.CommitCmdsNoneError(invalid_commit_cmd_model)])