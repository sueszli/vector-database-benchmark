"""Tests for classifier services."""
from __future__ import annotations
import copy
import datetime
import json
import os
from core import feconf
from core import utils
from core.domain import classifier_domain
from core.domain import classifier_services
from core.domain import exp_domain
from core.domain import exp_fetchers
from core.domain import exp_services
from core.domain import fs_services
from core.domain import state_domain
from core.domain import translation_domain
from core.platform import models
from core.tests import test_utils
from proto_files import text_classifier_pb2
from typing import Dict, List, Optional, Tuple
MYPY = False
if MYPY:
    from mypy_imports import classifier_models
    from mypy_imports import datastore_services
datastore_services = models.Registry.import_datastore_services()
secrets_services = models.Registry.import_secrets_services()
(classifier_models,) = models.Registry.import_models([models.Names.CLASSIFIER])

class ClassifierServicesTests(test_utils.ClassifierTestBase):
    """Test "classify" using the sample explorations.

    Since the end to end tests cover correct classification, and frontend tests
    test hard rules, ReaderClassifyTests is only checking that the string
    classifier is actually called.
    """

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        self._init_classify_inputs('16')

    def _init_classify_inputs(self, exploration_id: str) -> None:
        if False:
            print('Hello World!')
        'Initializes all the classification inputs of the exploration\n        corresponding to the given exploration id.\n        '
        test_exp_filepath = os.path.join(feconf.TESTS_DATA_DIR, 'string_classifier_test.yaml')
        yaml_content = utils.get_file_contents(test_exp_filepath)
        assets_list: List[Tuple[str, bytes]] = []
        with self.swap(feconf, 'ENABLE_ML_CLASSIFIERS', True):
            exp_services.save_new_exploration_from_yaml_and_assets(feconf.SYSTEM_COMMITTER_ID, yaml_content, exploration_id, assets_list)
        self.exp_id = exploration_id
        self.exp_state = exp_fetchers.get_exploration_by_id(exploration_id).states['Home']

    def _create_classifier_training_job(self, algorithm_id: str, interaction_id: str, exp_id: str, exp_version: int, next_scheduled_check_time: datetime.datetime, training_data: classifier_models.TrainingDataUnionType, state_name: str, status: str, classifier_data: Dict[str, str], algorithm_version: int) -> str:
        if False:
            i = 10
            return i + 15
        'Creates a new classifier training job model and stores\n        classfier data in a file.\n        '
        job_id = classifier_models.ClassifierTrainingJobModel.create(algorithm_id, interaction_id, exp_id, exp_version, next_scheduled_check_time, training_data, state_name, status, algorithm_version)
        classifier_data_proto = text_classifier_pb2.TextClassifierFrozenModel()
        classifier_data_proto.model_json = json.dumps(classifier_data)
        fs_services.save_classifier_data(exp_id, job_id, classifier_data_proto)
        return job_id

    def test_creation_of_jobs_and_mappings(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test the handle_trainable_states method and\n        get_job_models_that_handle_non_trainable_states method by triggering\n        update_exploration() method.\n        '
        exploration = exp_fetchers.get_exploration_by_id(self.exp_id)
        state = exploration.states['Home']
        all_jobs = classifier_models.ClassifierTrainingJobModel.get_all()
        self.assertEqual(all_jobs.count(), 1)
        all_mappings = classifier_models.StateTrainingJobsMappingModel.get_all()
        self.assertEqual(all_mappings.count(), 1)
        new_answer_group = copy.deepcopy(state.interaction.answer_groups[1])
        new_answer_group.outcome.feedback.content_id = 'new_feedback'
        new_answer_group.rule_specs[0].inputs['x']['contentId'] = 'rule_input_4'
        new_answer_group.rule_specs[0].inputs['x']['normalizedStrSet'] = ['Divide']
        state.recorded_voiceovers.voiceovers_mapping['new_feedback'] = {}
        state.recorded_voiceovers.voiceovers_mapping['rule_input_4'] = {}
        state.interaction.answer_groups.insert(3, new_answer_group)
        answer_groups: List[state_domain.AnswerGroupDict] = []
        for answer_group in state.interaction.answer_groups:
            answer_groups.append(answer_group.to_dict())
        change_list = [exp_domain.ExplorationChange({'cmd': 'edit_state_property', 'state_name': 'Home', 'property_name': 'answer_groups', 'new_value': answer_groups}), exp_domain.ExplorationChange({'cmd': 'edit_state_property', 'state_name': 'Home', 'property_name': 'recorded_voiceovers', 'new_value': state.recorded_voiceovers.to_dict()})]
        with self.swap(feconf, 'ENABLE_ML_CLASSIFIERS', True):
            exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.exp_id, change_list, '')
        all_jobs = classifier_models.ClassifierTrainingJobModel.get_all()
        self.assertEqual(all_jobs.count(), 2)
        all_mappings = classifier_models.StateTrainingJobsMappingModel.get_all()
        self.assertEqual(all_mappings.count(), 2)
        change_list = [exp_domain.ExplorationChange({'cmd': 'edit_exploration_property', 'property_name': 'title', 'new_value': 'New title'})]
        with self.swap(feconf, 'ENABLE_ML_CLASSIFIERS', True):
            exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.exp_id, change_list, '')
        all_jobs = classifier_models.ClassifierTrainingJobModel.get_all()
        self.assertEqual(all_jobs.count(), 2)
        all_mappings = classifier_models.StateTrainingJobsMappingModel.get_all()
        self.assertEqual(all_mappings.count(), 3)
        change_list = [exp_domain.ExplorationChange({'cmd': 'rename_state', 'old_state_name': 'Home', 'new_state_name': 'Home2'}), exp_domain.ExplorationChange({'cmd': 'rename_state', 'old_state_name': 'Home2', 'new_state_name': 'Home3'})]
        with self.swap(feconf, 'ENABLE_ML_CLASSIFIERS', True):
            exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.exp_id, change_list, '')
        all_jobs = classifier_models.ClassifierTrainingJobModel.get_all()
        self.assertEqual(all_jobs.count(), 2)
        all_mappings = classifier_models.StateTrainingJobsMappingModel.get_all()
        self.assertEqual(all_mappings.count(), 4)

    def test_that_models_are_recreated_if_not_available(self) -> None:
        if False:
            while True:
                i = 10
        'Test ensures that classifier models for state are retrained if\n        they are not available.\n        '
        exploration = exp_fetchers.get_exploration_by_id(self.exp_id)
        state = exploration.states['Home']
        all_jobs = classifier_models.ClassifierTrainingJobModel.get_all()
        self.assertEqual(all_jobs.count(), 1)
        all_mappings = classifier_models.StateTrainingJobsMappingModel.get_all()
        self.assertEqual(all_mappings.count(), 1)
        new_answer_group = copy.deepcopy(state.interaction.answer_groups[1])
        new_answer_group.outcome.feedback.content_id = 'new_feedback'
        new_answer_group.rule_specs[0].inputs['x']['contentId'] = 'rule_input_4'
        new_answer_group.rule_specs[0].inputs['x']['normalizedStrSet'] = ['Multiplication']
        state.recorded_voiceovers.voiceovers_mapping['new_feedback'] = {}
        state.recorded_voiceovers.voiceovers_mapping['rule_input_4'] = {}
        state.interaction.answer_groups.insert(3, new_answer_group)
        answer_groups = []
        for answer_group in state.interaction.answer_groups:
            answer_groups.append(answer_group.to_dict())
        change_list = [exp_domain.ExplorationChange({'cmd': 'edit_state_property', 'state_name': 'Home', 'property_name': 'answer_groups', 'new_value': answer_groups}), exp_domain.ExplorationChange({'cmd': 'edit_state_property', 'state_name': 'Home', 'property_name': 'recorded_voiceovers', 'new_value': state.recorded_voiceovers.to_dict()})]
        with self.swap(feconf, 'ENABLE_ML_CLASSIFIERS', True):
            exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.exp_id, change_list, '')
        all_jobs = classifier_models.ClassifierTrainingJobModel.get_all()
        self.assertEqual(all_jobs.count(), 2)
        all_mappings = classifier_models.StateTrainingJobsMappingModel.get_all()
        self.assertEqual(all_mappings.count(), 2)
        change_list = [exp_domain.ExplorationChange({'cmd': 'edit_exploration_property', 'property_name': 'title', 'new_value': 'New title'})]
        with self.swap(feconf, 'ENABLE_ML_CLASSIFIERS', False):
            exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.exp_id, change_list, '')
        all_jobs = classifier_models.ClassifierTrainingJobModel.get_all()
        self.assertEqual(all_jobs.count(), 2)
        all_mappings = classifier_models.StateTrainingJobsMappingModel.get_all()
        self.assertEqual(all_mappings.count(), 2)
        change_list = [exp_domain.ExplorationChange({'cmd': 'edit_exploration_property', 'property_name': 'title', 'new_value': 'New title'})]
        with self.swap(feconf, 'ENABLE_ML_CLASSIFIERS', True):
            exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.exp_id, change_list, '')
        all_jobs = classifier_models.ClassifierTrainingJobModel.get_all()
        self.assertEqual(all_jobs.count(), 3)
        all_mappings = classifier_models.StateTrainingJobsMappingModel.get_all()
        self.assertEqual(all_mappings.count(), 3)

    def test_get_new_job_models_for_trainable_states(self) -> None:
        if False:
            while True:
                i = 10
        'Test the handle_trainable_states method.'
        exploration = exp_fetchers.get_exploration_by_id(self.exp_id)
        state_names = ['Home']
        job_models = classifier_services.get_new_job_models_for_trainable_states(exploration, state_names)
        datastore_services.put_multi(job_models)
        all_jobs = classifier_models.ClassifierTrainingJobModel.get_all()
        self.assertEqual(all_jobs.count(), 2)
        for (index, job) in enumerate(all_jobs):
            if index == 1:
                job_id = job.id
        classifier_training_job = classifier_services.get_classifier_training_job_by_id(job_id)
        self.assertEqual(classifier_training_job.exp_id, self.exp_id)
        self.assertEqual(classifier_training_job.state_name, 'Home')

    def test_handle_trainable_states_raises_error_for_invalid_interaction_id(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the handle_trainable_states method.'
        exploration = exp_fetchers.get_exploration_by_id(self.exp_id)
        state_names = ['Home']
        exploration.states['Home'].interaction.id = 'Invalid_id'
        with self.assertRaisesRegex(Exception, 'No classifier algorithm found for Invalid_id interaction'):
            classifier_services.get_new_job_models_for_trainable_states(exploration, state_names)

    def test_get_new_job_models_for_non_trainable_states(self) -> None:
        if False:
            return 10
        'Test the get_job_models_that_handle_non_trainable_states method.'
        exploration = exp_fetchers.get_exploration_by_id(self.exp_id)
        next_scheduled_check_time = datetime.datetime.utcnow()
        state_names = ['Home']
        change_list = [exp_domain.ExplorationChange({'cmd': 'rename_state', 'old_state_name': 'Old home', 'new_state_name': 'Home'})]
        exp_versions_diff = exp_domain.ExplorationVersionsDiff(change_list)
        exploration.version = 1
        with self.assertRaisesRegex(Exception, 'This method should not be called by exploration with version number 1'):
            (_, job_models) = classifier_services.get_new_job_models_for_non_trainable_states(exploration, state_names, exp_versions_diff)
            datastore_services.put_multi(job_models)
        exploration.version += 1
        (_, job_models) = classifier_services.get_new_job_models_for_non_trainable_states(exploration, state_names, exp_versions_diff)
        datastore_services.put_multi(job_models)
        all_mappings = classifier_models.StateTrainingJobsMappingModel.get_all()
        self.assertEqual(all_mappings.count(), 1)
        algorithm_id = feconf.INTERACTION_CLASSIFIER_MAPPING['TextInput']['algorithm_id']
        job_id = self._create_classifier_training_job(algorithm_id, 'TextInput', self.exp_id, exploration.version - 1, next_scheduled_check_time, [], 'Old home', feconf.TRAINING_JOB_STATUS_COMPLETE, {}, 1)
        classifier_models.StateTrainingJobsMappingModel.create(self.exp_id, exploration.version - 1, 'Old home', {algorithm_id: job_id})
        all_mappings = classifier_models.StateTrainingJobsMappingModel.get_all()
        self.assertEqual(all_mappings.count(), 2)
        (_, job_models) = classifier_services.get_new_job_models_for_non_trainable_states(exploration, state_names, exp_versions_diff)
        datastore_services.put_multi(job_models)
        all_mappings = classifier_models.StateTrainingJobsMappingModel.get_all()
        self.assertEqual(all_mappings.count(), 3)
        for (index, mapping) in enumerate(all_mappings):
            if index == 2:
                mapping_id = mapping.id
        state_training_jobs_mapping = classifier_models.StateTrainingJobsMappingModel.get(mapping_id)
        self.assertEqual(state_training_jobs_mapping.exp_id, self.exp_id)
        self.assertEqual(state_training_jobs_mapping.state_name, 'Home')

    def test_retrieval_of_classifier_training_jobs(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the get_classifier_training_job_by_id method.'
        with self.assertRaisesRegex(Exception, 'Entity for class ClassifierTrainingJobModel with id fake_id not found'):
            classifier_services.get_classifier_training_job_by_id('fake_id')
        exp_id = u'1'
        state_name = 'Home'
        interaction_id = 'TextInput'
        next_scheduled_check_time = datetime.datetime.utcnow()
        job_id = self._create_classifier_training_job(feconf.INTERACTION_CLASSIFIER_MAPPING['TextInput']['algorithm_id'], interaction_id, exp_id, 1, next_scheduled_check_time, [], state_name, feconf.TRAINING_JOB_STATUS_NEW, {}, 1)
        classifier_training_job = classifier_services.get_classifier_training_job_by_id(job_id)
        self.assertEqual(classifier_training_job.algorithm_id, feconf.INTERACTION_CLASSIFIER_MAPPING['TextInput']['algorithm_id'])
        self.assertEqual(classifier_training_job.interaction_id, interaction_id)
        self.assertEqual(classifier_training_job.exp_id, exp_id)
        self.assertEqual(classifier_training_job.exp_version, 1)
        self.assertEqual(classifier_training_job.next_scheduled_check_time, next_scheduled_check_time)
        self.assertEqual(classifier_training_job.training_data, [])
        classifier_data = self._get_classifier_data_from_classifier_training_job(classifier_training_job)
        self.assertEqual(json.loads(classifier_data.model_json), {})
        self.assertEqual(classifier_training_job.state_name, state_name)
        self.assertEqual(classifier_training_job.status, feconf.TRAINING_JOB_STATUS_NEW)
        self.assertEqual(classifier_training_job.algorithm_version, 1)

    def test_deletion_of_classifier_training_jobs(self) -> None:
        if False:
            while True:
                i = 10
        'Test the delete_classifier_training_job method.'
        exp_id = u'1'
        state_name = 'Home'
        interaction_id = 'TextInput'
        next_scheduled_check_time = datetime.datetime.utcnow()
        job_id = self._create_classifier_training_job(feconf.INTERACTION_CLASSIFIER_MAPPING['TextInput']['algorithm_id'], interaction_id, exp_id, 1, next_scheduled_check_time, [], state_name, feconf.TRAINING_JOB_STATUS_NEW, {}, 1)
        self.assertTrue(job_id)
        classifier_services.delete_classifier_training_job(job_id)
        with self.assertRaisesRegex(Exception, 'Entity for class ClassifierTrainingJobModel with id %s not found' % job_id):
            classifier_services.get_classifier_training_job_by_id(job_id)

    def test_mark_training_job_complete(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the mark_training_job_complete method.'
        exp_id = u'1'
        next_scheduled_check_time = datetime.datetime.utcnow()
        state_name = 'Home'
        interaction_id = 'TextInput'
        job_id = self._create_classifier_training_job(feconf.INTERACTION_CLASSIFIER_MAPPING['TextInput']['algorithm_id'], interaction_id, exp_id, 1, next_scheduled_check_time, [], state_name, feconf.TRAINING_JOB_STATUS_PENDING, {}, 1)
        classifier_training_job = classifier_services.get_classifier_training_job_by_id(job_id)
        self.assertEqual(classifier_training_job.status, feconf.TRAINING_JOB_STATUS_PENDING)
        classifier_services.mark_training_job_complete(job_id)
        classifier_training_job = classifier_services.get_classifier_training_job_by_id(job_id)
        self.assertEqual(classifier_training_job.status, feconf.TRAINING_JOB_STATUS_COMPLETE)
        with self.assertRaisesRegex(Exception, 'The status change %s to %s is not valid.' % (feconf.TRAINING_JOB_STATUS_COMPLETE, feconf.TRAINING_JOB_STATUS_COMPLETE)):
            classifier_services.mark_training_job_complete(job_id)

    def test_mark_training_job_pending(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the mark_training_job_pending method.'
        exp_id = u'1'
        state_name = 'Home'
        interaction_id = 'TextInput'
        job_id = self._create_classifier_training_job(feconf.INTERACTION_CLASSIFIER_MAPPING[interaction_id]['algorithm_id'], interaction_id, exp_id, 1, datetime.datetime.utcnow(), [], state_name, feconf.TRAINING_JOB_STATUS_NEW, {}, 1)
        classifier_training_job = classifier_services.get_classifier_training_job_by_id(job_id)
        self.assertEqual(classifier_training_job.status, feconf.TRAINING_JOB_STATUS_NEW)
        classifier_services.mark_training_job_pending(job_id)
        classifier_training_job = classifier_services.get_classifier_training_job_by_id(job_id)
        self.assertEqual(classifier_training_job.status, feconf.TRAINING_JOB_STATUS_PENDING)
        with self.assertRaisesRegex(Exception, 'The status change %s to %s is not valid.' % (feconf.TRAINING_JOB_STATUS_PENDING, feconf.TRAINING_JOB_STATUS_PENDING)):
            classifier_services.mark_training_job_pending(job_id)

    def test_mark_training_jobs_failed(self) -> None:
        if False:
            print('Hello World!')
        'Test the mark_training_job_failed method.'
        exp_id = u'1'
        state_name = 'Home'
        interaction_id = 'TextInput'
        algorithm_id = feconf.INTERACTION_CLASSIFIER_MAPPING[interaction_id]['algorithm_id']
        algorithm_version = feconf.INTERACTION_CLASSIFIER_MAPPING[interaction_id]['algorithm_version']
        job_id = self._create_classifier_training_job(algorithm_id, interaction_id, exp_id, 1, datetime.datetime.utcnow(), [], state_name, feconf.TRAINING_JOB_STATUS_PENDING, {}, algorithm_version)
        classifier_training_job = classifier_services.get_classifier_training_job_by_id(job_id)
        self.assertEqual(classifier_training_job.status, feconf.TRAINING_JOB_STATUS_PENDING)
        classifier_services.mark_training_jobs_failed([job_id])
        classifier_training_job = classifier_services.get_classifier_training_job_by_id(job_id)
        self.assertEqual(classifier_training_job.status, feconf.TRAINING_JOB_STATUS_FAILED)
        with self.assertRaisesRegex(Exception, 'The status change %s to %s is not valid.' % (feconf.TRAINING_JOB_STATUS_FAILED, feconf.TRAINING_JOB_STATUS_FAILED)):
            classifier_services.mark_training_jobs_failed([job_id])

    def test_fetch_next_job(self) -> None:
        if False:
            return 10
        'Test the fetch_next_jobs method.'
        exp1_id = '1'
        state_name = 'Home'
        interaction_id = 'TextInput'
        exp2_id = '0'
        job1_id = self._create_classifier_training_job(feconf.INTERACTION_CLASSIFIER_MAPPING[interaction_id]['algorithm_id'], interaction_id, exp1_id, 1, datetime.datetime.utcnow(), [], state_name, feconf.TRAINING_JOB_STATUS_NEW, {}, 1)
        self._create_classifier_training_job(feconf.INTERACTION_CLASSIFIER_MAPPING[interaction_id]['algorithm_id'], interaction_id, exp2_id, 1, datetime.datetime.utcnow(), [], state_name, feconf.TRAINING_JOB_STATUS_PENDING, {}, 1)
        classifier_services.fetch_next_job()
        next_job = classifier_services.fetch_next_job()
        assert next_job is not None
        self.assertEqual(job1_id, next_job.job_id)
        next_job = classifier_services.fetch_next_job()
        self.assertIsNone(next_job)

    def test_store_classifier_data(self) -> None:
        if False:
            print('Hello World!')
        'Test the store_classifier_data method.'
        exp_id = u'1'
        next_scheduled_check_time = datetime.datetime.utcnow()
        state_name = 'Home'
        interaction_id = 'TextInput'
        job_id = self._create_classifier_training_job(feconf.INTERACTION_CLASSIFIER_MAPPING['TextInput']['algorithm_id'], interaction_id, exp_id, 1, next_scheduled_check_time, [], state_name, feconf.TRAINING_JOB_STATUS_PENDING, {}, 1)
        classifier_training_job = classifier_services.get_classifier_training_job_by_id(job_id)
        classifier_data = self._get_classifier_data_from_classifier_training_job(classifier_training_job)
        self.assertEqual(json.loads(classifier_data.model_json), {})
        classifier_data_proto = text_classifier_pb2.TextClassifierFrozenModel()
        classifier_data_proto.model_json = json.dumps({'classifier_data': 'data'})
        classifier_services.store_classifier_data(job_id, classifier_data_proto)
        classifier_training_job = classifier_services.get_classifier_training_job_by_id(job_id)
        classifier_data = self._get_classifier_data_from_classifier_training_job(classifier_training_job)
        self.assertDictEqual(json.loads(classifier_data.model_json), {'classifier_data': 'data'})

    def test_retrieval_of_classifier_training_jobs_from_exploration_attributes(self) -> None:
        if False:
            print('Hello World!')
        'Test the get_classifier_training_job method.'
        exp_id = u'1'
        next_scheduled_check_time = datetime.datetime.utcnow()
        state_name = u'टेक्स्ट'
        algorithm_id = feconf.INTERACTION_CLASSIFIER_MAPPING['TextInput']['algorithm_id']
        algorithm_version = feconf.INTERACTION_CLASSIFIER_MAPPING['TextInput']['algorithm_version']
        job_id = self._create_classifier_training_job(algorithm_id, 'TextInput', exp_id, 1, next_scheduled_check_time, [], state_name, feconf.TRAINING_JOB_STATUS_NEW, {}, algorithm_version)
        classifier_models.StateTrainingJobsMappingModel.create(exp_id, 1, state_name, {algorithm_id: job_id})
        classifier_training_job = classifier_services.get_classifier_training_job(exp_id, 1, state_name, algorithm_id)
        assert classifier_training_job is not None
        self.assertEqual(classifier_training_job.exp_id, exp_id)
        self.assertEqual(classifier_training_job.exp_version, 1)
        self.assertEqual(classifier_training_job.state_name, state_name)
        self.assertEqual(classifier_training_job.job_id, job_id)
        false_state_name = 'false_name'
        classifier_training_job = classifier_services.get_classifier_training_job(exp_id, 1, false_state_name, algorithm_id)
        self.assertIsNone(classifier_training_job)

    def test_can_not_mark_training_jobs_complete_due_to_invalid_job_id(self) -> None:
        if False:
            return 10
        with self.assertRaisesRegex(Exception, 'The ClassifierTrainingJobModel corresponding to the job_id of the ClassifierTrainingJob does not exist.'):
            classifier_services.mark_training_job_complete('invalid_job_id')

    def test_can_not_mark_training_jobs_failed_due_to_invalid_job_id(self) -> None:
        if False:
            return 10
        with self.assertRaisesRegex(Exception, 'The ClassifierTrainingJobModel corresponding to the job_id of the ClassifierTrainingJob does not exist.'):
            classifier_services.mark_training_jobs_failed(['invalid_job_id'])

    def test_can_not_mark_training_jobs_pending_due_to_invalid_job_id(self) -> None:
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(Exception, 'The ClassifierTrainingJobModel corresponding to the job_id of the ClassifierTrainingJob does not exist.'):
            classifier_services.mark_training_job_pending('invalid_job_id')

    def test_can_not_store_classifier_data_due_to_invalid_job_id(self) -> None:
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(Exception, 'The ClassifierTrainingJobModel corresponding to the job_id of the ClassifierTrainingJob does not exist.'):
            classifier_services.store_classifier_data('invalid_job_id', {})

    def test_generate_signature(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the generate_signature method.'
        vm_id = feconf.DEFAULT_VM_ID
        secret = feconf.DEFAULT_VM_SHARED_SECRET
        message = b'test message'
        signature = classifier_services.generate_signature(secret.encode('utf-8'), message, vm_id)
        expected_signature = '9c2f9f607c0eefc2b8ba153bad9331843a6efc71c82e690f5f0341bbc38b7fa7'
        self.assertEqual(signature, expected_signature)

    def test_verify_signature(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test the verify_signature method.'

        def _mock_get_secret(name: str) -> Optional[str]:
            if False:
                i = 10
                return i + 15
            if name == 'VM_ID':
                return 'vm_default'
            elif name == 'SHARED_SECRET_KEY':
                return '1a2b3c4e'
            return None
        vm_id = feconf.DEFAULT_VM_ID
        message = 'test message'
        expected_signature = '9c2f9f607c0eefc2b8ba153bad9331843a6efc71c82e690f5f0341bbc38b7fa7'
        invalid_signature = 'invalid signature'
        invalid_vm_id = 'invalid vm_id'
        oppia_ml_auth_info = classifier_domain.OppiaMLAuthInfo(message.encode('utf-8'), vm_id, expected_signature)
        with self.swap_with_checks(secrets_services, 'get_secret', _mock_get_secret, expected_args=[('VM_ID',), ('SHARED_SECRET_KEY',)]):
            self.assertTrue(classifier_services.verify_signature(oppia_ml_auth_info))
        oppia_ml_auth_info = classifier_domain.OppiaMLAuthInfo(message.encode('utf-8'), vm_id, invalid_signature)
        with self.swap_with_checks(secrets_services, 'get_secret', _mock_get_secret, expected_args=[('VM_ID',), ('SHARED_SECRET_KEY',)]):
            self.assertFalse(classifier_services.verify_signature(oppia_ml_auth_info))
        oppia_ml_auth_info = classifier_domain.OppiaMLAuthInfo(message.encode('utf-8'), invalid_vm_id, expected_signature)
        with self.swap_with_checks(secrets_services, 'get_secret', _mock_get_secret, expected_args=[('VM_ID',), ('SHARED_SECRET_KEY',)]):
            self.assertFalse(classifier_services.verify_signature(oppia_ml_auth_info))

    def test_get_state_training_jobs_mapping(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the get_state_training_jobs_mapping method.'
        exp_id = u'1'
        next_scheduled_check_time = datetime.datetime.utcnow()
        state_name = u'Home'
        algorithm_id = feconf.INTERACTION_CLASSIFIER_MAPPING['TextInput']['algorithm_id']
        algorithm_version = feconf.INTERACTION_CLASSIFIER_MAPPING['TextInput']['algorithm_version']
        job_id = self._create_classifier_training_job(algorithm_id, 'TextInput', exp_id, 1, next_scheduled_check_time, [], state_name, feconf.TRAINING_JOB_STATUS_NEW, {}, algorithm_version)
        classifier_models.StateTrainingJobsMappingModel.create(exp_id, 1, state_name, {algorithm_id: job_id})
        state_training_jobs_mapping = classifier_services.get_state_training_jobs_mapping(exp_id, 1, state_name)
        assert state_training_jobs_mapping is not None
        self.assertEqual(state_training_jobs_mapping.exp_id, exp_id)
        self.assertEqual(state_training_jobs_mapping.state_name, 'Home')
        invalid_state_name = 'invalid name'
        state_training_jobs_mapping = classifier_services.get_state_training_jobs_mapping(exp_id, 1, invalid_state_name)
        self.assertIsNone(state_training_jobs_mapping)

    def test_migrate_state_training_jobs(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the migrate_state_training_jobs method.'
        state_name = 'Home'
        mock_interaction_classifier_mapping = {'TextInput': {'algorithm_id': 'NewTextClassifier', 'algorithm_version': 1}}
        expected_state_training_jobs = classifier_services.get_state_training_jobs_mapping(self.exp_id, 1, state_name)
        assert expected_state_training_jobs is not None
        with self.swap(feconf, 'INTERACTION_CLASSIFIER_MAPPING', mock_interaction_classifier_mapping):
            classifier_services.migrate_state_training_jobs(expected_state_training_jobs)
        state_training_jobs_mapping = classifier_services.get_state_training_jobs_mapping(self.exp_id, 1, 'Home')
        assert state_training_jobs_mapping is not None
        self.assertIn('NewTextClassifier', state_training_jobs_mapping.algorithm_ids_to_job_ids)
        mock_interaction_classifier_mapping = {'TextInput': {'algorithm_id': 'NewTextClassifier', 'algorithm_version': 2}}
        with self.swap(feconf, 'INTERACTION_CLASSIFIER_MAPPING', mock_interaction_classifier_mapping):
            classifier_services.migrate_state_training_jobs(expected_state_training_jobs)
        next_job = classifier_services.fetch_next_job()
        assert next_job is not None
        self.assertEqual(mock_interaction_classifier_mapping['TextInput']['algorithm_version'], next_job.algorithm_version)

    def test_reverted_exploration_maintains_classifier_model_mapping(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test if the classifier model mapping is maintained when an\n        exploration is reverted.\n        '
        state_name = 'Home'
        exploration = exp_fetchers.get_exploration_by_id(self.exp_id)
        interaction_id = exploration.states[state_name].interaction.id
        assert interaction_id is not None
        algorithm_id = feconf.INTERACTION_CLASSIFIER_MAPPING[interaction_id]['algorithm_id']
        change_list = [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_EXPLORATION_PROPERTY, 'property_name': 'title', 'new_value': 'A new title'})]
        with self.swap(feconf, 'ENABLE_ML_CLASSIFIERS', True):
            exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.exp_id, change_list, '')
        current_exploration = exp_fetchers.get_exploration_by_id(self.exp_id)
        old_job = classifier_services.get_classifier_training_job(self.exp_id, current_exploration.version, state_name, algorithm_id)
        assert old_job is not None
        old_job_id = old_job.job_id
        with self.swap(feconf, 'ENABLE_ML_CLASSIFIERS', True):
            exp_services.revert_exploration(feconf.SYSTEM_COMMITTER_ID, self.exp_id, current_exploration.version, current_exploration.version - 1)
        reverted_exploration = exp_fetchers.get_exploration_by_id(self.exp_id)
        self.assertEqual(reverted_exploration.version, current_exploration.version + 1)
        new_job = classifier_services.get_classifier_training_job(self.exp_id, reverted_exploration.version, state_name, algorithm_id)
        assert new_job is not None
        new_job_id = new_job.job_id
        self.assertEqual(old_job_id, new_job_id)

    def test_migrate_state_training_jobs_with_invalid_interaction_id(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the migrate_state_training_jobs method.'
        exploration = self.save_new_valid_exploration('44', feconf.SYSTEM_COMMITTER_ID, objective='The objective', category='Algebra')
        content_id_generator = translation_domain.ContentIdGenerator(exploration.next_content_id_index)
        self.assertEqual(exploration.version, 1)
        change_list = [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_ADD_STATE, 'state_name': 'New state', 'content_id_for_state_content': content_id_generator.generate(translation_domain.ContentType.CONTENT), 'content_id_for_default_outcome': content_id_generator.generate(translation_domain.ContentType.DEFAULT_OUTCOME)}), exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_EXPLORATION_PROPERTY, 'property_name': 'next_content_id_index', 'new_value': content_id_generator.next_content_id_index, 'old_value': 0}), exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_STATE_PROPERTY, 'state_name': 'New state', 'property_name': exp_domain.STATE_PROPERTY_INTERACTION_ID, 'new_value': None})]
        exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, exploration.id, change_list, '')
        state_training_jobs_mapping = classifier_domain.StateTrainingJobsMapping('44', 2, 'New state', {})
        with self.assertRaisesRegex(Exception, 'Interaction id does not exist.'):
            classifier_services.migrate_state_training_jobs(state_training_jobs_mapping)