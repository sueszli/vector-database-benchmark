"""Unit tests for core.domain.exp_fetchers."""
from __future__ import annotations
from core import feconf
from core.domain import caching_services
from core.domain import exp_domain
from core.domain import exp_fetchers
from core.domain import exp_services
from core.domain import rights_manager
from core.domain import state_domain
from core.domain import stats_services
from core.domain import translation_domain
from core.domain import user_services
from core.platform import models
from core.tests import test_utils
from typing import Final
MYPY = False
if MYPY:
    from mypy_imports import exp_models
(exp_models,) = models.Registry.import_models([models.Names.EXPLORATION])

class ExplorationRetrievalTests(test_utils.GenericTestBase):
    """Test the exploration retrieval methods."""
    EXP_1_ID: Final = 'exploration_1_id'
    EXP_2_ID: Final = 'exploration_2_id'
    EXP_3_ID: Final = 'exploration_3_id'

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.signup(self.OWNER_EMAIL, self.OWNER_USERNAME)
        self.owner_id = self.get_user_id_from_email(self.OWNER_EMAIL)
        self.exploration_1 = self.save_new_default_exploration(self.EXP_1_ID, self.owner_id, title='Aa')
        self.content_id_generator_1 = translation_domain.ContentIdGenerator(self.exploration_1.next_content_id_index)
        self.exploration_2 = self.save_new_default_exploration(self.EXP_2_ID, self.owner_id, title='Bb')
        self.exploration_3 = self.save_new_default_exploration(self.EXP_3_ID, self.owner_id, title='Cc')

    def test_get_exploration_summaries_matching_ids(self) -> None:
        if False:
            i = 10
            return i + 15
        summaries = exp_fetchers.get_exploration_summaries_matching_ids([self.EXP_1_ID, self.EXP_2_ID, self.EXP_3_ID, 'nonexistent'])
        assert summaries[0] is not None
        self.assertEqual(summaries[0].title, self.exploration_1.title)
        assert summaries[1] is not None
        self.assertEqual(summaries[1].title, self.exploration_2.title)
        assert summaries[2] is not None
        self.assertEqual(summaries[2].title, self.exploration_3.title)
        self.assertIsNone(summaries[3])

    def test_get_exploration_summaries_subscribed_to(self) -> None:
        if False:
            return 10
        summaries = exp_fetchers.get_exploration_summaries_subscribed_to(self.owner_id)
        self.assertEqual(summaries[0].title, self.exploration_1.title)
        self.assertEqual(summaries[1].title, self.exploration_2.title)
        self.assertEqual(summaries[2].title, self.exploration_3.title)

    def test_get_new_exploration_id(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertIsNotNone(exp_fetchers.get_new_exploration_id())

    def test_get_new_unique_progress_url_id(self) -> None:
        if False:
            while True:
                i = 10
        self.assertIsNotNone(exp_fetchers.get_new_unique_progress_url_id())

    def test_get_exploration_summary_by_id(self) -> None:
        if False:
            print('Hello World!')
        fake_eid = 'fake_eid'
        fake_exp = exp_fetchers.get_exploration_summary_by_id(fake_eid, strict=False)
        self.assertIsNone(fake_exp)
        exp_summary = exp_fetchers.get_exploration_summary_by_id(self.EXP_1_ID)
        self.assertIsNotNone(exp_summary)
        self.assertEqual(exp_summary.id, self.EXP_1_ID)

    def test_get_exploration_summaries_from_models(self) -> None:
        if False:
            i = 10
            return i + 15
        exp_ids = [self.EXP_1_ID, self.EXP_2_ID, self.EXP_3_ID]
        exp_summary_models = []
        exp_summary_models_with_none = exp_models.ExpSummaryModel.get_multi(exp_ids)
        for model in exp_summary_models_with_none:
            assert model is not None
            exp_summary_models.append(model)
        exp_summary_dict = exp_fetchers.get_exploration_summaries_from_models(exp_summary_models)
        for key in exp_summary_dict:
            self.assertIn(key, exp_ids)

    def test_retrieval_of_fake_exploration(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertIsNone(exp_fetchers.get_exploration_by_id('fake_eid', strict=False))

    def test_get_exploration_summaries_where_user_has_role(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        exp_ids = [self.EXP_1_ID, self.EXP_2_ID, self.EXP_3_ID]
        exp_summaries = exp_fetchers.get_exploration_summaries_where_user_has_role(self.owner_id)
        self.assertEqual(len(exp_summaries), 3)
        for exp_summary in exp_summaries:
            self.assertIn(exp_summary.id, exp_ids)

    def test_retrieval_of_explorations(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the get_exploration_by_id() method.'
        with self.assertRaisesRegex(Exception, 'Entity .* not found'):
            exp_fetchers.get_exploration_by_id('fake_eid')
        retrieved_exploration = exp_fetchers.get_exploration_by_id(self.EXP_1_ID)
        self.assertEqual(self.exploration_1.id, retrieved_exploration.id)
        self.assertEqual(self.exploration_1.title, retrieved_exploration.title)
        with self.assertRaisesRegex(Exception, 'Entity for class ExplorationModel with id fake_exploration not found'):
            exp_fetchers.get_exploration_by_id('fake_exploration')

    def test_retrieval_of_multiple_exploration_versions_for_fake_exp_id(self) -> None:
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'The given entity_id fake_exp_id is invalid'):
            exp_fetchers.get_multiple_versioned_exp_interaction_ids_mapping_by_version('fake_exp_id', [1, 2, 3])

    def test_retrieval_of_exp_versions_for_invalid_state_schema_version(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        error_regex = 'Exploration\\(id=%s, version=%s, states_schema_version=%s\\) does not match the latest schema version %s' % (self.EXP_1_ID, '1', feconf.CURRENT_STATE_SCHEMA_VERSION, '61')
        with self.swap(feconf, 'CURRENT_STATE_SCHEMA_VERSION', 61):
            with self.assertRaisesRegex(Exception, error_regex):
                exp_fetchers.get_multiple_versioned_exp_interaction_ids_mapping_by_version(self.EXP_1_ID, [1])

    def test_retrieval_of_multiple_exploration_versions(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        change_list = [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_ADD_STATE, 'state_name': 'New state', 'content_id_for_state_content': self.content_id_generator_1.generate(translation_domain.ContentType.CONTENT), 'content_id_for_default_outcome': self.content_id_generator_1.generate(translation_domain.ContentType.DEFAULT_OUTCOME)}), exp_domain.ExplorationChange({'cmd': 'edit_exploration_property', 'property_name': 'next_content_id_index', 'new_value': self.content_id_generator_1.next_content_id_index})]
        exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.EXP_1_ID, change_list, '')
        change_list = [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_ADD_STATE, 'state_name': 'New state 2', 'content_id_for_state_content': self.content_id_generator_1.generate(translation_domain.ContentType.CONTENT), 'content_id_for_default_outcome': self.content_id_generator_1.generate(translation_domain.ContentType.DEFAULT_OUTCOME)}), exp_domain.ExplorationChange({'cmd': 'edit_exploration_property', 'property_name': 'next_content_id_index', 'new_value': self.content_id_generator_1.next_content_id_index})]
        exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.EXP_1_ID, change_list, '')
        exploration_latest = exp_fetchers.get_exploration_by_id(self.EXP_1_ID)
        latest_version = exploration_latest.version
        explorations = exp_fetchers.get_multiple_versioned_exp_interaction_ids_mapping_by_version(self.EXP_1_ID, list(range(1, latest_version + 1)))
        self.assertEqual(len(explorations), 3)
        self.assertEqual(explorations[0].version, 1)
        self.assertEqual(explorations[1].version, 2)
        self.assertEqual(explorations[2].version, 3)

    def test_version_number_errors_for_get_multiple_exploration_versions(self) -> None:
        if False:
            print('Hello World!')
        change_list = [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_ADD_STATE, 'state_name': 'New state', 'content_id_for_state_content': self.content_id_generator_1.generate(translation_domain.ContentType.CONTENT), 'content_id_for_default_outcome': self.content_id_generator_1.generate(translation_domain.ContentType.DEFAULT_OUTCOME)}), exp_domain.ExplorationChange({'cmd': 'edit_exploration_property', 'property_name': 'next_content_id_index', 'new_value': self.content_id_generator_1.next_content_id_index})]
        exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.EXP_1_ID, change_list, '')
        change_list = [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_ADD_STATE, 'state_name': 'New state 2', 'content_id_for_state_content': self.content_id_generator_1.generate(translation_domain.ContentType.CONTENT), 'content_id_for_default_outcome': self.content_id_generator_1.generate(translation_domain.ContentType.DEFAULT_OUTCOME)}), exp_domain.ExplorationChange({'cmd': 'edit_exploration_property', 'property_name': 'next_content_id_index', 'new_value': self.content_id_generator_1.next_content_id_index})]
        exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.EXP_1_ID, change_list, '')
        with self.assertRaisesRegex(ValueError, 'Requested version number 4 cannot be higher than the current version number 3.'):
            exp_fetchers.get_multiple_versioned_exp_interaction_ids_mapping_by_version(self.EXP_1_ID, [1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, 'At least one version number is invalid'):
            exp_fetchers.get_multiple_versioned_exp_interaction_ids_mapping_by_version(self.EXP_1_ID, [1, 2, 2.5, 3])

    def test_retrieval_of_multiple_uncached_explorations(self) -> None:
        if False:
            print('Hello World!')
        exp_ids = [self.EXP_1_ID, self.EXP_2_ID, self.EXP_3_ID]
        caching_services.delete_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, None, exp_ids)
        uncached_explorations = exp_fetchers.get_multiple_explorations_by_id(exp_ids, False)
        self.assertEqual(len(uncached_explorations), 3)
        for key in uncached_explorations:
            self.assertIn(key, uncached_explorations)

    def test_retrieval_of_multiple_explorations(self) -> None:
        if False:
            while True:
                i = 10
        exps = {}
        chars = 'abcde'
        exp_ids = ['%s%s' % (self.EXP_1_ID, c) for c in chars]
        for _id in exp_ids:
            exp = self.save_new_valid_exploration(_id, self.owner_id)
            exps[_id] = exp
        result = exp_fetchers.get_multiple_explorations_by_id(exp_ids)
        for _id in exp_ids:
            self.assertEqual(result[_id].title, exps[_id].title)
        result = exp_fetchers.get_multiple_explorations_by_id(exp_ids + ['doesnt_exist'], strict=False)
        for _id in exp_ids:
            self.assertEqual(result[_id].title, exps[_id].title)
        self.assertNotIn('doesnt_exist', result)
        with self.assertRaisesRegex(Exception, "Couldn't find explorations with the following ids:\ndoesnt_exist"):
            exp_fetchers.get_multiple_explorations_by_id(exp_ids + ['doesnt_exist'])

    def test_exploration_user_data_is_none_before_starting_exploration(self) -> None:
        if False:
            while True:
                i = 10
        auth_id = 'test_id'
        user_email = 'test@email.com'
        user_id = user_services.create_new_user(auth_id, user_email).user_id
        self.assertIsNone(exp_fetchers.get_exploration_user_data(user_id, self.EXP_1_ID))

    def test_get_exploration_user_data(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        auth_id = 'test_id'
        username = 'testname'
        user_email = 'test@email.com'
        user_id = user_services.create_new_user(auth_id, user_email).user_id
        user_services.set_username(user_id, username)
        user_services.update_learner_checkpoint_progress(user_id, self.EXP_1_ID, 'Introduction', 1)
        expected_user_data_dict = {'rating': None, 'rated_on': None, 'draft_change_list': None, 'draft_change_list_last_updated': None, 'draft_change_list_exp_version': None, 'draft_change_list_id': 0, 'mute_suggestion_notifications': feconf.DEFAULT_SUGGESTION_NOTIFICATIONS_MUTED_PREFERENCE, 'mute_feedback_notifications': feconf.DEFAULT_FEEDBACK_NOTIFICATIONS_MUTED_PREFERENCE, 'furthest_reached_checkpoint_exp_version': 1, 'furthest_reached_checkpoint_state_name': 'Introduction', 'most_recently_reached_checkpoint_exp_version': 1, 'most_recently_reached_checkpoint_state_name': 'Introduction'}
        exp_user_data = exp_fetchers.get_exploration_user_data(user_id, self.EXP_1_ID)
        assert exp_user_data is not None
        self.assertEqual(expected_user_data_dict, exp_user_data.to_dict())

    def test_get_exploration_version_history(self) -> None:
        if False:
            i = 10
            return i + 15
        version_history = exp_fetchers.get_exploration_version_history(self.EXP_1_ID, 2)
        self.assertIsNone(version_history)
        exp_services.update_exploration(self.owner_id, self.EXP_1_ID, [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_ADD_STATE, 'state_name': 'New state', 'content_id_for_state_content': self.content_id_generator_1.generate(translation_domain.ContentType.CONTENT), 'content_id_for_default_outcome': self.content_id_generator_1.generate(translation_domain.ContentType.DEFAULT_OUTCOME)}), exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_EXPLORATION_PROPERTY, 'property_name': 'next_content_id_index', 'new_value': self.content_id_generator_1.next_content_id_index, 'old_value': 0})], 'A commit message.')
        version_history = exp_fetchers.get_exploration_version_history(self.EXP_1_ID, 2)
        self.assertIsNotNone(version_history)
        if version_history is not None:
            self.assertEqual(version_history.committer_ids, [self.owner_id])
            self.assertEqual(version_history.state_version_history['New state'].to_dict(), state_domain.StateVersionHistory(None, None, self.owner_id).to_dict())

class LoggedOutUserProgressTests(test_utils.GenericTestBase):
    """Tests the fetching of the logged-out user progress."""
    UNIQUE_PROGRESS_URL_ID = 'pid123'
    EXP_1_ID = 'exploration_1_id'

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.signup(self.OWNER_EMAIL, self.OWNER_USERNAME)
        self.owner_id = self.get_user_id_from_email(self.OWNER_EMAIL)
        self.exploration_1 = self.save_new_default_exploration(self.EXP_1_ID, self.owner_id, title='Aa')

    def test_get_logged_out_user_progress(self) -> None:
        if False:
            while True:
                i = 10
        logged_out_user_data = exp_fetchers.get_logged_out_user_progress(self.UNIQUE_PROGRESS_URL_ID)
        self.assertIsNone(logged_out_user_data)
        exp_services.update_logged_out_user_progress(self.EXP_1_ID, self.UNIQUE_PROGRESS_URL_ID, 'Introduction', 1)
        expected_progress_dict = {'exploration_id': self.EXP_1_ID, 'furthest_reached_checkpoint_state_name': 'Introduction', 'furthest_reached_checkpoint_exp_version': 1, 'most_recently_reached_checkpoint_state_name': 'Introduction', 'most_recently_reached_checkpoint_exp_version': 1, 'last_updated': None}
        logged_out_user_data = exp_fetchers.get_logged_out_user_progress(self.UNIQUE_PROGRESS_URL_ID)
        assert logged_out_user_data is not None
        self.assertEqual(expected_progress_dict['exploration_id'], logged_out_user_data.exploration_id)
        self.assertEqual(expected_progress_dict['furthest_reached_checkpoint_state_name'], logged_out_user_data.furthest_reached_checkpoint_state_name)
        self.assertEqual(expected_progress_dict['furthest_reached_checkpoint_exp_version'], logged_out_user_data.furthest_reached_checkpoint_exp_version)
        self.assertEqual(expected_progress_dict['most_recently_reached_checkpoint_state_name'], logged_out_user_data.most_recently_reached_checkpoint_state_name)
        self.assertEqual(expected_progress_dict['most_recently_reached_checkpoint_exp_version'], logged_out_user_data.most_recently_reached_checkpoint_exp_version)

class ExplorationConversionPipelineTests(test_utils.GenericTestBase):
    """Tests the exploration model -> exploration conversion pipeline."""
    OLD_EXP_ID: Final = 'exp_id0'
    NEW_EXP_ID: Final = 'exp_id1'
    UPGRADED_EXP_YAML: Final = "author_notes: ''\nauto_tts_enabled: false\nblurb: ''\ncategory: Art\ncorrectness_feedback_enabled: true\nedits_allowed: true\ninit_state_name: Introduction\nlanguage_code: en\nnext_content_id_index: 6\nobjective: Exp objective...\nparam_changes: []\nparam_specs: {}\nschema_version: %d\nstates:\n  End:\n    card_is_checkpoint: false\n    classifier_model_id: null\n    content:\n      content_id: content_0\n      html: <p>Congratulations, you have finished!</p>\n    interaction:\n      answer_groups: []\n      confirmed_unclassified_answers: []\n      customization_args:\n        recommendedExplorationIds:\n          value: []\n      default_outcome: null\n      hints: []\n      id: EndExploration\n      solution: null\n    linked_skill_id: null\n    param_changes: []\n    recorded_voiceovers:\n      voiceovers_mapping:\n        content_0: {}\n    solicit_answer_details: false\n  %s:\n    card_is_checkpoint: true\n    classifier_model_id: null\n    content:\n      content_id: content_1\n      html: ''\n    interaction:\n      answer_groups:\n      - outcome:\n          dest: End\n          dest_if_really_stuck: null\n          feedback:\n            content_id: feedback_3\n            html: <p>Correct!</p>\n          labelled_as_correct: false\n          missing_prerequisite_skill_id: null\n          param_changes: []\n          refresher_exploration_id: null\n        rule_specs:\n        - inputs:\n            x:\n              contentId: rule_input_4\n              normalizedStrSet:\n              - InputString\n          rule_type: Equals\n        tagged_skill_misconception_id: null\n        training_data: []\n      confirmed_unclassified_answers: []\n      customization_args:\n        catchMisspellings:\n          value: false\n        placeholder:\n          value:\n            content_id: ca_placeholder_5\n            unicode_str: ''\n        rows:\n          value: 1\n      default_outcome:\n        dest: Introduction\n        dest_if_really_stuck: null\n        feedback:\n          content_id: default_outcome_2\n          html: ''\n        labelled_as_correct: false\n        missing_prerequisite_skill_id: null\n        param_changes: []\n        refresher_exploration_id: null\n      hints: []\n      id: TextInput\n      solution: null\n    linked_skill_id: null\n    param_changes: []\n    recorded_voiceovers:\n      voiceovers_mapping:\n        ca_placeholder_5: {}\n        content_1: {}\n        default_outcome_2: {}\n        feedback_3: {}\n        rule_input_4: {}\n    solicit_answer_details: false\nstates_schema_version: %d\ntags: []\ntitle: Old Title\n" % (exp_domain.Exploration.CURRENT_EXP_SCHEMA_VERSION, feconf.DEFAULT_INIT_STATE_NAME, feconf.CURRENT_STATE_SCHEMA_VERSION)
    STATES_AT_V41 = {'Introduction': {'classifier_model_id': None, 'content': {'content_id': 'content', 'html': ''}, 'interaction': {'answer_groups': [{'outcome': {'dest': 'End', 'feedback': {'content_id': 'feedback_1', 'html': '<p>Correct!</p>'}, 'labelled_as_correct': False, 'missing_prerequisite_skill_id': None, 'param_changes': [], 'refresher_exploration_id': None}, 'rule_specs': [{'inputs': {'x': {'contentId': 'rule_input_3', 'normalizedStrSet': ['InputString']}}, 'rule_type': 'Equals'}], 'tagged_skill_misconception_id': None, 'training_data': []}], 'confirmed_unclassified_answers': [], 'customization_args': {'placeholder': {'value': {'content_id': 'ca_placeholder_2', 'unicode_str': ''}}, 'rows': {'value': 1}}, 'default_outcome': {'dest': 'Introduction', 'feedback': {'content_id': 'default_outcome', 'html': ''}, 'labelled_as_correct': False, 'missing_prerequisite_skill_id': None, 'param_changes': [], 'refresher_exploration_id': None}, 'hints': [], 'id': 'TextInput', 'solution': None}, 'next_content_id_index': 4, 'param_changes': [], 'recorded_voiceovers': {'voiceovers_mapping': {'ca_placeholder_2': {}, 'content': {}, 'default_outcome': {}, 'feedback_1': {}, 'rule_input_3': {}}}, 'solicit_answer_details': False, 'written_translations': {'translations_mapping': {'ca_placeholder_2': {}, 'content': {}, 'default_outcome': {}, 'feedback_1': {}, 'rule_input_3': {}}}}, 'End': {'classifier_model_id': None, 'content': {'content_id': 'content', 'html': '<p>Congratulations, you have finished!</p>'}, 'interaction': {'answer_groups': [], 'confirmed_unclassified_answers': [], 'customization_args': {'recommendedExplorationIds': {'value': []}}, 'default_outcome': None, 'hints': [], 'id': 'EndExploration', 'solution': None}, 'next_content_id_index': 0, 'param_changes': [], 'recorded_voiceovers': {'voiceovers_mapping': {'content': {}}}, 'solicit_answer_details': False, 'written_translations': {'translations_mapping': {'content': {}}}}}
    ALBERT_EMAIL: Final = 'albert@example.com'
    ALBERT_NAME: Final = 'albert'

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.signup(self.ALBERT_EMAIL, self.ALBERT_NAME)
        self.albert_id = self.get_user_id_from_email(self.ALBERT_EMAIL)
        swap_states_schema_41 = self.swap(feconf, 'CURRENT_STATE_SCHEMA_VERSION', 41)
        swap_exp_schema_46 = self.swap(exp_domain.Exploration, 'CURRENT_EXP_SCHEMA_VERSION', 46)
        with swap_states_schema_41, swap_exp_schema_46:
            exploration = exp_domain.Exploration.create_default_exploration(self.OLD_EXP_ID, title='Old Title', category='Art', objective='Exp objective...')
            exploration_model = exp_models.ExplorationModel(id=self.OLD_EXP_ID)
            exp_services.populate_exp_model_fields(exploration_model, exploration)
        exploration_model.states = self.STATES_AT_V41
        rights_manager.create_new_exploration_rights(exploration_model.id, self.albert_id)
        exploration_model.commit(self.albert_id, 'Created new exploration.', [])
        exp_services.regenerate_exploration_summary_with_new_contributor(self.OLD_EXP_ID, self.albert_id)
        stats_services.create_exp_issues_for_new_exploration(exploration_model.id, exploration_model.version)
        new_exp = self.save_new_valid_exploration(self.NEW_EXP_ID, self.albert_id)
        self._up_to_date_yaml = new_exp.to_yaml()
        caching_services.delete_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, None, [self.OLD_EXP_ID, self.NEW_EXP_ID])

    def test_converts_exp_model_with_default_states_schema_version(self) -> None:
        if False:
            print('Hello World!')
        exploration = exp_fetchers.get_exploration_by_id(self.OLD_EXP_ID)
        self.assertEqual(exploration.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)
        self.assertEqual(exploration.to_yaml(), '%sversion: 1\n' % self.UPGRADED_EXP_YAML)

    def test_does_not_convert_up_to_date_exploration(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        exploration = exp_fetchers.get_exploration_by_id(self.NEW_EXP_ID)
        self.assertEqual(exploration.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)
        self.assertEqual(exploration.to_yaml(), self._up_to_date_yaml)

    def test_migration_with_invalid_state_schema(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.save_new_valid_exploration('fake_eid', self.albert_id)
        swap_earlier_state_to_60 = self.swap(feconf, 'EARLIEST_SUPPORTED_STATE_SCHEMA_VERSION', 60)
        swap_current_state_61 = self.swap(feconf, 'CURRENT_STATE_SCHEMA_VERSION', 61)
        with swap_earlier_state_to_60, swap_current_state_61:
            exploration_model = exp_models.ExplorationModel.get('fake_eid', strict=True, version=None)
            error_regex = 'Sorry, we can only process v%d\\-v%d exploration state schemas at present.' % (feconf.EARLIEST_SUPPORTED_STATE_SCHEMA_VERSION, feconf.CURRENT_STATE_SCHEMA_VERSION)
            with self.assertRaisesRegex(Exception, error_regex):
                exp_fetchers.get_exploration_from_model(exploration_model)

    def test_migration_then_reversion_maintains_valid_exploration(self) -> None:
        if False:
            while True:
                i = 10
        'This integration test simulates the behavior of the domain layer\n        prior to the introduction of a states schema. In particular, it deals\n        with an exploration that was created before any states schema\n        migrations occur. The exploration is constructed using multiple change\n        lists, then a migration is run. The test thereafter tests if\n        reverting to a version prior to the migration still maintains a valid\n        exploration. It tests both the exploration domain object and the\n        exploration model stored in the datastore for validity.\n        Note: It is important to distinguish between when the test is testing\n        the exploration domain versus its model. It is operating at the domain\n        layer when using exp_fetchers.get_exploration_by_id. Otherwise, it\n        loads the model explicitly using exp_models.ExplorationModel.get and\n        then converts it to an exploration domain object for validation using\n        exp_fetchers.get_exploration_from_model. This is NOT the same process\n        as exp_fetchers.get_exploration_by_id as it skips many steps which\n        include the conversion pipeline (which is crucial to this test).\n        '
        exp_id: str = 'exp_id2'
        end_state_name: str = 'End'
        swap_states_schema_41 = self.swap(feconf, 'CURRENT_STATE_SCHEMA_VERSION', 41)
        swap_exp_schema_46 = self.swap(exp_domain.Exploration, 'CURRENT_EXP_SCHEMA_VERSION', 46)
        with swap_states_schema_41, swap_exp_schema_46:
            exploration = exp_domain.Exploration.create_default_exploration(exp_id, title='Old Title', category='Art', objective='Exp objective...')
            exploration_model = exp_models.ExplorationModel(id=exp_id)
            exp_services.populate_exp_model_fields(exploration_model, exploration)
        exploration_model.states = self.STATES_AT_V41
        rights_manager.create_new_exploration_rights(exploration_model.id, self.albert_id)
        exploration_model.commit(self.albert_id, 'Created new exploration.', [])
        exp_services.regenerate_exploration_summary_with_new_contributor(exp_id, self.albert_id)
        stats_services.create_exp_issues_for_new_exploration(exploration_model.id, exploration_model.version)
        caching_services.delete_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, None, [exp_id])
        exploration_model = exp_models.ExplorationModel.get(exp_id, strict=True, version=None)
        exploration_model.title = 'New title'
        exploration_model.commit(self.albert_id, 'Changed title and states.', [])
        exploration_model = exp_models.ExplorationModel.get(exp_id, strict=True, version=None)
        exp_fetchers.get_exploration_from_model(exploration_model)
        exploration_model.states['New state'] = {'solicit_answer_details': False, 'written_translations': {'translations_mapping': {'content': {}, 'default_outcome': {}, 'ca_placeholder_0': {}}}, 'recorded_voiceovers': {'voiceovers_mapping': {'content': {}, 'default_outcome': {}, 'ca_placeholder_0': {}}}, 'param_changes': [], 'classifier_model_id': None, 'content': {'content_id': 'content', 'html': '<p>Unicode Characters 😍😍😍😍</p>'}, 'next_content_id_index': 5, 'interaction': {'answer_groups': [], 'confirmed_unclassified_answers': [], 'customization_args': {'buttonText': {'value': {'content_id': 'ca_placeholder_0', 'unicode_str': 'Click me!'}}}, 'default_outcome': {'dest': end_state_name, 'feedback': {'content_id': 'default_outcome', 'html': ''}, 'labelled_as_correct': False, 'missing_prerequisite_skill_id': None, 'param_changes': [], 'refresher_exploration_id': None}, 'hints': [], 'id': 'Continue', 'solution': None}}
        init_state = exploration_model.states[feconf.DEFAULT_INIT_STATE_NAME]
        init_state['interaction']['default_outcome']['dest'] = 'New state'
        exploration_model.commit('committer_id_v3', 'Added new state', [])
        exploration_model = exp_models.ExplorationModel.get(exp_id, strict=True, version=None)
        commit_cmds = [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_MIGRATE_STATES_SCHEMA_TO_LATEST_VERSION, 'from_version': str(exploration_model.states_schema_version), 'to_version': str(feconf.CURRENT_STATE_SCHEMA_VERSION)})]
        exp_services.update_exploration(feconf.MIGRATION_BOT_USERNAME, exploration_model.id, commit_cmds, 'Update exploration states from schema version %d to %d.' % (exploration_model.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION))
        exploration_model = exp_models.ExplorationModel.get(exp_id, strict=True, version=None)
        exploration = exp_fetchers.get_exploration_from_model(exploration_model, run_conversion=False)
        self.assertEqual(exploration.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)
        exploration.validate(strict=True)
        exp_services.revert_exploration('committer_id_v4', exp_id, 4, 1)
        exploration_model = exp_models.ExplorationModel.get(exp_id, strict=True, version=None)
        self.assertEqual(exploration_model.states_schema_version, 41)
        exploration = exp_fetchers.get_exploration_by_id(exp_id)
        self.assertEqual(exploration.to_yaml(), '%sversion: 5\n' % self.UPGRADED_EXP_YAML)
        exploration.validate(strict=True)
        snapshots_metadata = exp_services.get_exploration_snapshots_metadata(exp_id)
        commit_dict_5 = {'committer_id': 'committer_id_v4', 'commit_message': 'Reverted exploration to version 1', 'version_number': 5}
        commit_dict_4 = {'committer_id': feconf.MIGRATION_BOT_USERNAME, 'commit_message': 'Update exploration states from schema version 41 to %d.' % feconf.CURRENT_STATE_SCHEMA_VERSION, 'commit_cmds': [{'cmd': exp_domain.CMD_MIGRATE_STATES_SCHEMA_TO_LATEST_VERSION, 'from_version': '41', 'to_version': str(feconf.CURRENT_STATE_SCHEMA_VERSION)}], 'version_number': 4}
        self.assertEqual(len(snapshots_metadata), 5)
        self.assertDictEqual(snapshots_metadata[3], {**snapshots_metadata[3], **commit_dict_4})
        self.assertDictEqual(snapshots_metadata[4], {**snapshots_metadata[4], **commit_dict_5})
        self.assertLess(snapshots_metadata[3]['created_on_ms'], snapshots_metadata[4]['created_on_ms'])
        exp_services.update_exploration(self.albert_id, exp_id, [], 'Resave after reversion')
        exploration_model = exp_models.ExplorationModel.get(exp_id, strict=True, version=None)
        exploration = exp_fetchers.get_exploration_from_model(exploration_model, run_conversion=False)
        self.assertEqual(exploration.to_yaml(), '%sversion: 6\n' % self.UPGRADED_EXP_YAML)
        exploration.validate()