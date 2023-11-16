"""Unit tests for jobs.batch_jobs.exp_migration_jobs."""
from __future__ import annotations
from core import feconf
from core import utils
from core.constants import constants
from core.domain import caching_services
from core.domain import exp_domain
from core.domain import exp_fetchers
from core.domain import exp_services
from core.domain import opportunity_services
from core.domain import rights_domain
from core.domain import rights_manager
from core.domain import story_domain
from core.domain import story_services
from core.domain import topic_domain
from core.domain import topic_services
from core.domain import translation_domain
from core.domain import translation_services
from core.domain import user_services
from core.jobs import job_test_utils
from core.jobs.batch_jobs import exp_migration_jobs
from core.jobs.types import job_run_result
from core.platform import models
from core.tests import test_utils
from typing import Sequence
MYPY = False
if MYPY:
    from mypy_imports import exp_models
    from mypy_imports import opportunity_models
    from mypy_imports import stats_models
    from mypy_imports import translation_models
(exp_models, opportunity_models, stats_models, translation_models) = models.Registry.import_models([models.Names.EXPLORATION, models.Names.OPPORTUNITY, models.Names.STATISTICS, models.Names.TRANSLATION])
EXP_V46_DICT = utils.dict_from_yaml("\nauthor_notes: ''\nauto_tts_enabled: true\nblurb: ''\ncategory: Art\ncorrectness_feedback_enabled: true\nedits_allowed: true\ninit_state_name: (untitled state)\nlanguage_code: en\nobjective: Objective for the exploration...\nparam_changes: []\nparam_specs: {}\nschema_version: 46\nstates:\n  (untitled state):\n    classifier_model_id: null\n    content:\n      content_id: content\n      html: ''\n    interaction:\n      answer_groups:\n      - outcome:\n          dest: END\n          feedback:\n            content_id: feedback_1\n            html: <p>Correct!</p>\n          labelled_as_correct: false\n          missing_prerequisite_skill_id: null\n          param_changes: []\n          refresher_exploration_id: null\n        rule_specs:\n        - inputs:\n            x:\n            - - <p>Choice 1</p>\n              - <p>Choice 2</p>\n          rule_type: IsEqualToOrdering\n        - inputs:\n            x:\n            - - <p>Choice 1</p>\n          rule_type: IsEqualToOrderingWithOneItemAtIncorrectPosition\n        - inputs:\n            x: <p>Choice 1</p>\n            y: 1\n          rule_type: HasElementXAtPositionY\n        - inputs:\n            x: <p>Choice 1</p>\n            y: <p>Choice 2</p>\n          rule_type: HasElementXBeforeElementY\n        tagged_skill_misconception_id: null\n        training_data: []\n      confirmed_unclassified_answers: []\n      customization_args:\n        allowMultipleItemsInSamePosition:\n          value: true\n        choices:\n          value:\n          - content_id: ca_choices_2\n            html: <p>Choice 1</p>\n          - content_id: ca_choices_3\n            html: <p>Choice 2</p>\n      default_outcome:\n        dest: (untitled state)\n        feedback:\n          content_id: default_outcome\n          html: ''\n        labelled_as_correct: false\n        missing_prerequisite_skill_id: null\n        param_changes: []\n        refresher_exploration_id: null\n      hints: []\n      id: DragAndDropSortInput\n      solution: null\n    linked_skill_id: null\n    next_content_id_index: 4\n    param_changes: []\n    recorded_voiceovers:\n      voiceovers_mapping:\n        ca_choices_2: {}\n        ca_choices_3: {}\n        content: {}\n        default_outcome: {}\n        feedback_1: {}\n    solicit_answer_details: false\n    written_translations:\n      translations_mapping:\n        ca_choices_2: {}\n        ca_choices_3: {}\n        content: {}\n        default_outcome: {}\n        feedback_1: {}\n  END:\n    classifier_model_id: null\n    content:\n      content_id: content\n      html: <p>Congratulations, you have finished!</p>\n    interaction:\n      answer_groups: []\n      confirmed_unclassified_answers: []\n      customization_args:\n        recommendedExplorationIds:\n          value: []\n      default_outcome: null\n      hints: []\n      id: EndExploration\n      solution: null\n    linked_skill_id: null\n    next_content_id_index: 0\n    param_changes: []\n    recorded_voiceovers:\n      voiceovers_mapping:\n        content: {}\n    solicit_answer_details: false\n    written_translations:\n      translations_mapping:\n        content: {}\nstates_schema_version: 41\ntags: []\ntitle: Title of exploration\n")

class MigrateExplorationJobTests(job_test_utils.JobTestBase, test_utils.GenericTestBase):
    JOB_CLASS = exp_migration_jobs.MigrateExplorationJob
    NEW_EXP_ID = 'exp_1'
    EXP_ID_ONE = 'exp_one'
    EXP_ID_TWO = 'exp_two'
    EXP_TITLE = 'title'

    def test_empty_storage(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_job_output_is_empty()

    def test_migrated_exp_is_not_migrated(self) -> None:
        if False:
            i = 10
            return i + 15
        exploration = exp_domain.Exploration.create_default_exploration(self.NEW_EXP_ID, title=self.EXP_TITLE, category='category')
        exp_services.save_new_exploration(feconf.SYSTEM_COMMITTER_ID, exploration)
        self.assertEqual(exploration.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='EXP PREVIOUSLY MIGRATED SUCCESS: 1'), job_run_result.JobRunResult(stdout='EXP PROCESSED SUCCESS: 1')])
        exp_model = exp_models.ExplorationModel.get(self.NEW_EXP_ID)
        self.assertEqual(exp_model.version, 1)

    def test_broken_exp_is_not_migrated(self) -> None:
        if False:
            return 10
        exploration_rights = rights_domain.ActivityRights(self.EXP_ID_ONE, [feconf.SYSTEM_COMMITTER_ID], [], [], [])
        commit_cmds = [{'cmd': rights_domain.CMD_CREATE_NEW}]
        exp_models.ExplorationRightsModel(id=exploration_rights.id, owner_ids=exploration_rights.owner_ids, editor_ids=exploration_rights.editor_ids, voice_artist_ids=exploration_rights.voice_artist_ids, viewer_ids=exploration_rights.viewer_ids, community_owned=exploration_rights.community_owned, status=exploration_rights.status, viewable_if_private=exploration_rights.viewable_if_private, first_published_msec=exploration_rights.first_published_msec).commit(feconf.SYSTEM_COMMITTER_ID, 'Created new exploration', commit_cmds)
        exp_model = self.create_model(exp_models.ExplorationModel, id=self.EXP_ID_ONE, title='title', category=' category', init_state_name='Introduction', states_schema_version=49)
        exp_model.update_timestamps()
        exp_model.commit(feconf.SYSTEM_COMMITTER_ID, 'Create exploration', [{'cmd': exp_domain.CMD_CREATE_NEW}])
        exp_model = exp_models.ExplorationModel(id=self.EXP_ID_TWO, category=EXP_V46_DICT['category'], title=EXP_V46_DICT['title'], objective=EXP_V46_DICT['objective'], language_code=EXP_V46_DICT['language_code'], tags=EXP_V46_DICT['tags'], blurb=EXP_V46_DICT['blurb'], author_notes=EXP_V46_DICT['author_notes'], states_schema_version=EXP_V46_DICT['states_schema_version'], init_state_name=EXP_V46_DICT['init_state_name'], states=EXP_V46_DICT['states'], auto_tts_enabled=EXP_V46_DICT['auto_tts_enabled'], correctness_feedback_enabled=EXP_V46_DICT['correctness_feedback_enabled'])
        rights_manager.create_new_exploration_rights(self.EXP_ID_TWO, feconf.SYSTEM_COMMITTER_ID)
        exp_model.commit(feconf.SYSTEM_COMMITTER_ID, 'Created new exploration', commit_cmds)
        self.assertEqual(exp_model.states_schema_version, 41)
        self.assert_job_output_is([job_run_result.JobRunResult(stderr='EXP PROCESSED ERROR: "(\'exp_one\', ValidationError(\'Names should not start or end with whitespace.\'))": 1'), job_run_result.JobRunResult(stdout='EXP PROCESSED SUCCESS: 1')])
        migrated_exp_model = exp_models.ExplorationModel.get(self.EXP_ID_ONE)
        self.assertEqual(migrated_exp_model.version, 1)
        migrated_exp_model = exp_models.ExplorationModel.get(self.EXP_ID_TWO)
        self.assertEqual(migrated_exp_model.version, 1)

    def create_story_linked_to_exploration(self) -> None:
        if False:
            while True:
                i = 10
        'Creates a new story linked to the test exploration.'
        topic_id = 'topic_id_1'
        story_id = 'story_id_1'
        topic = topic_domain.Topic.create_default_topic(topic_id, 'topic', 'abbrev', 'description', 'fragment')
        topic.thumbnail_filename = 'thumbnail.svg'
        topic.thumbnail_bg_color = '#C6DCDA'
        topic.subtopics = [topic_domain.Subtopic(1, 'Title', ['skill_id_1'], 'image.svg', constants.ALLOWED_THUMBNAIL_BG_COLORS['subtopic'][0], 21131, 'dummy-subtopic-url')]
        topic.next_subtopic_id = 2
        topic.skill_ids_for_diagnostic_test = ['skill_id_1']
        topic_services.save_new_topic(feconf.SYSTEM_COMMITTER_ID, topic)
        topic_services.publish_topic(topic_id, feconf.SYSTEM_COMMITTER_ID)
        story = story_domain.Story.create_default_story(story_id, 'A story title', 'description', topic_id, 'story-one')
        story_services.save_new_story(feconf.SYSTEM_COMMITTER_ID, story)
        topic_services.add_canonical_story(feconf.SYSTEM_COMMITTER_ID, topic_id, story_id)
        topic_services.publish_story(topic_id, story_id, feconf.SYSTEM_COMMITTER_ID)
        change_list = [story_domain.StoryChange({'cmd': story_domain.CMD_ADD_STORY_NODE, 'node_id': story_domain.NODE_ID_PREFIX + '1', 'title': 'Title 1'}), story_domain.StoryChange({'cmd': story_domain.CMD_UPDATE_STORY_NODE_PROPERTY, 'property_name': story_domain.STORY_NODE_PROPERTY_EXPLORATION_ID, 'node_id': story_domain.NODE_ID_PREFIX + '1', 'old_value': None, 'new_value': self.NEW_EXP_ID})]
        story_services.update_story(feconf.SYSTEM_COMMITTER_ID, story_id, change_list, 'Added node.')

    def test_unmigrated_valid_published_exp_migrates(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        exp_model = exp_models.ExplorationModel(id=self.NEW_EXP_ID, category=EXP_V46_DICT['category'], title=EXP_V46_DICT['title'], objective=EXP_V46_DICT['objective'], language_code=EXP_V46_DICT['language_code'], tags=EXP_V46_DICT['tags'], blurb=EXP_V46_DICT['blurb'], author_notes=EXP_V46_DICT['author_notes'], states_schema_version=EXP_V46_DICT['states_schema_version'], init_state_name=EXP_V46_DICT['init_state_name'], states=EXP_V46_DICT['states'], auto_tts_enabled=EXP_V46_DICT['auto_tts_enabled'], correctness_feedback_enabled=EXP_V46_DICT['correctness_feedback_enabled'])
        rights_manager.create_new_exploration_rights(self.NEW_EXP_ID, feconf.SYSTEM_COMMITTER_ID)
        exp_model.commit(feconf.SYSTEM_COMMITTER_ID, '', [])
        exp_summary_model = exp_models.ExpSummaryModel(**{'id': self.NEW_EXP_ID, 'title': exp_model.title, 'category': exp_model.category, 'objective': exp_model.objective, 'language_code': exp_model.language_code, 'tags': exp_model.tags, 'ratings': None, 'scaled_average_rating': 4.0, 'exploration_model_last_updated': exp_model.last_updated, 'exploration_model_created_on': exp_model.created_on, 'first_published_msec': None, 'status': constants.ACTIVITY_STATUS_PRIVATE, 'community_owned': False, 'owner_ids': [feconf.SYSTEM_COMMITTER_ID], 'editor_ids': [], 'voice_artist_ids': [], 'viewer_ids': [], 'contributor_ids': [], 'contributors_summary': {}, 'version': exp_model.version})
        exp_summary_model.update_timestamps()
        exp_summary_model.put()
        for lang_code in ['hi', 'bn']:
            translation_models.EntityTranslationsModel.create_new(feconf.TranslatableEntityType.EXPLORATION.value, exp_model.id, exp_model.version, lang_code, {}).put()
        all_translation_models: Sequence[translation_models.EntityTranslationsModel] = translation_models.EntityTranslationsModel.get_all().fetch()
        self.assertEqual(len(all_translation_models), 2)
        owner_action = user_services.get_user_actions_info(feconf.SYSTEM_COMMITTER_ID)
        exp_services.publish_exploration_and_update_user_profiles(owner_action, self.NEW_EXP_ID)
        opportunity_model = opportunity_models.ExplorationOpportunitySummaryModel(id=self.NEW_EXP_ID, topic_id='topic_id1', topic_name='topic', story_id='story_id_1', story_title='A story title', chapter_title='Title 1', content_count=20, incomplete_translation_language_codes=['hi', 'ar'], translation_counts={'hi': 1, 'ar': 2}, language_codes_needing_voice_artists=['en'], language_codes_with_assigned_voice_artists=[])
        opportunity_model.put()
        self.create_story_linked_to_exploration()
        self.assertEqual(exp_model.states_schema_version, 41)
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='EXP MIGRATED SUCCESS: 1'), job_run_result.JobRunResult(stdout='EXP PROCESSED SUCCESS: 1'), job_run_result.JobRunResult(stdout='EXP RELATED MODELS GENERATED SUCCESS: 1')])
        updated_opp_model = opportunity_models.ExplorationOpportunitySummaryModel.get(self.NEW_EXP_ID)
        updated_opp_summary = opportunity_services.get_exploration_opportunity_summary_from_model(updated_opp_model)
        expected_opp_summary_dict = {'id': 'exp_1', 'topic_name': 'topic', 'chapter_title': 'Title 1', 'story_title': 'A story title', 'content_count': 4, 'translation_counts': {'hi': 0, 'bn': 0}, 'translation_in_review_counts': {}, 'is_pinned': False}
        self.assertEqual(updated_opp_summary.to_dict(), expected_opp_summary_dict)
        all_translation_models = translation_models.EntityTranslationsModel.get_all().fetch()
        self.assertEqual(len(all_translation_models), 4)
        self.assertItemsEqual([m.entity_version for m in all_translation_models], [1, 1, 2, 2])

    def test_unmigrated_invalid_published_exp_raise_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        exp_model = exp_models.ExplorationModel(id=self.NEW_EXP_ID, category=EXP_V46_DICT['category'], title='', objective=EXP_V46_DICT['objective'], language_code=EXP_V46_DICT['language_code'], tags=EXP_V46_DICT['tags'], blurb=EXP_V46_DICT['blurb'], author_notes=EXP_V46_DICT['author_notes'], states_schema_version=EXP_V46_DICT['states_schema_version'], init_state_name=EXP_V46_DICT['init_state_name'], states=EXP_V46_DICT['states'], auto_tts_enabled=EXP_V46_DICT['auto_tts_enabled'], correctness_feedback_enabled=EXP_V46_DICT['correctness_feedback_enabled'])
        rights_manager.create_new_exploration_rights(self.NEW_EXP_ID, feconf.SYSTEM_COMMITTER_ID)
        exp_model.commit(feconf.SYSTEM_COMMITTER_ID, '', [])
        exp_summary_model = exp_models.ExpSummaryModel(**{'id': self.NEW_EXP_ID, 'title': exp_model.title, 'category': exp_model.category, 'objective': exp_model.objective, 'language_code': exp_model.language_code, 'tags': exp_model.tags, 'ratings': None, 'scaled_average_rating': 4.0, 'exploration_model_last_updated': exp_model.last_updated, 'exploration_model_created_on': exp_model.created_on, 'first_published_msec': None, 'status': constants.ACTIVITY_STATUS_PRIVATE, 'community_owned': False, 'owner_ids': [feconf.SYSTEM_COMMITTER_ID], 'editor_ids': [], 'voice_artist_ids': [], 'viewer_ids': [], 'contributor_ids': [], 'contributors_summary': {}, 'version': exp_model.version})
        exp_summary_model.update_timestamps()
        exp_summary_model.put()
        owner_action = user_services.get_user_actions_info(feconf.SYSTEM_COMMITTER_ID)
        exp_services.publish_exploration_and_update_user_profiles(owner_action, self.NEW_EXP_ID)
        opportunity_model = opportunity_models.ExplorationOpportunitySummaryModel(id=self.NEW_EXP_ID, topic_id='topic_id1', topic_name='topic', story_id='story_id_1', story_title='A story title', chapter_title='Title 1', content_count=20, incomplete_translation_language_codes=['hi', 'ar'], translation_counts={'hi': 1, 'ar': 2}, language_codes_needing_voice_artists=['en'], language_codes_with_assigned_voice_artists=[])
        opportunity_model.put()
        self.create_story_linked_to_exploration()
        self.assertEqual(exp_model.states_schema_version, 41)
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_1\', ValidationError("Please fix the following issues before saving this exploration: 1. A title must be specified (in the \'Settings\' tab). "))": 1')])

    def test_unmigrated_exp_with_invalid_related_data_raise_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        exp_model = exp_models.ExplorationModel(id=self.NEW_EXP_ID, category=EXP_V46_DICT['category'], title='A title', objective=EXP_V46_DICT['objective'], language_code=EXP_V46_DICT['language_code'], tags=EXP_V46_DICT['tags'], blurb=EXP_V46_DICT['blurb'], author_notes=EXP_V46_DICT['author_notes'], states_schema_version=EXP_V46_DICT['states_schema_version'], init_state_name=EXP_V46_DICT['init_state_name'], states=EXP_V46_DICT['states'], auto_tts_enabled=EXP_V46_DICT['auto_tts_enabled'], correctness_feedback_enabled=EXP_V46_DICT['correctness_feedback_enabled'])
        rights_manager.create_new_exploration_rights(self.NEW_EXP_ID, feconf.SYSTEM_COMMITTER_ID)
        exp_model.commit(feconf.SYSTEM_COMMITTER_ID, '', [])
        exp_summary_model = exp_models.ExpSummaryModel(**{'id': self.NEW_EXP_ID, 'title': exp_model.title, 'category': exp_model.category, 'objective': exp_model.objective, 'language_code': exp_model.language_code, 'tags': exp_model.tags, 'ratings': None, 'scaled_average_rating': 4.0, 'exploration_model_last_updated': exp_model.last_updated, 'exploration_model_created_on': exp_model.created_on, 'first_published_msec': None, 'status': constants.ACTIVITY_STATUS_PRIVATE, 'community_owned': False, 'owner_ids': [feconf.SYSTEM_COMMITTER_ID], 'editor_ids': [], 'voice_artist_ids': [], 'viewer_ids': [], 'contributor_ids': [], 'contributors_summary': {}, 'version': exp_model.version})
        exp_summary_model.update_timestamps()
        exp_summary_model.put()
        owner_action = user_services.get_user_actions_info(feconf.SYSTEM_COMMITTER_ID)
        exp_services.publish_exploration_and_update_user_profiles(owner_action, self.NEW_EXP_ID)
        opportunity_model = opportunity_models.ExplorationOpportunitySummaryModel(id=self.NEW_EXP_ID, topic_id='topic_id1', topic_name='topic', story_id='story_id_1', story_title='A story title', chapter_title='Title 1', content_count=20, incomplete_translation_language_codes=['hi', 'ar'], translation_counts={'hi': 1, 'ar': 2}, language_codes_needing_voice_artists=['en'], language_codes_with_assigned_voice_artists=[])
        opportunity_model.put()
        self.create_story_linked_to_exploration()
        self.assertEqual(exp_model.states_schema_version, 41)
        with self.swap_to_always_raise(translation_services, 'compute_translation_related_change', Exception('Error generating related models')):
            self.assert_job_output_is([job_run_result.JobRunResult(stderr='EXP RELATED MODELS GENERATED ERROR: "(\'exp_1\', Exception(\'Error generating related models\'))": 1'), job_run_result.JobRunResult(stdout='EXP PROCESSED SUCCESS: 1'), job_run_result.JobRunResult(stdout='EXP MIGRATED SUCCESS: 1')])

class AuditExplorationMigrationJobTests(job_test_utils.JobTestBase, test_utils.GenericTestBase):
    JOB_CLASS = exp_migration_jobs.AuditExplorationMigrationJob
    NEW_EXP_ID = 'exp_1'
    EXP_TITLE = 'title'

    def test_empty_storage(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assert_job_output_is_empty()

    def test_migrated_exp_is_not_migrated(self) -> None:
        if False:
            return 10
        exploration = exp_domain.Exploration.create_default_exploration(self.NEW_EXP_ID, title=self.EXP_TITLE, category='category')
        exp_services.save_new_exploration(feconf.SYSTEM_COMMITTER_ID, exploration)
        self.assertEqual(exploration.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='EXP PREVIOUSLY MIGRATED SUCCESS: 1'), job_run_result.JobRunResult(stdout='EXP PROCESSED SUCCESS: 1')])
        exp_model = exp_models.ExplorationModel.get(self.NEW_EXP_ID)
        self.assertEqual(exp_model.version, 1)

    def test_broken_exp_is_not_migrated(self) -> None:
        if False:
            i = 10
            return i + 15
        exploration_rights = rights_domain.ActivityRights(self.NEW_EXP_ID, [feconf.SYSTEM_COMMITTER_ID], [], [], [])
        commit_cmds = [{'cmd': rights_domain.CMD_CREATE_NEW}]
        exp_models.ExplorationRightsModel(id=exploration_rights.id, owner_ids=exploration_rights.owner_ids, editor_ids=exploration_rights.editor_ids, voice_artist_ids=exploration_rights.voice_artist_ids, viewer_ids=exploration_rights.viewer_ids, community_owned=exploration_rights.community_owned, status=exploration_rights.status, viewable_if_private=exploration_rights.viewable_if_private, first_published_msec=exploration_rights.first_published_msec).commit(feconf.SYSTEM_COMMITTER_ID, 'Created new exploration', commit_cmds)
        exp_model = self.create_model(exp_models.ExplorationModel, id=self.NEW_EXP_ID, title='title', category=' category', init_state_name='Introduction', states_schema_version=49)
        exp_model.update_timestamps()
        exp_model.commit(feconf.SYSTEM_COMMITTER_ID, 'Create exploration', [{'cmd': exp_domain.CMD_CREATE_NEW}])
        self.assert_job_output_is([job_run_result.JobRunResult(stderr='EXP PROCESSED ERROR: "(\'exp_1\', ValidationError(\'Names should not start or end with whitespace.\'))": 1')])
        migrated_exp_model = exp_models.ExplorationModel.get(self.NEW_EXP_ID)
        self.assertEqual(migrated_exp_model.version, 1)

    def create_story_linked_to_exploration(self) -> None:
        if False:
            i = 10
            return i + 15
        'Creates a new story linked to the test exploration.'
        topic_id = 'topic_id_1'
        story_id = 'story_id_1'
        topic = topic_domain.Topic.create_default_topic(topic_id, 'topic', 'abbrev', 'description', 'fragment')
        topic.thumbnail_filename = 'thumbnail.svg'
        topic.thumbnail_bg_color = '#C6DCDA'
        topic.subtopics = [topic_domain.Subtopic(1, 'Title', ['skill_id_1'], 'image.svg', constants.ALLOWED_THUMBNAIL_BG_COLORS['subtopic'][0], 21131, 'dummy-subtopic-url')]
        topic.next_subtopic_id = 2
        topic.skill_ids_for_diagnostic_test = ['skill_id_1']
        topic_services.save_new_topic(feconf.SYSTEM_COMMITTER_ID, topic)
        topic_services.publish_topic(topic_id, feconf.SYSTEM_COMMITTER_ID)
        story = story_domain.Story.create_default_story(story_id, 'A story title', 'description', topic_id, 'story-one')
        story_services.save_new_story(feconf.SYSTEM_COMMITTER_ID, story)
        topic_services.add_canonical_story(feconf.SYSTEM_COMMITTER_ID, topic_id, story_id)
        topic_services.publish_story(topic_id, story_id, feconf.SYSTEM_COMMITTER_ID)
        change_list = [story_domain.StoryChange({'cmd': story_domain.CMD_ADD_STORY_NODE, 'node_id': story_domain.NODE_ID_PREFIX + '1', 'title': 'Title 1'}), story_domain.StoryChange({'cmd': story_domain.CMD_UPDATE_STORY_NODE_PROPERTY, 'property_name': story_domain.STORY_NODE_PROPERTY_EXPLORATION_ID, 'node_id': story_domain.NODE_ID_PREFIX + '1', 'old_value': None, 'new_value': self.NEW_EXP_ID})]
        story_services.update_story(feconf.SYSTEM_COMMITTER_ID, story_id, change_list, 'Added node.')

    def test_unmigrated_exp_is_migrated(self) -> None:
        if False:
            return 10
        exp_model = exp_models.ExplorationModel(id=self.NEW_EXP_ID, category=EXP_V46_DICT['category'], title=EXP_V46_DICT['title'], objective=EXP_V46_DICT['objective'], language_code=EXP_V46_DICT['language_code'], tags=EXP_V46_DICT['tags'], blurb=EXP_V46_DICT['blurb'], author_notes=EXP_V46_DICT['author_notes'], states_schema_version=EXP_V46_DICT['states_schema_version'], init_state_name=EXP_V46_DICT['init_state_name'], states=EXP_V46_DICT['states'], auto_tts_enabled=EXP_V46_DICT['auto_tts_enabled'], correctness_feedback_enabled=EXP_V46_DICT['correctness_feedback_enabled'])
        rights_manager.create_new_exploration_rights(self.NEW_EXP_ID, feconf.SYSTEM_COMMITTER_ID)
        exp_model.commit(feconf.SYSTEM_COMMITTER_ID, '', [])
        exp_summary_model = exp_models.ExpSummaryModel(**{'id': self.NEW_EXP_ID, 'title': exp_model.title, 'category': exp_model.category, 'objective': exp_model.objective, 'language_code': exp_model.language_code, 'tags': exp_model.tags, 'ratings': None, 'scaled_average_rating': 4.0, 'exploration_model_last_updated': exp_model.last_updated, 'exploration_model_created_on': exp_model.created_on, 'first_published_msec': None, 'status': constants.ACTIVITY_STATUS_PRIVATE, 'community_owned': False, 'owner_ids': [feconf.SYSTEM_COMMITTER_ID], 'editor_ids': [], 'voice_artist_ids': [], 'viewer_ids': [], 'contributor_ids': [], 'contributors_summary': {}, 'version': exp_model.version})
        exp_summary_model.update_timestamps()
        exp_summary_model.put()
        translation_models.EntityTranslationsModel.create_new(feconf.TranslatableEntityType.EXPLORATION.value, exp_model.id, exp_model.version, 'hi', {}).put()
        all_translation_models: Sequence[translation_models.EntityTranslationsModel] = translation_models.EntityTranslationsModel.get_all().fetch()
        self.assertEqual(len(all_translation_models), 1)
        owner_action = user_services.get_user_actions_info(feconf.SYSTEM_COMMITTER_ID)
        exp_services.publish_exploration_and_update_user_profiles(owner_action, self.NEW_EXP_ID)
        opportunity_model = opportunity_models.ExplorationOpportunitySummaryModel(id=self.NEW_EXP_ID, topic_id='topic_id1', topic_name='topic', story_id='story_id_1', story_title='A story title', chapter_title='Title 1', content_count=20, incomplete_translation_language_codes=['hi', 'ar'], translation_counts={'hi': 1, 'ar': 2}, language_codes_needing_voice_artists=['en'], language_codes_with_assigned_voice_artists=[])
        opportunity_model.put()
        self.create_story_linked_to_exploration()
        exploration_model = exp_models.ExplorationModel.get(self.NEW_EXP_ID)
        self.assertEqual(exploration_model.states_schema_version, 41)
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='EXP PROCESSED SUCCESS: 1'), job_run_result.JobRunResult(stdout='EXP MIGRATED SUCCESS: 1', stderr='')])
        updated_opp_model = opportunity_models.ExplorationOpportunitySummaryModel.get(self.NEW_EXP_ID)
        updated_opp_summary = opportunity_services.get_exploration_opportunity_summary_from_model(updated_opp_model)
        expected_opp_summary_dict = {'id': 'exp_1', 'topic_name': 'topic', 'chapter_title': 'Title 1', 'story_title': 'A story title', 'content_count': 4, 'translation_counts': {'hi': 0}, 'translation_in_review_counts': {}, 'is_pinned': False}
        self.assertEqual(updated_opp_summary.to_dict(), expected_opp_summary_dict)

    def test_unmigrated_invalid_published_exp_raise_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        exp_model = exp_models.ExplorationModel(id=self.NEW_EXP_ID, category=EXP_V46_DICT['category'], title='', objective=EXP_V46_DICT['objective'], language_code=EXP_V46_DICT['language_code'], tags=EXP_V46_DICT['tags'], blurb=EXP_V46_DICT['blurb'], author_notes=EXP_V46_DICT['author_notes'], states_schema_version=EXP_V46_DICT['states_schema_version'], init_state_name=EXP_V46_DICT['init_state_name'], states=EXP_V46_DICT['states'], auto_tts_enabled=EXP_V46_DICT['auto_tts_enabled'], correctness_feedback_enabled=EXP_V46_DICT['correctness_feedback_enabled'])
        rights_manager.create_new_exploration_rights(self.NEW_EXP_ID, feconf.SYSTEM_COMMITTER_ID)
        exp_model.commit(feconf.SYSTEM_COMMITTER_ID, '', [])
        exp_summary_model = exp_models.ExpSummaryModel(**{'id': self.NEW_EXP_ID, 'title': exp_model.title, 'category': exp_model.category, 'objective': exp_model.objective, 'language_code': exp_model.language_code, 'tags': exp_model.tags, 'ratings': None, 'scaled_average_rating': 4.0, 'exploration_model_last_updated': exp_model.last_updated, 'exploration_model_created_on': exp_model.created_on, 'first_published_msec': None, 'status': constants.ACTIVITY_STATUS_PRIVATE, 'community_owned': False, 'owner_ids': [feconf.SYSTEM_COMMITTER_ID], 'editor_ids': [], 'voice_artist_ids': [], 'viewer_ids': [], 'contributor_ids': [], 'contributors_summary': {}, 'version': exp_model.version})
        exp_summary_model.update_timestamps()
        exp_summary_model.put()
        owner_action = user_services.get_user_actions_info(feconf.SYSTEM_COMMITTER_ID)
        exp_services.publish_exploration_and_update_user_profiles(owner_action, self.NEW_EXP_ID)
        opportunity_model = opportunity_models.ExplorationOpportunitySummaryModel(id=self.NEW_EXP_ID, topic_id='topic_id1', topic_name='topic', story_id='story_id_1', story_title='A story title', chapter_title='Title 1', content_count=20, incomplete_translation_language_codes=['hi', 'ar'], translation_counts={'hi': 1, 'ar': 2}, language_codes_needing_voice_artists=['en'], language_codes_with_assigned_voice_artists=[])
        opportunity_model.put()
        self.create_story_linked_to_exploration()
        self.assertEqual(exp_model.states_schema_version, 41)
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_1\', ValidationError("Please fix the following issues before saving this exploration: 1. A title must be specified (in the \'Settings\' tab). "))": 1')])

class RegenerateMissingExplorationStatsModelsJobTests(job_test_utils.JobTestBase, test_utils.GenericTestBase):
    """Tests for the RegenerateExplorationStatsJob."""
    JOB_CLASS = exp_migration_jobs.RegenerateMissingExplorationStatsModelsJob
    NEW_EXP_ID = 'exp_1'
    EXP_TITLE = 'title'

    def test_empty_storage(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_job_output_is_empty()

    def test_job_regenerates_missing_stats_models(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        exp_id = 'ID1'
        owner_id = 'owner_id'
        self.save_new_default_exploration(exp_id, 'owner_id')
        exp_services.update_exploration(owner_id, exp_id, [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_EXPLORATION_PROPERTY, 'property_name': 'title', 'new_value': 'New title 1'})], 'Changed title.')
        exp_services.update_exploration(owner_id, exp_id, [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_EXPLORATION_PROPERTY, 'property_name': 'title', 'new_value': 'New title 2'})], 'Changed title.')
        exp_services.update_exploration(owner_id, exp_id, [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_EXPLORATION_PROPERTY, 'property_name': 'title', 'new_value': 'New title 3'})], 'Changed title.')
        exp_services.update_exploration(owner_id, exp_id, [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_EXPLORATION_PROPERTY, 'property_name': 'title', 'new_value': 'New title 4'})], 'Changed title.')
        exp_services.update_exploration(owner_id, exp_id, [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_EDIT_EXPLORATION_PROPERTY, 'property_name': 'title', 'new_value': 'New title 5'})], 'Changed title.')
        exp_stats_model_for_version_2 = stats_models.ExplorationStatsModel.get_model(exp_id, 2)
        assert exp_stats_model_for_version_2 is not None
        exp_stats_model_for_version_2.delete()
        exp_stats_model_for_version_4 = stats_models.ExplorationStatsModel.get_model(exp_id, 4)
        assert exp_stats_model_for_version_4 is not None
        exp_stats_model_for_version_4.delete()
        self.assertIsNone(stats_models.ExplorationStatsModel.get_model(exp_id, 2))
        self.assertIsNone(stats_models.ExplorationStatsModel.get_model(exp_id, 4))
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='EXP PROCESSED SUCCESS: 1')])
        self.assertIsNotNone(stats_models.ExplorationStatsModel.get_model(exp_id, 2))
        self.assertIsNotNone(stats_models.ExplorationStatsModel.get_model(exp_id, 4))

    def test_job_regenerates_missing_stats_models_when_no_models_exist(self) -> None:
        if False:
            return 10
        exp_id = 'ID1'
        self.save_new_default_exploration(exp_id, 'owner_id')
        exp_stats_model_for_version_1 = stats_models.ExplorationStatsModel.get_model(exp_id, 1)
        assert exp_stats_model_for_version_1 is not None
        exp_stats_model_for_version_1.delete()
        self.assertIsNone(stats_models.ExplorationStatsModel.get_model(exp_id, 1))
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'ID1\', Exception(\'No ExplorationStatsModels found\'))": 1')])
        self.assertIsNotNone(stats_models.ExplorationStatsModel.get_model(exp_id, 1))

class ExpSnapshotsMigrationAuditJobTests(job_test_utils.JobTestBase, test_utils.GenericTestBase):
    """Tests for ExplorationMigrationAuditJob."""
    JOB_CLASS = exp_migration_jobs.ExpSnapshotsMigrationAuditJob
    ALBERT_EMAIL = 'albert@example.com'
    ALBERT_NAME = 'albert'
    VALID_EXP_ID = 'exp_id0'
    NEW_EXP_ID = 'exp_id1'
    EXP_TITLE = 'title'

    def test_migration_audit_job_does_not_convert_up_to_date_exp(self) -> None:
        if False:
            while True:
                i = 10
        'Tests that the snapshot migration audit job does not convert a\n        snapshot that is already the latest states schema version.\n        '
        exploration = exp_domain.Exploration.create_default_exploration(self.VALID_EXP_ID, title='title', category='category')
        content_id_generator = translation_domain.ContentIdGenerator()
        init_state = exploration.states[exploration.init_state_name]
        self.set_interaction_for_state(init_state, 'EndExploration', content_id_generator)
        exploration.next_content_id_index = content_id_generator.next_content_id_index
        init_state.update_interaction_default_outcome(None)
        exp_services.save_new_exploration(feconf.SYSTEM_COMMITTER_ID, exploration)
        self.assertEqual(exploration.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception(\'Snapshot is already at latest schema version\'))": 1')])

    def test_migration_audit_job_skips_deleted_explorations(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests that the snapshot migration job skips deleted explorations\n        and does not attempt to migrate any of the snapshots.\n        '
        exploration = exp_domain.Exploration.create_default_exploration(self.VALID_EXP_ID, title='title', category='category')
        content_id_generator = translation_domain.ContentIdGenerator()
        init_state = exploration.states[exploration.init_state_name]
        self.set_interaction_for_state(init_state, 'EndExploration', content_id_generator)
        exploration.next_content_id_index = content_id_generator.next_content_id_index
        init_state.update_interaction_default_outcome(None)
        exp_services.save_new_exploration(feconf.SYSTEM_COMMITTER_ID, exploration)
        exp_services.regenerate_exploration_and_contributors_summaries(self.VALID_EXP_ID)
        exp_services.delete_exploration(feconf.SYSTEM_COMMITTER_ID, self.VALID_EXP_ID)
        with self.assertRaisesRegex(Exception, 'Entity .* not found'):
            exp_fetchers.get_exploration_by_id(self.VALID_EXP_ID)
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception(\'Exploration does not exist.\'))": 2')])
        with self.assertRaisesRegex(Exception, 'Entity .* not found'):
            exp_fetchers.get_exploration_by_id(self.NEW_EXP_ID)

    def test_migration_job_audit_success(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that the audit job runs correctly on snapshots that use a\n        previous state schema.\n        '
        exp_model = exp_models.ExplorationModel(id=self.VALID_EXP_ID, category=EXP_V46_DICT['category'], title=EXP_V46_DICT['title'], objective=EXP_V46_DICT['objective'], language_code=EXP_V46_DICT['language_code'], tags=EXP_V46_DICT['tags'], blurb=EXP_V46_DICT['blurb'], author_notes=EXP_V46_DICT['author_notes'], states_schema_version=EXP_V46_DICT['states_schema_version'], init_state_name=EXP_V46_DICT['init_state_name'], states=EXP_V46_DICT['states'], auto_tts_enabled=EXP_V46_DICT['auto_tts_enabled'], correctness_feedback_enabled=EXP_V46_DICT['correctness_feedback_enabled'])
        rights_manager.create_new_exploration_rights(self.VALID_EXP_ID, feconf.SYSTEM_COMMITTER_ID)
        exp_model.commit(feconf.SYSTEM_COMMITTER_ID, '', [])
        exp_summary_model = exp_models.ExpSummaryModel(**{'id': self.VALID_EXP_ID, 'title': exp_model.title, 'category': exp_model.category, 'objective': exp_model.objective, 'language_code': exp_model.language_code, 'tags': exp_model.tags, 'ratings': None, 'scaled_average_rating': 4.0, 'exploration_model_last_updated': exp_model.last_updated, 'exploration_model_created_on': exp_model.created_on, 'first_published_msec': None, 'status': constants.ACTIVITY_STATUS_PRIVATE, 'community_owned': False, 'owner_ids': [feconf.SYSTEM_COMMITTER_ID], 'editor_ids': [], 'voice_artist_ids': [], 'viewer_ids': [], 'contributor_ids': [], 'contributors_summary': {}, 'version': exp_model.version})
        exp_summary_model.update_timestamps()
        exp_summary_model.put()
        self.assertLess(exp_model.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)
        caching_services.delete_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, None, [self.VALID_EXP_ID])
        latest_schema_version = str(feconf.CURRENT_STATE_SCHEMA_VERSION)
        migration_change_list = [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_MIGRATE_STATES_SCHEMA_TO_LATEST_VERSION, 'from_version': '46', 'to_version': latest_schema_version})]
        exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.VALID_EXP_ID, migration_change_list, 'Ran Exploration Migration job.')
        exploration_model = exp_models.ExplorationModel.get(self.VALID_EXP_ID)
        self.assertEqual(exploration_model.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception(\'Snapshot is already at latest schema version\'))": 1'), job_run_result.JobRunResult(stdout='EXP PROCESSED SUCCESS: 1', stderr='')])

    def test_migration_job_audit_failure(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that the audit job catches any errors that would otherwise\n        occur during the migration.\n        '
        exp_model = exp_models.ExplorationModel(id=self.VALID_EXP_ID, category=EXP_V46_DICT['category'], title=EXP_V46_DICT['title'], objective=EXP_V46_DICT['objective'], language_code=EXP_V46_DICT['language_code'], tags=EXP_V46_DICT['tags'], blurb=EXP_V46_DICT['blurb'], author_notes=EXP_V46_DICT['author_notes'], states_schema_version=EXP_V46_DICT['states_schema_version'], init_state_name=EXP_V46_DICT['init_state_name'], states=EXP_V46_DICT['states'], auto_tts_enabled=EXP_V46_DICT['auto_tts_enabled'], correctness_feedback_enabled=EXP_V46_DICT['correctness_feedback_enabled'])
        rights_manager.create_new_exploration_rights(self.VALID_EXP_ID, feconf.SYSTEM_COMMITTER_ID)
        exp_model.commit(feconf.SYSTEM_COMMITTER_ID, '', [])
        exp_summary_model = exp_models.ExpSummaryModel(**{'id': self.VALID_EXP_ID, 'title': exp_model.title, 'category': exp_model.category, 'objective': exp_model.objective, 'language_code': exp_model.language_code, 'tags': exp_model.tags, 'ratings': None, 'scaled_average_rating': 4.0, 'exploration_model_last_updated': exp_model.last_updated, 'exploration_model_created_on': exp_model.created_on, 'first_published_msec': None, 'status': constants.ACTIVITY_STATUS_PRIVATE, 'community_owned': False, 'owner_ids': [feconf.SYSTEM_COMMITTER_ID], 'editor_ids': [], 'voice_artist_ids': [], 'viewer_ids': [], 'contributor_ids': [], 'contributors_summary': {}, 'version': exp_model.version})
        exp_summary_model.update_timestamps()
        exp_summary_model.put()
        caching_services.delete_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, None, [self.VALID_EXP_ID])
        latest_schema_version = str(feconf.CURRENT_STATE_SCHEMA_VERSION)
        migration_change_list = [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_MIGRATE_STATES_SCHEMA_TO_LATEST_VERSION, 'from_version': '42', 'to_version': latest_schema_version})]
        exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.VALID_EXP_ID, migration_change_list, 'Ran Exploration Migration job.')
        exploration_model = exp_models.ExplorationModel.get(self.VALID_EXP_ID)
        self.assertEqual(exploration_model.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)
        mock_conversion = classmethod(lambda cls, exploration_dict: exploration_dict['property_that_dne'])
        with self.swap(exp_domain.Exploration, '_convert_states_v46_dict_to_v47_dict', mock_conversion):
            self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception("Exploration snapshot exp_id0 failed migration to states v47: \'property_that_dne\'"))": 1'), job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception(\'Snapshot is already at latest schema version\'))": 1')])

    def test_audit_job_detects_invalid_exploration(self) -> None:
        if False:
            i = 10
            return i + 15
        exploration = exp_domain.Exploration.create_default_exploration(self.VALID_EXP_ID, title='title', category='category')
        exp_services.save_new_exploration(feconf.SYSTEM_COMMITTER_ID, exploration)
        exploration_model = exp_models.ExplorationModel.get(self.VALID_EXP_ID)
        exploration_model.language_code = 'invalid_language_code'
        exploration_model.commit(feconf.SYSTEM_COMMITTER_ID, 'Changed language_code.', [])
        caching_services.delete_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, None, [self.VALID_EXP_ID])
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception(\'Exploration exp_id0 failed non-strict validation\'))": 2')])

    def test_audit_job_detects_exploration_that_is_not_up_to_date(self) -> None:
        if False:
            while True:
                i = 10
        swap_states_schema_41 = self.swap(feconf, 'CURRENT_STATE_SCHEMA_VERSION', 41)
        swap_exp_schema_46 = self.swap(exp_domain.Exploration, 'CURRENT_EXP_SCHEMA_VERSION', 46)
        with swap_states_schema_41, swap_exp_schema_46:
            exploration = exp_domain.Exploration.create_default_exploration(self.VALID_EXP_ID, title='title', category='category')
            exp_services.save_new_exploration(feconf.SYSTEM_COMMITTER_ID, exploration)
        self.assertLess(exploration.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)
        swap_states_schema_42 = self.swap(feconf, 'CURRENT_STATE_SCHEMA_VERSION', 42)
        swap_exp_schema_47 = self.swap(exp_domain.Exploration, 'CURRENT_EXP_SCHEMA_VERSION', 47)
        with swap_states_schema_42, swap_exp_schema_47:
            self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception(\'Exploration is not at latest schema version\'))": 1')])

    def test_audit_job_handles_missing_states_schema_version(self) -> None:
        if False:
            i = 10
            return i + 15
        swap_exp_schema_37 = self.swap(exp_domain.Exploration, 'CURRENT_EXP_SCHEMA_VERSION', 37)
        with swap_exp_schema_37:
            with self.swap(feconf, 'CURRENT_STATE_SCHEMA_VERSION', 44):
                exploration = exp_domain.Exploration.create_default_exploration(self.VALID_EXP_ID, title='title', category='category')
                exp_services.save_new_exploration(feconf.SYSTEM_COMMITTER_ID, exploration)
            caching_services.delete_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, None, [self.VALID_EXP_ID])
            migration_change_list = [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_MIGRATE_STATES_SCHEMA_TO_LATEST_VERSION, 'from_version': '41', 'to_version': '44'})]
            with self.swap(feconf, 'CURRENT_STATE_SCHEMA_VERSION', 44):
                exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.VALID_EXP_ID, migration_change_list, 'Ran Exploration Migration job.')
            exploration_model = exp_models.ExplorationModel.get(self.VALID_EXP_ID)
            self.assertEqual(exploration_model.states_schema_version, 44)
            snapshot_content_model = exp_models.ExplorationSnapshotContentModel.get('%s-1' % self.VALID_EXP_ID)
            del snapshot_content_model.content['states_schema_version']
            snapshot_content_model.update_timestamps(update_last_updated_time=False)
            snapshot_content_model.put()
            with self.swap(feconf, 'CURRENT_STATE_SCHEMA_VERSION', 44):
                self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception("Exploration snapshot exp_id0 failed migration to states v1: type object \'Exploration\' has no attribute \'_convert_states_v0_dict_to_v1_dict\'"))": 1'), job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception(\'Snapshot is already at latest schema version\'))": 1')])

class ExpSnapshotsMigrationJobTests(job_test_utils.JobTestBase, test_utils.GenericTestBase):
    JOB_CLASS = exp_migration_jobs.ExpSnapshotsMigrationJob
    ALBERT_EMAIL = 'albert@example.com'
    ALBERT_NAME = 'albert'
    VALID_EXP_ID = 'exp_id0'
    NEW_EXP_ID = 'exp_id1'
    EXP_TITLE = 'title'

    def test_migration_job_does_not_convert_up_to_date_exp(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that the exploration migration job does not convert a\n        snapshot that is already the latest states schema version.\n        '
        exploration = exp_domain.Exploration.create_default_exploration(self.VALID_EXP_ID, title='title', category='category')
        content_id_generator = translation_domain.ContentIdGenerator()
        init_state = exploration.states[exploration.init_state_name]
        self.set_interaction_for_state(init_state, 'EndExploration', content_id_generator)
        exploration.next_content_id_index = content_id_generator.next_content_id_index
        init_state.update_interaction_default_outcome(None)
        exp_services.save_new_exploration(feconf.SYSTEM_COMMITTER_ID, exploration)
        self.assertEqual(exploration.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception(\'Snapshot is already at latest schema version\'))": 1')])

    def test_migration_job_succeeds_on_default_exploration(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        exp_model = exp_models.ExplorationModel(id=self.VALID_EXP_ID, category=EXP_V46_DICT['category'], title=EXP_V46_DICT['title'], objective=EXP_V46_DICT['objective'], language_code=EXP_V46_DICT['language_code'], tags=EXP_V46_DICT['tags'], blurb=EXP_V46_DICT['blurb'], author_notes=EXP_V46_DICT['author_notes'], states_schema_version=EXP_V46_DICT['states_schema_version'], init_state_name=EXP_V46_DICT['init_state_name'], states=EXP_V46_DICT['states'], auto_tts_enabled=EXP_V46_DICT['auto_tts_enabled'], correctness_feedback_enabled=EXP_V46_DICT['correctness_feedback_enabled'])
        rights_manager.create_new_exploration_rights(self.VALID_EXP_ID, feconf.SYSTEM_COMMITTER_ID)
        exp_model.commit(feconf.SYSTEM_COMMITTER_ID, '', [])
        exp_summary_model = exp_models.ExpSummaryModel(**{'id': self.VALID_EXP_ID, 'title': exp_model.title, 'category': exp_model.category, 'objective': exp_model.objective, 'language_code': exp_model.language_code, 'tags': exp_model.tags, 'ratings': None, 'scaled_average_rating': 4.0, 'exploration_model_last_updated': exp_model.last_updated, 'exploration_model_created_on': exp_model.created_on, 'first_published_msec': None, 'status': constants.ACTIVITY_STATUS_PRIVATE, 'community_owned': False, 'owner_ids': [feconf.SYSTEM_COMMITTER_ID], 'editor_ids': [], 'voice_artist_ids': [], 'viewer_ids': [], 'contributor_ids': [], 'contributors_summary': {}, 'version': exp_model.version})
        exp_summary_model.update_timestamps()
        exp_summary_model.put()
        caching_services.delete_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, None, [self.VALID_EXP_ID])
        latest_schema_version = str(feconf.CURRENT_STATE_SCHEMA_VERSION)
        migration_change_list = [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_MIGRATE_STATES_SCHEMA_TO_LATEST_VERSION, 'from_version': '46', 'to_version': latest_schema_version})]
        exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.VALID_EXP_ID, migration_change_list, 'Ran Exploration Migration job.')
        exploration_model = exp_models.ExplorationModel.get(self.VALID_EXP_ID)
        self.assertEqual(exploration_model.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception(\'Snapshot is already at latest schema version\'))": 1'), job_run_result.JobRunResult(stdout='EXP PROCESSED SUCCESS: 1', stderr='')])

    def test_migration_job_skips_deleted_explorations(self) -> None:
        if False:
            print('Hello World!')
        'Tests that the exploration migration job skips deleted explorations\n        and does not attempt to migrate.\n        '
        exploration = exp_domain.Exploration.create_default_exploration(self.VALID_EXP_ID, title='title', category='category')
        content_id_generator = translation_domain.ContentIdGenerator()
        init_state = exploration.states[exploration.init_state_name]
        self.set_interaction_for_state(init_state, 'EndExploration', content_id_generator)
        exploration.next_content_id_index = content_id_generator.next_content_id_index
        init_state.update_interaction_default_outcome(None)
        exp_services.save_new_exploration(feconf.SYSTEM_COMMITTER_ID, exploration)
        exp_services.regenerate_exploration_and_contributors_summaries(self.VALID_EXP_ID)
        exp_services.delete_exploration(feconf.SYSTEM_COMMITTER_ID, self.VALID_EXP_ID)
        with self.assertRaisesRegex(Exception, 'Entity .* not found'):
            exp_fetchers.get_exploration_by_id(self.VALID_EXP_ID)
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception(\'Exploration does not exist.\'))": 2')])
        with self.assertRaisesRegex(Exception, 'Entity .* not found'):
            exp_fetchers.get_exploration_by_id(self.NEW_EXP_ID)

    def test_migration_job_detects_invalid_exploration(self) -> None:
        if False:
            print('Hello World!')
        exploration = exp_domain.Exploration.create_default_exploration(self.VALID_EXP_ID, title='title', category='category')
        exp_services.save_new_exploration(feconf.SYSTEM_COMMITTER_ID, exploration)
        exploration_model = exp_models.ExplorationModel.get(self.VALID_EXP_ID)
        exploration_model.language_code = 'invalid_language_code'
        exploration_model.commit(feconf.SYSTEM_COMMITTER_ID, 'Changed language_code.', [])
        caching_services.delete_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, None, [self.VALID_EXP_ID])
        self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception(\'Exploration exp_id0 failed non-strict validation\'))": 2')])

    def test_migration_job_detects_exploration_that_is_not_up_to_date(self) -> None:
        if False:
            while True:
                i = 10
        swap_states_schema_41 = self.swap(feconf, 'CURRENT_STATE_SCHEMA_VERSION', 41)
        swap_exp_schema_46 = self.swap(exp_domain.Exploration, 'CURRENT_EXP_SCHEMA_VERSION', 46)
        with swap_states_schema_41, swap_exp_schema_46:
            exploration = exp_domain.Exploration.create_default_exploration(self.VALID_EXP_ID, title='title', category='category')
            exp_services.save_new_exploration(feconf.SYSTEM_COMMITTER_ID, exploration)
        self.assertLess(exploration.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)
        swap_states_schema_42 = self.swap(feconf, 'CURRENT_STATE_SCHEMA_VERSION', 42)
        swap_exp_schema_47 = self.swap(exp_domain.Exploration, 'CURRENT_EXP_SCHEMA_VERSION', 47)
        with swap_states_schema_42, swap_exp_schema_47:
            self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception(\'Exploration is not at latest schema version\'))": 1')])

    def test_migration_job_audit_failure(self) -> None:
        if False:
            print('Hello World!')
        exp_model = exp_models.ExplorationModel(id=self.VALID_EXP_ID, category=EXP_V46_DICT['category'], title=EXP_V46_DICT['title'], objective=EXP_V46_DICT['objective'], language_code=EXP_V46_DICT['language_code'], tags=EXP_V46_DICT['tags'], blurb=EXP_V46_DICT['blurb'], author_notes=EXP_V46_DICT['author_notes'], states_schema_version=EXP_V46_DICT['states_schema_version'], init_state_name=EXP_V46_DICT['init_state_name'], states=EXP_V46_DICT['states'], auto_tts_enabled=EXP_V46_DICT['auto_tts_enabled'], correctness_feedback_enabled=EXP_V46_DICT['correctness_feedback_enabled'])
        rights_manager.create_new_exploration_rights(self.VALID_EXP_ID, feconf.SYSTEM_COMMITTER_ID)
        exp_model.commit(feconf.SYSTEM_COMMITTER_ID, '', [])
        exp_summary_model = exp_models.ExpSummaryModel(**{'id': self.VALID_EXP_ID, 'title': exp_model.title, 'category': exp_model.category, 'objective': exp_model.objective, 'language_code': exp_model.language_code, 'tags': exp_model.tags, 'ratings': None, 'scaled_average_rating': 4.0, 'exploration_model_last_updated': exp_model.last_updated, 'exploration_model_created_on': exp_model.created_on, 'first_published_msec': None, 'status': constants.ACTIVITY_STATUS_PRIVATE, 'community_owned': False, 'owner_ids': [feconf.SYSTEM_COMMITTER_ID], 'editor_ids': [], 'voice_artist_ids': [], 'viewer_ids': [], 'contributor_ids': [], 'contributors_summary': {}, 'version': exp_model.version})
        exp_summary_model.update_timestamps()
        exp_summary_model.put()
        caching_services.delete_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, None, [self.VALID_EXP_ID])
        latest_schema_version = str(feconf.CURRENT_STATE_SCHEMA_VERSION)
        migration_change_list = [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_MIGRATE_STATES_SCHEMA_TO_LATEST_VERSION, 'from_version': '46', 'to_version': latest_schema_version})]
        exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.VALID_EXP_ID, migration_change_list, 'Ran Exploration Migration job.')
        exploration_model = exp_models.ExplorationModel.get(self.VALID_EXP_ID)
        self.assertEqual(exploration_model.states_schema_version, feconf.CURRENT_STATE_SCHEMA_VERSION)
        mock_conversion = classmethod(lambda cls, exploration_dict: exploration_dict['property_that_dne'])
        with self.swap(exp_domain.Exploration, '_convert_states_v46_dict_to_v47_dict', mock_conversion):
            self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception("Exploration snapshot exp_id0 failed migration to states v47: \'property_that_dne\'"))": 1'), job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception(\'Snapshot is already at latest schema version\'))": 1')])

    def test_audit_job_handles_missing_states_schema_version(self) -> None:
        if False:
            return 10
        swap_exp_schema_37 = self.swap(exp_domain.Exploration, 'CURRENT_EXP_SCHEMA_VERSION', 37)
        with swap_exp_schema_37:
            with self.swap(feconf, 'CURRENT_STATE_SCHEMA_VERSION', 44):
                exploration = exp_domain.Exploration.create_default_exploration(self.VALID_EXP_ID, title='title', category='category')
                exp_services.save_new_exploration(feconf.SYSTEM_COMMITTER_ID, exploration)
            caching_services.delete_multi(caching_services.CACHE_NAMESPACE_EXPLORATION, None, [self.VALID_EXP_ID])
            migration_change_list = [exp_domain.ExplorationChange({'cmd': exp_domain.CMD_MIGRATE_STATES_SCHEMA_TO_LATEST_VERSION, 'from_version': '41', 'to_version': '44'})]
            with self.swap(feconf, 'CURRENT_STATE_SCHEMA_VERSION', 44):
                exp_services.update_exploration(feconf.SYSTEM_COMMITTER_ID, self.VALID_EXP_ID, migration_change_list, 'Ran Exploration Migration job.')
            exploration_model = exp_models.ExplorationModel.get(self.VALID_EXP_ID)
            self.assertEqual(exploration_model.states_schema_version, 44)
            snapshot_content_model = exp_models.ExplorationSnapshotContentModel.get('%s-1' % self.VALID_EXP_ID)
            del snapshot_content_model.content['states_schema_version']
            snapshot_content_model.update_timestamps(update_last_updated_time=False)
            snapshot_content_model.put()
            with self.swap(feconf, 'CURRENT_STATE_SCHEMA_VERSION', 44):
                self.assert_job_output_is([job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception("Exploration snapshot exp_id0 failed migration to states v1: type object \'Exploration\' has no attribute \'_convert_states_v0_dict_to_v1_dict\'"))": 1'), job_run_result.JobRunResult(stdout='', stderr='EXP PROCESSED ERROR: "(\'exp_id0\', Exception(\'Snapshot is already at latest schema version\'))": 1')])