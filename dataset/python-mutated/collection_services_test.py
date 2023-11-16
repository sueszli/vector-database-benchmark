"""Unit tests for core.domain.collection_services."""
from __future__ import annotations
import datetime
import logging
import os
from core import feconf
from core import utils
from core.constants import constants
from core.domain import collection_domain
from core.domain import collection_services
from core.domain import rights_domain
from core.domain import rights_manager
from core.domain import user_services
from core.platform import models
from core.tests import test_utils
from typing import Dict, Final, List, Optional
MYPY = False
if MYPY:
    from mypy_imports import collection_models
    from mypy_imports import datastore_services
    from mypy_imports import search_services as gae_search_services
    from mypy_imports import user_models
(collection_models, user_models) = models.Registry.import_models([models.Names.COLLECTION, models.Names.USER])
datastore_services = models.Registry.import_datastore_services()
gae_search_services = models.Registry.import_search_services()

def count_at_least_editable_collection_summaries(user_id: str) -> int:
    if False:
        while True:
            i = 10
    'Returns the count of collection summaries that are atleast editable.'
    return len(collection_services.get_collection_summary_dicts_from_models(collection_models.CollectionSummaryModel.get_at_least_editable(user_id=user_id)))

class CollectionServicesUnitTests(test_utils.GenericTestBase):
    """Test the collection services module."""
    COLLECTION_0_ID: Final = 'A_collection_0_id'
    COLLECTION_1_ID: Final = 'A_collection_1_id'

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        'Before each individual test, create dummy users.'
        super().setUp()
        self.signup(self.OWNER_EMAIL, self.OWNER_USERNAME)
        self.signup(self.EDITOR_EMAIL, self.EDITOR_USERNAME)
        self.signup(self.VIEWER_EMAIL, self.VIEWER_USERNAME)
        self.signup(self.CURRICULUM_ADMIN_EMAIL, self.CURRICULUM_ADMIN_USERNAME)
        self.owner_id = self.get_user_id_from_email(self.OWNER_EMAIL)
        self.editor_id = self.get_user_id_from_email(self.EDITOR_EMAIL)
        self.viewer_id = self.get_user_id_from_email(self.VIEWER_EMAIL)
        self.set_curriculum_admins([self.CURRICULUM_ADMIN_USERNAME])
        self.user_id_admin = self.get_user_id_from_email(self.CURRICULUM_ADMIN_EMAIL)
        self.owner = user_services.get_user_actions_info(self.owner_id)

class MockCollectionModel(collection_models.CollectionModel):
    nodes = datastore_services.JsonProperty(repeated=True)

class CollectionQueriesUnitTests(CollectionServicesUnitTests):
    """Tests query methods."""

    def test_get_collection_titles_and_categories(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(collection_services.get_collection_titles_and_categories([]), {})
        self.save_new_default_collection('A', self.owner_id, title='TitleA')
        self.assertEqual(collection_services.get_collection_titles_and_categories(['A']), {'A': {'category': 'A category', 'title': 'TitleA'}})
        self.save_new_default_collection('B', self.owner_id, title='TitleB')
        self.assertEqual(collection_services.get_collection_titles_and_categories(['A']), {'A': {'category': 'A category', 'title': 'TitleA'}})
        self.assertEqual(collection_services.get_collection_titles_and_categories(['A', 'B']), {'A': {'category': 'A category', 'title': 'TitleA'}, 'B': {'category': 'A category', 'title': 'TitleB'}})
        self.assertEqual(collection_services.get_collection_titles_and_categories(['A', 'C']), {'A': {'category': 'A category', 'title': 'TitleA'}})

    def test_get_collection_from_model(self) -> None:
        if False:
            while True:
                i = 10
        rights_manager.create_new_collection_rights('collection_id', self.owner_id)
        collection_model = collection_models.CollectionModel(id='collection_id', category='category', title='title', objective='objective', collection_contents={'nodes': {}})
        collection_model.commit(self.owner_id, 'collection model created', [{'cmd': 'create_new', 'title': 'title', 'category': 'category'}])
        collection = collection_services.get_collection_from_model(collection_model)
        self.assertEqual(collection.id, 'collection_id')
        self.assertEqual(collection.title, 'title')
        self.assertEqual(collection.category, 'category')
        self.assertEqual(collection.objective, 'objective')
        self.assertEqual(collection.language_code, constants.DEFAULT_LANGUAGE_CODE)
        self.assertEqual(collection.version, 1)
        self.assertEqual(collection.schema_version, feconf.CURRENT_COLLECTION_SCHEMA_VERSION)

    def test_get_collection_from_model_with_schema_version_2_copies_nodes(self) -> None:
        if False:
            return 10
        collection_model = MockCollectionModel(id='collection_id', category='category', title='title', schema_version=2, objective='objective', version=1, nodes=[{'exploration_id': 'exp_id1', 'acquired_skills': ['11'], 'prerequisite_skills': ['22']}, {'exploration_id': 'exp_id2', 'acquired_skills': ['33'], 'prerequisite_skills': ['44']}])
        collection = collection_services.get_collection_from_model(collection_model)
        self.assertEqual(collection.id, 'collection_id')
        self.assertEqual(collection.title, 'title')
        self.assertEqual(collection.category, 'category')
        self.assertEqual(collection.objective, 'objective')
        self.assertEqual(collection.language_code, constants.DEFAULT_LANGUAGE_CODE)
        self.assertEqual(collection.version, 1)
        self.assertEqual(collection.nodes[0].to_dict(), {'exploration_id': 'exp_id1'})
        self.assertEqual(collection.nodes[1].to_dict(), {'exploration_id': 'exp_id2'})
        self.assertEqual(collection.schema_version, feconf.CURRENT_COLLECTION_SCHEMA_VERSION)

    def test_get_collection_from_model_with_invalid_schema_version_raises_error(self) -> None:
        if False:
            while True:
                i = 10
        rights_manager.create_new_collection_rights('collection_id', self.owner_id)
        collection_model = collection_models.CollectionModel(id='collection_id', category='category', title='title', schema_version=0, objective='objective', collection_contents={'nodes': {}})
        collection_model.commit(self.owner_id, 'collection model created', [{'cmd': 'create_new', 'title': 'title', 'category': 'category'}])
        with self.assertRaisesRegex(Exception, 'Sorry, we can only process v1-v%d collection schemas at present.' % feconf.CURRENT_COLLECTION_SCHEMA_VERSION):
            collection_services.get_collection_from_model(collection_model)

    def test_get_different_collections_by_version(self) -> None:
        if False:
            i = 10
            return i + 15
        self.save_new_valid_collection('collection_id', self.owner_id)
        collection_services.update_collection(self.owner_id, 'collection_id', [{'cmd': collection_domain.CMD_EDIT_COLLECTION_PROPERTY, 'property_name': 'objective', 'new_value': 'Some new objective'}, {'cmd': collection_domain.CMD_EDIT_COLLECTION_PROPERTY, 'property_name': 'title', 'new_value': 'Some new title'}, {'cmd': collection_domain.CMD_EDIT_COLLECTION_PROPERTY, 'property_name': 'category', 'new_value': 'Some new category'}], 'Changed properties')
        collection = collection_services.get_collection_by_id('collection_id', version=1)
        self.assertEqual(collection.id, 'collection_id')
        self.assertEqual(collection.category, 'A category')
        self.assertEqual(collection.objective, 'An objective')
        self.assertEqual(collection.language_code, constants.DEFAULT_LANGUAGE_CODE)
        self.assertEqual(collection.schema_version, feconf.CURRENT_COLLECTION_SCHEMA_VERSION)
        collection = collection_services.get_collection_by_id('collection_id', version=0)
        self.assertEqual(collection.id, 'collection_id')
        self.assertEqual(collection.title, 'Some new title')
        self.assertEqual(collection.category, 'Some new category')
        self.assertEqual(collection.objective, 'Some new objective')
        self.assertEqual(collection.language_code, constants.DEFAULT_LANGUAGE_CODE)
        self.assertEqual(collection.schema_version, feconf.CURRENT_COLLECTION_SCHEMA_VERSION)

    def test_get_collection_summary_by_id_with_invalid_collection_id(self) -> None:
        if False:
            print('Hello World!')
        collection = collection_services.get_collection_summary_by_id('invalid_collection_id')
        self.assertIsNone(collection)

    def test_save_collection_without_change_list_raises_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        collection = self.save_new_valid_collection('collection_id', self.owner_id)
        apply_change_list_swap = self.swap(collection_services, 'apply_change_list', lambda _, __: collection)
        with apply_change_list_swap, self.assertRaisesRegex(Exception, 'Unexpected error: received an invalid change list when trying to save collection'):
            collection_services.update_collection(self.owner_id, 'collection_id', None, 'commit message')

    def test_save_collection_with_mismatch_of_versions_raises_error(self) -> None:
        if False:
            print('Hello World!')
        self.save_new_valid_collection('collection_id', self.owner_id)
        collection_model = collection_models.CollectionModel.get('collection_id')
        collection_model.version = 0
        with self.assertRaisesRegex(Exception, 'Unexpected error: trying to update version 0 of collection from version 1. Please reload the page and try again.'):
            collection_services.update_collection(self.owner_id, 'collection_id', [{'cmd': collection_domain.CMD_EDIT_COLLECTION_PROPERTY, 'property_name': 'objective', 'new_value': 'Some new objective'}], 'changed objective')

    def test_get_multiple_collections_from_model_by_id(self) -> None:
        if False:
            print('Hello World!')
        rights_manager.create_new_collection_rights('collection_id_1', self.owner_id)
        collection_model = collection_models.CollectionModel(id='collection_id_1', category='category 1', title='title 1', objective='objective 1', collection_contents={'nodes': {}})
        collection_model.commit(self.owner_id, 'collection model created', [{'cmd': 'create_new', 'title': 'title 1', 'category': 'category 1'}])
        rights_manager.create_new_collection_rights('collection_id_2', self.owner_id)
        collection_model = collection_models.CollectionModel(id='collection_id_2', category='category 2', title='title 2', objective='objective 2', collection_contents={'nodes': {}})
        collection_model.commit(self.owner_id, 'collection model created', [{'cmd': 'create_new', 'title': 'title 2', 'category': 'category 2'}])
        collections = collection_services.get_multiple_collections_by_id(['collection_id_1', 'collection_id_2'])
        self.assertEqual(len(collections), 2)
        self.assertEqual(collections['collection_id_1'].title, 'title 1')
        self.assertEqual(collections['collection_id_1'].category, 'category 1')
        self.assertEqual(collections['collection_id_1'].objective, 'objective 1')
        self.assertEqual(collections['collection_id_2'].title, 'title 2')
        self.assertEqual(collections['collection_id_2'].category, 'category 2')
        self.assertEqual(collections['collection_id_2'].objective, 'objective 2')

    def test_get_multiple_collections_by_id_with_invalid_collection_id(self) -> None:
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, "Couldn't find collections with the following ids"):
            collection_services.get_multiple_collections_by_id(['collection_id_1', 'collection_id_2'])

    def test_get_explorations_completed_in_collections(self) -> None:
        if False:
            while True:
                i = 10
        collection = self.save_new_valid_collection('collection_id', self.owner_id, exploration_id='exp_id')
        self.save_new_valid_exploration('exp_id_1', self.owner_id)
        collection.add_node('exp_id_1')
        completed_exp_ids = collection_services.get_explorations_completed_in_collections(self.owner_id, ['collection_id'])
        self.assertEqual(completed_exp_ids, [[]])
        collection_services.record_played_exploration_in_collection_context(self.owner_id, 'collection_id', 'exp_id')
        completed_exp_ids = collection_services.get_explorations_completed_in_collections(self.owner_id, ['collection_id'])
        self.assertEqual(completed_exp_ids, [['exp_id']])
        collection_services.record_played_exploration_in_collection_context(self.owner_id, 'collection_id', 'exp_id_1')
        completed_exp_ids = collection_services.get_explorations_completed_in_collections(self.owner_id, ['collection_id'])
        self.assertEqual(completed_exp_ids, [['exp_id', 'exp_id_1']])

    def test_update_collection_by_swapping_collection_nodes(self) -> None:
        if False:
            i = 10
            return i + 15
        collection = self.save_new_valid_collection('collection_id', self.owner_id, exploration_id='exp_id_1')
        self.save_new_valid_exploration('exp_id_2', self.owner_id)
        collection_services.update_collection(self.owner_id, 'collection_id', [{'cmd': collection_domain.CMD_ADD_COLLECTION_NODE, 'exploration_id': 'exp_id_2'}], 'Added new exploration')
        collection = collection_services.get_collection_by_id('collection_id')
        self.assertEqual(collection.nodes[0].exploration_id, 'exp_id_1')
        self.assertEqual(collection.nodes[1].exploration_id, 'exp_id_2')
        collection_services.update_collection(self.owner_id, 'collection_id', [{'cmd': collection_domain.CMD_SWAP_COLLECTION_NODES, 'first_index': 0, 'second_index': 1}], 'Swapped collection nodes')
        collection = collection_services.get_collection_by_id('collection_id')
        self.assertEqual(collection.nodes[0].exploration_id, 'exp_id_2')
        self.assertEqual(collection.nodes[1].exploration_id, 'exp_id_1')

    def test_update_collection_with_invalid_cmd_raises_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        observed_log_messages = []

        def _mock_logging_function(msg: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            'Mocks logging.error().'
            observed_log_messages.append(msg)
        logging_swap = self.swap(logging, 'error', _mock_logging_function)
        self.save_new_valid_collection('collection_id', self.owner_id)
        with self.assertRaisesRegex(Exception, 'Command invalid command is not allowed'), logging_swap:
            collection_services.update_collection(self.owner_id, 'collection_id', [{'cmd': 'invalid command'}], 'Commit message')
        self.assertEqual(len(observed_log_messages), 1)
        self.assertEqual(observed_log_messages[0], "ValidationError Command invalid command is not allowed collection_id [{'cmd': 'invalid command'}]")

class CollectionProgressUnitTests(CollectionServicesUnitTests):
    """Tests functions which deal with any progress a user has made within a
    collection, including query and recording methods related to explorations
    which are played in the context of the collection.
    """
    COL_ID_0: Final = '0_collection_id'
    COL_ID_1: Final = '1_collection_id'
    EXP_ID_0: Final = '0_exploration_id'
    EXP_ID_1: Final = '1_exploration_id'
    EXP_ID_2: Final = '2_exploration_id'

    def _get_progress_model(self, user_id: str, collection_id: str) -> Optional[user_models.CollectionProgressModel]:
        if False:
            while True:
                i = 10
        'Returns the CollectionProgressModel for the given user id and\n        collection id.\n        '
        return user_models.CollectionProgressModel.get(user_id, collection_id)

    def _record_completion(self, user_id: str, collection_id: str, exploration_id: str) -> None:
        if False:
            while True:
                i = 10
        'Records the played exploration in the collection by the user\n        corresponding to the given user id.\n        '
        collection_services.record_played_exploration_in_collection_context(user_id, collection_id, exploration_id)

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.save_new_valid_collection(self.COL_ID_0, self.owner_id, exploration_id=self.EXP_ID_0)
        for exp_id in [self.EXP_ID_1, self.EXP_ID_2]:
            self.save_new_valid_exploration(exp_id, self.owner_id)
            collection_services.update_collection(self.owner_id, self.COL_ID_0, [{'cmd': collection_domain.CMD_ADD_COLLECTION_NODE, 'exploration_id': exp_id}], 'Added new exploration')

    def test_get_completed_exploration_ids(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(collection_services.get_completed_exploration_ids('Fake', self.COL_ID_0), [])
        self.assertEqual(collection_services.get_completed_exploration_ids(self.owner_id, 'Fake'), [])
        self.assertIsNone(self._get_progress_model(self.owner_id, self.COL_ID_0))
        self.assertEqual(collection_services.get_completed_exploration_ids(self.owner_id, self.COL_ID_0), [])
        self._record_completion(self.owner_id, self.COL_ID_0, self.EXP_ID_0)
        self.assertIsNotNone(self._get_progress_model(self.owner_id, self.COL_ID_0))
        self.assertEqual(collection_services.get_completed_exploration_ids(self.owner_id, self.COL_ID_0), [self.EXP_ID_0])
        self._record_completion(self.owner_id, self.COL_ID_0, self.EXP_ID_2)
        self._record_completion(self.owner_id, self.COL_ID_0, self.EXP_ID_1)
        self.assertEqual(collection_services.get_completed_exploration_ids(self.owner_id, self.COL_ID_0), [self.EXP_ID_0, self.EXP_ID_2, self.EXP_ID_1])

    def test_get_next_exploration_id_to_complete_by_user(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(collection_services.get_next_exploration_id_to_complete_by_user('Fake', self.COL_ID_0), self.EXP_ID_0)
        with self.assertRaisesRegex(Exception, 'Entity for class CollectionModel with id Fake not found'):
            collection_services.get_next_exploration_id_to_complete_by_user(self.owner_id, 'Fake')
        self.assertEqual(collection_services.get_collection_by_id(self.COL_ID_0).first_exploration_id, self.EXP_ID_0)
        self.assertEqual(collection_services.get_next_exploration_id_to_complete_by_user(self.owner_id, self.COL_ID_0), self.EXP_ID_0)
        self._record_completion(self.owner_id, self.COL_ID_0, self.EXP_ID_0)
        self.assertEqual(collection_services.get_next_exploration_id_to_complete_by_user(self.owner_id, self.COL_ID_0), self.EXP_ID_1)
        self._record_completion(self.owner_id, self.COL_ID_0, self.EXP_ID_1)
        self._record_completion(self.owner_id, self.COL_ID_0, self.EXP_ID_2)
        self.assertEqual(collection_services.get_next_exploration_id_to_complete_by_user(self.owner_id, self.COL_ID_0), None)

    def test_record_played_exploration_in_collection_context(self) -> None:
        if False:
            return 10
        completion_model = self._get_progress_model(self.owner_id, self.COL_ID_0)
        self.assertIsNone(completion_model)
        collection_services.record_played_exploration_in_collection_context(self.owner_id, self.COL_ID_0, self.EXP_ID_0)
        completion_model = self._get_progress_model(self.owner_id, self.COL_ID_0)
        assert completion_model is not None
        self.assertEqual(completion_model.completed_explorations, [self.EXP_ID_0])
        collection_services.record_played_exploration_in_collection_context(self.owner_id, self.COL_ID_0, self.EXP_ID_0)
        completion_model = self._get_progress_model(self.owner_id, self.COL_ID_0)
        assert completion_model is not None
        self.assertEqual(completion_model.completed_explorations, [self.EXP_ID_0])
        self.save_new_default_collection(self.COL_ID_1, self.owner_id)
        collection_services.record_played_exploration_in_collection_context(self.owner_id, self.COL_ID_1, self.EXP_ID_0)
        collection_services.record_played_exploration_in_collection_context(self.owner_id, self.COL_ID_1, self.EXP_ID_1)
        completion_model = self._get_progress_model(self.owner_id, self.COL_ID_0)
        assert completion_model is not None
        self.assertEqual(completion_model.completed_explorations, [self.EXP_ID_0])
        collection_services.record_played_exploration_in_collection_context(self.owner_id, self.COL_ID_0, self.EXP_ID_2)
        collection_services.record_played_exploration_in_collection_context(self.owner_id, self.COL_ID_0, self.EXP_ID_1)
        completion_model = self._get_progress_model(self.owner_id, self.COL_ID_0)
        assert completion_model is not None
        self.assertEqual(completion_model.completed_explorations, [self.EXP_ID_0, self.EXP_ID_2, self.EXP_ID_1])

class CollectionSummaryQueriesUnitTests(CollectionServicesUnitTests):
    """Tests collection query methods which operate on CollectionSummary
    objects.
    """
    COL_ID_0: Final = '0_arch_bridges_in_england'
    COL_ID_1: Final = '1_welcome_introduce_oppia'
    COL_ID_2: Final = '2_welcome_introduce_oppia_interactions'
    COL_ID_3: Final = '3_welcome'
    COL_ID_4: Final = '4_languages_learning_basic_verbs_in_spanish'
    COL_ID_5: Final = '5_languages_private_collection_in_spanish'

    def setUp(self) -> None:
        if False:
            return 10
        super().setUp()
        self.save_new_default_collection(self.COL_ID_0, self.owner_id, title='Bridges in England', category='Architecture')
        self.save_new_default_collection(self.COL_ID_1, self.owner_id, title='Introduce Oppia', category='Welcome')
        self.save_new_default_collection(self.COL_ID_2, self.owner_id, title='Introduce Interactions in Oppia', category='Welcome')
        self.save_new_default_collection(self.COL_ID_3, self.owner_id, title='Welcome', category='Welcome')
        self.save_new_default_collection(self.COL_ID_4, self.owner_id, title='Learning basic verbs in Spanish', category='Languages')
        self.save_new_default_collection(self.COL_ID_5, self.owner_id, title='Private collection in Spanish', category='Languages')
        rights_manager.publish_collection(self.owner, self.COL_ID_0)
        rights_manager.publish_collection(self.owner, self.COL_ID_1)
        rights_manager.publish_collection(self.owner, self.COL_ID_2)
        rights_manager.publish_collection(self.owner, self.COL_ID_3)
        rights_manager.publish_collection(self.owner, self.COL_ID_4)
        collection_services.index_collections_given_ids([self.COL_ID_0, self.COL_ID_1, self.COL_ID_2, self.COL_ID_3, self.COL_ID_4])

    def _create_search_query(self, terms: List[str], categories: List[str]) -> str:
        if False:
            i = 10
            return i + 15
        'Returns the search query derived from terms and categories.'
        query = ' '.join(terms)
        if categories:
            query += ' category=(' + ' OR '.join(['"%s"' % category for category in categories]) + ')'
        return query

    def test_get_collection_summaries_matching_ids(self) -> None:
        if False:
            print('Hello World!')
        summaries = collection_services.get_collection_summaries_matching_ids([self.COL_ID_0, self.COL_ID_1, self.COL_ID_2, 'nonexistent'])
        assert summaries[0] is not None
        self.assertEqual(summaries[0].title, 'Bridges in England')
        assert summaries[1] is not None
        self.assertEqual(summaries[1].title, 'Introduce Oppia')
        assert summaries[2] is not None
        self.assertEqual(summaries[2].title, 'Introduce Interactions in Oppia')
        self.assertIsNone(summaries[3])

    def test_get_collection_summaries_subscribed_to(self) -> None:
        if False:
            return 10
        summaries = collection_services.get_collection_summaries_subscribed_to(self.owner_id)
        self.assertEqual(summaries[0].title, 'Bridges in England')
        self.assertEqual(summaries[1].title, 'Introduce Oppia')
        self.assertEqual(summaries[2].title, 'Introduce Interactions in Oppia')
        self.assertEqual(summaries[3].title, 'Welcome')
        self.assertEqual(summaries[4].title, 'Learning basic verbs in Spanish')
        self.assertEqual(summaries[5].title, 'Private collection in Spanish')

    def test_publish_collection_raise_exception_for_invalid_collection_id(self) -> None:
        if False:
            return 10
        system_user = user_services.get_system_user()
        with self.assertRaisesRegex(Exception, 'No collection summary model exists for the given id: Invalid_collection_id'):
            with self.swap_to_always_return(rights_manager, 'publish_collection', True):
                collection_services.publish_collection_and_update_user_profiles(system_user, 'Invalid_collection_id')

    def test_get_collection_summaries_with_no_query(self) -> None:
        if False:
            while True:
                i = 10
        (col_ids, search_cursor) = collection_services.get_collection_ids_matching_query('', [], [])
        self.assertEqual(sorted(col_ids), [self.COL_ID_0, self.COL_ID_1, self.COL_ID_2, self.COL_ID_3, self.COL_ID_4])
        self.assertIsNone(search_cursor)

    def test_get_collection_summaries_with_deleted_collections(self) -> None:
        if False:
            i = 10
            return i + 15
        collection_services.delete_collection(self.owner_id, self.COL_ID_0)
        collection_services.delete_collection(self.owner_id, self.COL_ID_2)
        collection_services.delete_collection(self.owner_id, self.COL_ID_4)
        col_ids = collection_services.get_collection_ids_matching_query('', [], [])[0]
        self.assertEqual(sorted(col_ids), [self.COL_ID_1, self.COL_ID_3])
        collection_services.delete_collection(self.owner_id, self.COL_ID_1)
        collection_services.delete_collection(self.owner_id, self.COL_ID_3)
        self.assertEqual(collection_services.get_collection_ids_matching_query('', [], []), ([], None))

    def test_get_collection_summaries_with_deleted_collections_multi(self) -> None:
        if False:
            return 10
        collection_services.delete_collections(self.owner_id, [self.COL_ID_0, self.COL_ID_2, self.COL_ID_4])
        col_ids = collection_services.get_collection_ids_matching_query('', [], [])[0]
        self.assertEqual(sorted(col_ids), [self.COL_ID_1, self.COL_ID_3])
        collection_services.delete_collections(self.owner_id, [self.COL_ID_1, self.COL_ID_3])
        self.assertEqual(collection_services.get_collection_ids_matching_query('', [], []), ([], None))

    def test_search_collection_summaries(self) -> None:
        if False:
            i = 10
            return i + 15
        col_ids = collection_services.get_collection_ids_matching_query('', ['Architecture'], [])[0]
        self.assertEqual(col_ids, [self.COL_ID_0])
        col_ids = collection_services.get_collection_ids_matching_query('Oppia', [], [])[0]
        self.assertEqual(sorted(col_ids), [self.COL_ID_1, self.COL_ID_2])
        col_ids = collection_services.get_collection_ids_matching_query('Oppia Introduce', [], [])[0]
        self.assertEqual(sorted(col_ids), [self.COL_ID_1, self.COL_ID_2])
        col_ids = collection_services.get_collection_ids_matching_query('England', [], [])[0]
        self.assertEqual(col_ids, [self.COL_ID_0])
        col_ids = collection_services.get_collection_ids_matching_query('in', [], [])[0]
        self.assertEqual(sorted(col_ids), [self.COL_ID_0, self.COL_ID_2, self.COL_ID_4])
        col_ids = collection_services.get_collection_ids_matching_query('in', ['Architecture', 'Welcome'], [])[0]
        self.assertEqual(sorted(col_ids), [self.COL_ID_0, self.COL_ID_2])

    def test_collection_summaries_pagination_in_filled_search_results(self) -> None:
        if False:
            print('Hello World!')
        with self.swap(feconf, 'SEARCH_RESULTS_PAGE_SIZE', 2):
            found_col_ids = []
            (col_ids, search_offset) = collection_services.get_collection_ids_matching_query('', [], [])
            self.assertEqual(len(col_ids), 2)
            self.assertIsNotNone(search_offset)
            found_col_ids += col_ids
            (col_ids, search_offset) = collection_services.get_collection_ids_matching_query('', [], [], offset=search_offset)
            self.assertEqual(len(col_ids), 2)
            self.assertIsNotNone(search_offset)
            found_col_ids += col_ids
            (col_ids, search_offset) = collection_services.get_collection_ids_matching_query('', [], [], offset=search_offset)
            self.assertEqual(len(col_ids), 1)
            self.assertIsNone(search_offset)
            found_col_ids += col_ids
            self.assertEqual(sorted(found_col_ids), [self.COL_ID_0, self.COL_ID_1, self.COL_ID_2, self.COL_ID_3, self.COL_ID_4])

class CollectionCreateAndDeleteUnitTests(CollectionServicesUnitTests):
    """Test creation and deletion methods."""

    def test_retrieval_of_collection(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the get_collection_by_id() method.'
        with self.assertRaisesRegex(Exception, 'Entity .* not found'):
            collection_services.get_collection_by_id('fake_eid')
        collection = self.save_new_default_collection(self.COLLECTION_0_ID, self.owner_id)
        retrieved_collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.id, retrieved_collection.id)
        self.assertEqual(collection.title, retrieved_collection.title)
        with self.assertRaisesRegex(Exception, 'Entity for class CollectionModel with id fake_collection not found'):
            collection_services.get_collection_by_id('fake_collection')

    def test_retrieval_of_multiple_collections(self) -> None:
        if False:
            return 10
        collections = {}
        chars = 'abcde'
        collection_ids = ['%s%s' % (self.COLLECTION_0_ID, c) for c in chars]
        for _id in collection_ids:
            collection = self.save_new_valid_collection(_id, self.owner_id)
            collections[_id] = collection
        result = collection_services.get_multiple_collections_by_id(collection_ids)
        for _id in collection_ids:
            self.assertEqual(result[_id].title, collections[_id].title)
        result = collection_services.get_multiple_collections_by_id(collection_ids + ['doesnt_exist'], strict=False)
        for _id in collection_ids:
            self.assertEqual(result[_id].title, collections[_id].title)
        self.assertNotIn('doesnt_exist', result)
        with self.assertRaisesRegex(Exception, "Couldn't find collections with the following ids:\ndoesnt_exist"):
            collection_services.get_multiple_collections_by_id(collection_ids + ['doesnt_exist'])

    def test_soft_deletion_of_collection(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that soft deletion of collection works correctly.'
        self.save_new_default_collection(self.COLLECTION_0_ID, self.owner_id)
        self.assertEqual(count_at_least_editable_collection_summaries(self.owner_id), 1)
        collection_services.delete_collection(self.owner_id, self.COLLECTION_0_ID)
        with self.assertRaisesRegex(Exception, 'Entity for class CollectionModel with id A_collection_0_id not found'):
            collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(count_at_least_editable_collection_summaries(self.owner_id), 0)
        self.assertIsNotNone(collection_models.CollectionModel.get_by_id(self.COLLECTION_0_ID))
        self.assertIsNone(collection_models.CollectionSummaryModel.get_by_id(self.COLLECTION_0_ID))
        self.assertIsNotNone(collection_models.CollectionCommitLogEntryModel.get_by_id('collection-%s-%s' % (self.COLLECTION_0_ID, 1)))
        collection_snapshot_id = collection_models.CollectionModel.get_snapshot_id(self.COLLECTION_0_ID, 1)
        self.assertIsNotNone(collection_models.CollectionSnapshotMetadataModel.get_by_id(collection_snapshot_id))
        self.assertIsNotNone(collection_models.CollectionSnapshotContentModel.get_by_id(collection_snapshot_id))
        collection_rights_snapshot_id = collection_models.CollectionRightsModel.get_snapshot_id(self.COLLECTION_0_ID, 1)
        self.assertIsNotNone(collection_models.CollectionRightsSnapshotMetadataModel.get_by_id(collection_rights_snapshot_id))
        self.assertIsNotNone(collection_models.CollectionRightsSnapshotContentModel.get_by_id(collection_rights_snapshot_id))

    def test_deletion_of_multiple_collections_empty(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that delete_collections with empty list works correctly.'
        collection_services.delete_collections(self.owner_id, [])

    def test_soft_deletion_of_multiple_collections(self) -> None:
        if False:
            while True:
                i = 10
        'Test that soft deletion of multiple collections works correctly.'
        self.save_new_default_collection(self.COLLECTION_0_ID, self.owner_id)
        self.save_new_default_collection(self.COLLECTION_1_ID, self.owner_id)
        self.assertEqual(count_at_least_editable_collection_summaries(self.owner_id), 2)
        collection_services.delete_collections(self.owner_id, [self.COLLECTION_0_ID, self.COLLECTION_1_ID])
        with self.assertRaisesRegex(Exception, 'Entity for class CollectionModel with id A_collection_0_id not found'):
            collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        with self.assertRaisesRegex(Exception, 'Entity for class CollectionModel with id A_collection_1_id not found'):
            collection_services.get_collection_by_id(self.COLLECTION_1_ID)
        self.assertEqual(count_at_least_editable_collection_summaries(self.owner_id), 0)
        self.assertIsNotNone(collection_models.CollectionModel.get_by_id(self.COLLECTION_0_ID))
        self.assertIsNotNone(collection_models.CollectionModel.get_by_id(self.COLLECTION_1_ID))
        self.assertIsNone(collection_models.CollectionSummaryModel.get_by_id(self.COLLECTION_0_ID))
        self.assertIsNone(collection_models.CollectionSummaryModel.get_by_id(self.COLLECTION_1_ID))
        self.assertIsNotNone(collection_models.CollectionCommitLogEntryModel.get_by_id('collection-%s-%s' % (self.COLLECTION_0_ID, 1)))
        self.assertIsNotNone(collection_models.CollectionCommitLogEntryModel.get_by_id('collection-%s-%s' % (self.COLLECTION_1_ID, 1)))
        collection_0_snapshot_id = collection_models.CollectionModel.get_snapshot_id(self.COLLECTION_0_ID, 1)
        collection_1_snapshot_id = collection_models.CollectionModel.get_snapshot_id(self.COLLECTION_1_ID, 1)
        self.assertIsNotNone(collection_models.CollectionSnapshotMetadataModel.get_by_id(collection_0_snapshot_id))
        self.assertIsNotNone(collection_models.CollectionSnapshotContentModel.get_by_id(collection_0_snapshot_id))
        self.assertIsNotNone(collection_models.CollectionSnapshotMetadataModel.get_by_id(collection_1_snapshot_id))
        self.assertIsNotNone(collection_models.CollectionSnapshotContentModel.get_by_id(collection_1_snapshot_id))
        collection_0_rights_snapshot_id = collection_models.CollectionRightsModel.get_snapshot_id(self.COLLECTION_0_ID, 1)
        collection_1_rights_snapshot_id = collection_models.CollectionRightsModel.get_snapshot_id(self.COLLECTION_1_ID, 1)
        self.assertIsNotNone(collection_models.CollectionRightsSnapshotMetadataModel.get_by_id(collection_0_rights_snapshot_id))
        self.assertIsNotNone(collection_models.CollectionRightsSnapshotContentModel.get_by_id(collection_0_rights_snapshot_id))
        self.assertIsNotNone(collection_models.CollectionRightsSnapshotMetadataModel.get_by_id(collection_1_rights_snapshot_id))
        self.assertIsNotNone(collection_models.CollectionRightsSnapshotContentModel.get_by_id(collection_1_rights_snapshot_id))

    def test_hard_deletion_of_collection(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that hard deletion of collection works correctly.'
        self.save_new_default_collection(self.COLLECTION_0_ID, self.owner_id)
        self.assertEqual(count_at_least_editable_collection_summaries(self.owner_id), 1)
        collection_services.delete_collection(self.owner_id, self.COLLECTION_0_ID, force_deletion=True)
        with self.assertRaisesRegex(Exception, 'Entity for class CollectionModel with id A_collection_0_id not found'):
            collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(count_at_least_editable_collection_summaries(self.owner_id), 0)
        self.assertIsNone(collection_models.CollectionModel.get_by_id(self.COLLECTION_1_ID))

    def test_hard_deletion_of_multiple_collections(self) -> None:
        if False:
            print('Hello World!')
        'Test that hard deletion of multiple collections works correctly.'
        self.save_new_default_collection(self.COLLECTION_0_ID, self.owner_id)
        self.save_new_default_collection(self.COLLECTION_1_ID, self.owner_id)
        self.assertEqual(count_at_least_editable_collection_summaries(self.owner_id), 2)
        collection_services.delete_collections(self.owner_id, [self.COLLECTION_0_ID, self.COLLECTION_1_ID], force_deletion=True)
        with self.assertRaisesRegex(Exception, 'Entity for class CollectionModel with id A_collection_0_id not found'):
            collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        with self.assertRaisesRegex(Exception, 'Entity for class CollectionModel with id A_collection_1_id not found'):
            collection_services.get_collection_by_id(self.COLLECTION_1_ID)
        self.assertEqual(count_at_least_editable_collection_summaries(self.owner_id), 0)
        self.assertIsNone(collection_models.CollectionModel.get_by_id(self.COLLECTION_0_ID))
        self.assertIsNone(collection_models.CollectionModel.get_by_id(self.COLLECTION_1_ID))
        self.assertIsNone(collection_models.CollectionSummaryModel.get_by_id(self.COLLECTION_0_ID))
        self.assertIsNone(collection_models.CollectionSummaryModel.get_by_id(self.COLLECTION_1_ID))

    def test_summaries_of_hard_deleted_collections(self) -> None:
        if False:
            return 10
        'Test that summaries of hard deleted collections are\n        correctly deleted.\n        '
        self.save_new_default_collection(self.COLLECTION_0_ID, self.owner_id)
        collection_services.delete_collection(self.owner_id, self.COLLECTION_0_ID, force_deletion=True)
        with self.assertRaisesRegex(Exception, 'Entity for class CollectionModel with id A_collection_0_id not found'):
            collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(count_at_least_editable_collection_summaries(self.owner_id), 0)
        self.assertNotIn(self.COLLECTION_0_ID, [collection.id for collection in collection_models.CollectionSummaryModel.get_all(include_deleted=True)])

    def test_collection_is_removed_from_index_when_deleted(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that deleted collection is removed from the search index.'
        self.save_new_default_collection(self.COLLECTION_0_ID, self.owner_id)

        def mock_delete_docs(doc_ids: List[str], index: str) -> None:
            if False:
                while True:
                    i = 10
            self.assertEqual(index, collection_services.SEARCH_INDEX_COLLECTIONS)
            self.assertEqual(doc_ids, [self.COLLECTION_0_ID])
        delete_docs_swap = self.swap(gae_search_services, 'delete_documents_from_index', mock_delete_docs)
        with delete_docs_swap:
            collection_services.delete_collection(self.owner_id, self.COLLECTION_0_ID)

    def test_collections_are_removed_from_index_when_deleted(self) -> None:
        if False:
            return 10
        'Tests that deleted collections are removed from the search index.'
        self.save_new_default_collection(self.COLLECTION_0_ID, self.owner_id)
        self.save_new_default_collection(self.COLLECTION_1_ID, self.owner_id)

        def mock_delete_docs(doc_ids: List[str], index: str) -> None:
            if False:
                print('Hello World!')
            self.assertEqual(index, collection_services.SEARCH_INDEX_COLLECTIONS)
            self.assertEqual(doc_ids, [self.COLLECTION_0_ID, self.COLLECTION_1_ID])
        delete_docs_swap = self.swap(gae_search_services, 'delete_documents_from_index', mock_delete_docs)
        with delete_docs_swap:
            collection_services.delete_collections(self.owner_id, [self.COLLECTION_0_ID, self.COLLECTION_1_ID])

    def test_create_new_collection(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        collection_domain.Collection.create_default_collection(self.COLLECTION_0_ID)

    def test_save_and_retrieve_collection(self) -> None:
        if False:
            while True:
                i = 10
        collection = self.save_new_valid_collection(self.COLLECTION_0_ID, self.owner_id)
        collection_services._save_collection(self.owner_id, collection, '', _get_collection_change_list('title', ''))
        retrieved_collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(retrieved_collection.title, 'A title')
        self.assertEqual(retrieved_collection.category, 'A category')
        self.assertEqual(len(retrieved_collection.nodes), 1)

    def test_save_and_retrieve_collection_summary(self) -> None:
        if False:
            i = 10
            return i + 15
        collection = self.save_new_valid_collection(self.COLLECTION_0_ID, self.owner_id)
        collection_services._save_collection(self.owner_id, collection, '', _get_collection_change_list('title', ''))
        collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, [{'cmd': 'edit_collection_property', 'property_name': 'title', 'new_value': 'A new title'}, {'cmd': 'edit_collection_property', 'property_name': 'category', 'new_value': 'A new category'}], 'Change title and category')
        retrieved_collection_summary = collection_services.get_collection_summary_by_id(self.COLLECTION_0_ID)
        assert retrieved_collection_summary is not None
        self.assertEqual(retrieved_collection_summary.contributor_ids, [self.owner_id])
        self.assertEqual(retrieved_collection_summary.title, 'A new title')
        self.assertEqual(retrieved_collection_summary.category, 'A new category')

    def test_update_collection_by_migration_bot(self) -> None:
        if False:
            while True:
                i = 10
        exp_id = 'exp_id'
        self.save_new_valid_collection(self.COLLECTION_0_ID, self.owner_id, exploration_id=exp_id)
        rights_manager.publish_exploration(self.owner, exp_id)
        rights_manager.publish_collection(self.owner, self.COLLECTION_0_ID)
        collection_services.update_collection(feconf.MIGRATION_BOT_USER_ID, self.COLLECTION_0_ID, [{'cmd': 'edit_collection_property', 'property_name': 'title', 'new_value': 'New title'}], 'Did migration.')
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.version, 2)

    def test_update_collection_schema(self) -> None:
        if False:
            i = 10
            return i + 15
        exp_id = 'exp_id'
        self.save_new_valid_collection(self.COLLECTION_0_ID, self.owner_id, exploration_id=exp_id)
        rights_manager.publish_exploration(self.owner, exp_id)
        rights_manager.publish_collection(self.owner, self.COLLECTION_0_ID)
        collection_services.update_collection(feconf.MIGRATION_BOT_USER_ID, self.COLLECTION_0_ID, [{'cmd': collection_domain.CMD_MIGRATE_SCHEMA_TO_LATEST_VERSION, 'from_version': 2, 'to_version': 3}], 'Did migration.')
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.version, 2)

class LoadingAndDeletionOfCollectionDemosTests(CollectionServicesUnitTests):

    def test_loading_and_validation_and_deletion_of_demo_collections(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test loading, validation and deletion of the demo collections.'
        self.assertEqual(collection_models.CollectionModel.get_collection_count(), 0)
        self.assertGreaterEqual(len(feconf.DEMO_COLLECTIONS), 1, msg='There must be at least one demo collection.')
        for collection_id in feconf.DEMO_COLLECTIONS:
            start_time = datetime.datetime.utcnow()
            collection_services.load_demo(collection_id)
            collection = collection_services.get_collection_by_id(collection_id)
            collection.validate()
            duration = datetime.datetime.utcnow() - start_time
            processing_time = duration.seconds + duration.microseconds / 1000000.0
            self.log_line('Loaded and validated collection %s (%.2f seconds)' % (collection.title, processing_time))
        self.assertEqual(collection_models.CollectionModel.get_collection_count(), len(feconf.DEMO_COLLECTIONS))
        for collection_id in feconf.DEMO_COLLECTIONS:
            collection_services.delete_demo(collection_id)
        self.assertEqual(collection_models.CollectionModel.get_collection_count(), 0)

    def test_load_demo_with_invalid_collection_id_raises_error(self) -> None:
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(Exception, 'Invalid demo collection id'):
            collection_services.load_demo('invalid_collection_id')

    def test_demo_file_path_ends_with_yaml(self) -> None:
        if False:
            while True:
                i = 10
        for collection_path in feconf.DEMO_COLLECTIONS.values():
            demo_filepath = os.path.join(feconf.SAMPLE_COLLECTIONS_DIR, collection_path)
            self.assertTrue(demo_filepath.endswith('yaml'))

class UpdateCollectionNodeTests(CollectionServicesUnitTests):
    """Test updating a single collection node."""
    EXPLORATION_ID: Final = 'exp_id_0'
    COLLECTION_TITLE: Final = 'title'
    COLLECTION_CATEGORY: Final = 'category'
    COLLECTION_OBJECTIVE: Final = 'objective'

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.save_new_valid_collection(self.COLLECTION_0_ID, self.owner_id, title=self.COLLECTION_TITLE, category=self.COLLECTION_CATEGORY, objective=self.COLLECTION_OBJECTIVE, exploration_id=self.EXPLORATION_ID)

    def test_add_node(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.exploration_ids, [self.EXPLORATION_ID])
        new_exp_id = 'new_exploration_id'
        self.save_new_valid_exploration(new_exp_id, self.owner_id)
        collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, [{'cmd': collection_domain.CMD_ADD_COLLECTION_NODE, 'exploration_id': new_exp_id}], 'Added new exploration')
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.exploration_ids, [self.EXPLORATION_ID, new_exp_id])

    def test_add_node_with_non_existent_exploration(self) -> None:
        if False:
            print('Hello World!')
        non_existent_exp_id = 'non_existent_exploration_id'
        with self.assertRaisesRegex(utils.ValidationError, 'Expected collection to only reference valid explorations'):
            collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, [{'cmd': collection_domain.CMD_ADD_COLLECTION_NODE, 'exploration_id': non_existent_exp_id}], 'Added non-existent exploration')

    def test_add_node_with_private_exploration_in_public_collection(self) -> None:
        if False:
            return 10
        'Ensures public collections cannot reference private explorations.'
        private_exp_id = 'private_exp_id0'
        self.save_new_valid_exploration(private_exp_id, self.owner_id)
        rights_manager.publish_collection(self.owner, self.COLLECTION_0_ID)
        self.assertTrue(rights_manager.is_collection_public(self.COLLECTION_0_ID))
        self.assertTrue(rights_manager.is_exploration_private(private_exp_id))
        with self.assertRaisesRegex(utils.ValidationError, 'Cannot reference a private exploration within a public collection'):
            collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, [{'cmd': collection_domain.CMD_ADD_COLLECTION_NODE, 'exploration_id': private_exp_id}], 'Added private exploration')

    def test_add_node_with_public_exploration_in_private_collection(self) -> None:
        if False:
            while True:
                i = 10
        'Ensures private collections can reference public and private\n        explorations.\n        '
        public_exp_id = 'public_exp_id0'
        private_exp_id = 'private_exp_id0'
        self.save_new_valid_exploration(public_exp_id, self.owner_id)
        self.save_new_valid_exploration(private_exp_id, self.owner_id)
        rights_manager.publish_exploration(self.owner, public_exp_id)
        self.assertTrue(rights_manager.is_collection_private(self.COLLECTION_0_ID))
        self.assertTrue(rights_manager.is_exploration_public(public_exp_id))
        self.assertTrue(rights_manager.is_exploration_private(private_exp_id))
        collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, [{'cmd': collection_domain.CMD_ADD_COLLECTION_NODE, 'exploration_id': public_exp_id}, {'cmd': collection_domain.CMD_ADD_COLLECTION_NODE, 'exploration_id': private_exp_id}], 'Added public and private explorations')

    def test_delete_node(self) -> None:
        if False:
            while True:
                i = 10
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.exploration_ids, [self.EXPLORATION_ID])
        collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, [{'cmd': collection_domain.CMD_DELETE_COLLECTION_NODE, 'exploration_id': self.EXPLORATION_ID}], 'Deleted exploration')
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.nodes, [])

    def test_update_collection_title(self) -> None:
        if False:
            while True:
                i = 10
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.title, self.COLLECTION_TITLE)
        collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, [{'cmd': collection_domain.CMD_EDIT_COLLECTION_PROPERTY, 'property_name': 'title', 'new_value': 'Some new title'}], 'Changed the title')
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.title, 'Some new title')

    def test_update_collection_category(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.category, self.COLLECTION_CATEGORY)
        collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, [{'cmd': collection_domain.CMD_EDIT_COLLECTION_PROPERTY, 'property_name': 'category', 'new_value': 'Some new category'}], 'Changed the category')
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.category, 'Some new category')

    def test_update_collection_objective(self) -> None:
        if False:
            i = 10
            return i + 15
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.objective, self.COLLECTION_OBJECTIVE)
        collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, [{'cmd': collection_domain.CMD_EDIT_COLLECTION_PROPERTY, 'property_name': 'objective', 'new_value': 'Some new objective'}], 'Changed the objective')
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.objective, 'Some new objective')

    def test_update_collection_language_code(self) -> None:
        if False:
            i = 10
            return i + 15
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.language_code, 'en')
        collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, [{'cmd': collection_domain.CMD_EDIT_COLLECTION_PROPERTY, 'property_name': 'language_code', 'new_value': 'fi'}], 'Changed the language to Finnish')
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.language_code, 'fi')

    def test_update_collection_tags(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.tags, [])
        collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, [{'cmd': collection_domain.CMD_EDIT_COLLECTION_PROPERTY, 'property_name': 'tags', 'new_value': ['test']}], 'Add a new tag')
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(collection.tags, ['test'])
        with self.assertRaisesRegex(utils.ValidationError, 'Expected tags to be unique, but found duplicates'):
            collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, [{'cmd': collection_domain.CMD_EDIT_COLLECTION_PROPERTY, 'property_name': 'tags', 'new_value': ['duplicate', 'duplicate']}], 'Add a new tag')

def _get_collection_change_list(property_name: str, new_value: str) -> List[Dict[str, str]]:
    if False:
        i = 10
        return i + 15
    'Generates a change list for a single collection property change.'
    return [{'cmd': collection_domain.CMD_EDIT_COLLECTION_PROPERTY, 'property_name': property_name, 'new_value': new_value}]

def _get_added_exploration_change_list(exploration_id: str) -> List[Dict[str, str]]:
    if False:
        print('Hello World!')
    'Generates a change list for adding an exploration to a collection.'
    return [{'cmd': collection_domain.CMD_ADD_COLLECTION_NODE, 'exploration_id': exploration_id}]

def _get_deleted_exploration_change_list(exploration_id: str) -> List[Dict[str, str]]:
    if False:
        for i in range(10):
            print('nop')
    'Generates a change list for deleting an exploration from a collection.'
    return [{'cmd': collection_domain.CMD_DELETE_COLLECTION_NODE, 'exploration_id': exploration_id}]

class CommitMessageHandlingTests(CollectionServicesUnitTests):
    """Test the handling of commit messages."""
    EXP_ID: Final = 'an_exploration_id'

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.save_new_valid_collection(self.COLLECTION_0_ID, self.owner_id, exploration_id=self.EXP_ID)

    def test_record_commit_message(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Check published collections record commit messages.'
        rights_manager.publish_collection(self.owner, self.COLLECTION_0_ID)
        rights_manager.publish_exploration(self.owner, self.EXP_ID)
        collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, _get_collection_change_list(collection_domain.COLLECTION_PROPERTY_TITLE, 'New Title'), 'A message')
        self.assertEqual(collection_services.get_collection_snapshots_metadata(self.COLLECTION_0_ID)[1]['commit_message'], 'A message')

    def test_demand_commit_message(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Check published collections demand commit messages.'
        rights_manager.publish_collection(self.owner, self.COLLECTION_0_ID)
        with self.assertRaisesRegex(ValueError, 'Collection is public so expected a commit message but received none.'):
            collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, _get_collection_change_list(collection_domain.COLLECTION_PROPERTY_TITLE, 'New Title'), '')

    def test_unpublished_collections_can_accept_commit_message(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test unpublished collections can accept optional commit messages.'
        collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, _get_collection_change_list(collection_domain.COLLECTION_PROPERTY_TITLE, 'New Title'), 'A message')
        collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, _get_collection_change_list(collection_domain.COLLECTION_PROPERTY_TITLE, 'New Title'), '')
        collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, _get_collection_change_list(collection_domain.COLLECTION_PROPERTY_TITLE, 'New Title'), None)

class CollectionSnapshotUnitTests(CollectionServicesUnitTests):
    """Test methods relating to collection snapshots."""
    SECOND_USERNAME: Final = 'abc123'
    SECOND_EMAIL: Final = 'abc123@gmail.com'

    def test_get_collection_snapshots_metadata(self) -> None:
        if False:
            print('Hello World!')
        self.signup(self.SECOND_EMAIL, self.SECOND_USERNAME)
        second_committer_id = self.get_user_id_from_email(self.SECOND_EMAIL)
        exp_id = 'exp_id0'
        v1_collection = self.save_new_valid_collection(self.COLLECTION_0_ID, self.owner_id, exploration_id=exp_id)
        snapshots_metadata = collection_services.get_collection_snapshots_metadata(self.COLLECTION_0_ID)
        self.assertEqual(len(snapshots_metadata), 1)
        self.assertDictContainsSubset({'commit_cmds': [{'cmd': 'create_new', 'title': 'A title', 'category': 'A category'}], 'committer_id': self.owner_id, 'commit_message': "New collection created with title 'A title'.", 'commit_type': 'create', 'version_number': 1}, snapshots_metadata[0])
        self.assertIn('created_on_ms', snapshots_metadata[0])
        rights_manager.publish_collection(self.owner, self.COLLECTION_0_ID)
        rights_manager.publish_exploration(self.owner, exp_id)
        snapshots_metadata = collection_services.get_collection_snapshots_metadata(self.COLLECTION_0_ID)
        self.assertEqual(len(snapshots_metadata), 1)
        self.assertDictContainsSubset({'commit_cmds': [{'cmd': 'create_new', 'title': 'A title', 'category': 'A category'}], 'committer_id': self.owner_id, 'commit_message': "New collection created with title 'A title'.", 'commit_type': 'create', 'version_number': 1}, snapshots_metadata[0])
        self.assertIn('created_on_ms', snapshots_metadata[0])
        change_list = [{'cmd': 'edit_collection_property', 'property_name': 'title', 'new_value': 'First title'}]
        collection_services.update_collection(self.owner_id, self.COLLECTION_0_ID, change_list, 'Changed title.')
        snapshots_metadata = collection_services.get_collection_snapshots_metadata(self.COLLECTION_0_ID)
        self.assertEqual(len(snapshots_metadata), 2)
        self.assertIn('created_on_ms', snapshots_metadata[0])
        self.assertDictContainsSubset({'commit_cmds': [{'cmd': 'create_new', 'title': 'A title', 'category': 'A category'}], 'committer_id': self.owner_id, 'commit_message': "New collection created with title 'A title'.", 'commit_type': 'create', 'version_number': 1}, snapshots_metadata[0])
        self.assertDictContainsSubset({'commit_cmds': change_list, 'committer_id': self.owner_id, 'commit_message': 'Changed title.', 'commit_type': 'edit', 'version_number': 2}, snapshots_metadata[1])
        self.assertLess(snapshots_metadata[0]['created_on_ms'], snapshots_metadata[1]['created_on_ms'])
        with self.assertRaisesRegex(Exception, 'version 1, which is too old'):
            collection_services._save_collection(second_committer_id, v1_collection, '', _get_collection_change_list('title', ''))
        new_change_list = [{'cmd': 'edit_collection_property', 'property_name': 'title', 'new_value': 'New title'}]
        collection_services.update_collection(second_committer_id, self.COLLECTION_0_ID, new_change_list, 'Second commit.')
        snapshots_metadata = collection_services.get_collection_snapshots_metadata(self.COLLECTION_0_ID)
        self.assertEqual(len(snapshots_metadata), 3)
        self.assertDictContainsSubset({'commit_cmds': [{'cmd': 'create_new', 'title': 'A title', 'category': 'A category'}], 'committer_id': self.owner_id, 'commit_message': "New collection created with title 'A title'.", 'commit_type': 'create', 'version_number': 1}, snapshots_metadata[0])
        self.assertDictContainsSubset({'commit_cmds': change_list, 'committer_id': self.owner_id, 'commit_message': 'Changed title.', 'commit_type': 'edit', 'version_number': 2}, snapshots_metadata[1])
        self.assertDictContainsSubset({'commit_cmds': new_change_list, 'committer_id': second_committer_id, 'commit_message': 'Second commit.', 'commit_type': 'edit', 'version_number': 3}, snapshots_metadata[2])
        self.assertLess(snapshots_metadata[1]['created_on_ms'], snapshots_metadata[2]['created_on_ms'])

    def test_versioning_with_add_and_delete_nodes(self) -> None:
        if False:
            return 10
        collection = self.save_new_valid_collection(self.COLLECTION_0_ID, self.owner_id)
        collection.title = 'First title'
        collection_services._save_collection(self.owner_id, collection, 'Changed title.', _get_collection_change_list('title', 'First title'))
        commit_dict_2 = {'committer_id': self.owner_id, 'commit_message': 'Changed title.', 'version_number': 2}
        snapshots_metadata = collection_services.get_collection_snapshots_metadata(self.COLLECTION_0_ID)
        self.assertEqual(len(snapshots_metadata), 2)
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        collection.add_node(self.save_new_valid_exploration('new_exploration_id', self.owner_id).id)
        collection_services._save_collection('committer_id_2', collection, 'Added new exploration', _get_added_exploration_change_list('new_exploration_id'))
        commit_dict_3 = {'committer_id': 'committer_id_2', 'commit_message': 'Added new exploration', 'version_number': 3}
        snapshots_metadata = collection_services.get_collection_snapshots_metadata(self.COLLECTION_0_ID)
        self.assertEqual(len(snapshots_metadata), 3)
        self.assertDictContainsSubset(commit_dict_3, snapshots_metadata[2])
        self.assertDictContainsSubset(commit_dict_2, snapshots_metadata[1])
        for ind in range(len(snapshots_metadata) - 1):
            self.assertLess(snapshots_metadata[ind]['created_on_ms'], snapshots_metadata[ind + 1]['created_on_ms'])
        with self.assertRaisesRegex(ValueError, 'is not part of this collection'):
            collection.delete_node('invalid_exploration_id')
        collection.delete_node('new_exploration_id')
        collection_services._save_collection('committer_id_3', collection, 'Deleted exploration', _get_deleted_exploration_change_list('new_exploration_id'))
        commit_dict_4 = {'committer_id': 'committer_id_3', 'commit_message': 'Deleted exploration', 'version_number': 4}
        snapshots_metadata = collection_services.get_collection_snapshots_metadata(self.COLLECTION_0_ID)
        self.assertEqual(len(snapshots_metadata), 4)
        self.assertDictContainsSubset(commit_dict_4, snapshots_metadata[3])
        self.assertDictContainsSubset(commit_dict_3, snapshots_metadata[2])
        self.assertDictContainsSubset(commit_dict_2, snapshots_metadata[1])
        for ind in range(len(snapshots_metadata) - 1):
            self.assertLess(snapshots_metadata[ind]['created_on_ms'], snapshots_metadata[ind + 1]['created_on_ms'])
        collection = collection_services.get_collection_by_id(self.COLLECTION_0_ID)
        self.assertEqual(len(collection.nodes), 1)

class CollectionSearchTests(CollectionServicesUnitTests):
    """Test collection search."""

    def test_index_collections_given_ids(self) -> None:
        if False:
            while True:
                i = 10
        all_collection_ids = ['id0', 'id1', 'id2', 'id3', 'id4']
        expected_collection_ids = all_collection_ids[:-1]
        all_collection_titles = ['title 0', 'title 1', 'title 2', 'title 3', 'title 4']
        expected_collection_titles = all_collection_titles[:-1]
        all_collection_categories = ['cat0', 'cat1', 'cat2', 'cat3', 'cat4']
        expected_collection_categories = all_collection_categories[:-1]

        def mock_add_documents_to_index(docs: List[Dict[str, str]], index: str) -> List[str]:
            if False:
                while True:
                    i = 10
            self.assertEqual(index, collection_services.SEARCH_INDEX_COLLECTIONS)
            ids = [doc['id'] for doc in docs]
            titles = [doc['title'] for doc in docs]
            categories = [doc['category'] for doc in docs]
            self.assertEqual(set(ids), set(expected_collection_ids))
            self.assertEqual(set(titles), set(expected_collection_titles))
            self.assertEqual(set(categories), set(expected_collection_categories))
            return ids
        add_docs_counter = test_utils.CallCounter(mock_add_documents_to_index)
        add_docs_swap = self.swap(gae_search_services, 'add_documents_to_index', add_docs_counter)
        for ind in range(5):
            self.save_new_valid_collection(all_collection_ids[ind], self.owner_id, title=all_collection_titles[ind], category=all_collection_categories[ind])
        for ind in range(4):
            rights_manager.publish_collection(self.owner, expected_collection_ids[ind])
        with add_docs_swap:
            collection_services.index_collections_given_ids(all_collection_ids)
        self.assertEqual(add_docs_counter.times_called, 1)

class CollectionSummaryTests(CollectionServicesUnitTests):
    """Test collection summaries."""
    ALBERT_EMAIL: Final = 'albert@example.com'
    BOB_EMAIL: Final = 'bob@example.com'
    ALBERT_NAME: Final = 'albert'
    BOB_NAME: Final = 'bob'
    COLLECTION_ID_1: Final = 'cid1'
    COLLECTION_ID_2: Final = 'cid2'

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.signup(self.ALBERT_EMAIL, self.ALBERT_NAME)
        self.signup(self.BOB_EMAIL, self.BOB_NAME)
        self.albert_id = self.get_user_id_from_email(self.ALBERT_EMAIL)
        self.bob_id = self.get_user_id_from_email(self.BOB_EMAIL)

    def test_is_editable_by(self) -> None:
        if False:
            return 10
        self.save_new_default_collection(self.COLLECTION_0_ID, self.owner_id)
        collection_summary = collection_services.get_collection_summary_by_id(self.COLLECTION_0_ID)
        assert collection_summary is not None
        self.assertTrue(collection_summary.is_editable_by(user_id=self.owner_id))
        self.assertFalse(collection_summary.is_editable_by(user_id=self.editor_id))
        self.assertFalse(collection_summary.is_editable_by(user_id=self.viewer_id))
        rights_manager.assign_role_for_collection(self.owner, self.COLLECTION_0_ID, self.viewer_id, rights_domain.ROLE_VIEWER)
        rights_manager.assign_role_for_collection(self.owner, self.COLLECTION_0_ID, self.editor_id, rights_domain.ROLE_EDITOR)
        collection_summary = collection_services.get_collection_summary_by_id(self.COLLECTION_0_ID)
        assert collection_summary is not None
        self.assertTrue(collection_summary.is_editable_by(user_id=self.owner_id))
        self.assertTrue(collection_summary.is_editable_by(user_id=self.editor_id))
        self.assertFalse(collection_summary.is_editable_by(user_id=self.viewer_id))

    def test_contributor_ids(self) -> None:
        if False:
            return 10
        albert = user_services.get_user_actions_info(self.albert_id)
        self.save_new_valid_collection(self.COLLECTION_0_ID, self.albert_id)
        changelist_cmds = [{'cmd': collection_domain.CMD_EDIT_COLLECTION_PROPERTY, 'property_name': 'title', 'new_value': 'Collection Bob title'}]
        collection_services.update_collection(self.bob_id, self.COLLECTION_0_ID, changelist_cmds, 'Changed title to Bob title.')
        rights_manager.assign_role_for_collection(albert, self.COLLECTION_0_ID, self.viewer_id, rights_domain.ROLE_VIEWER)
        rights_manager.assign_role_for_collection(albert, self.COLLECTION_0_ID, self.editor_id, rights_domain.ROLE_EDITOR)
        collection_summary = collection_services.get_collection_summary_by_id(self.COLLECTION_0_ID)
        assert collection_summary is not None
        self.assertItemsEqual(collection_summary.contributor_ids, [self.albert_id, self.bob_id])

    def _check_contributors_summary(self, collection_id: str, expected: Dict[str, int]) -> None:
        if False:
            print('Hello World!')
        'Checks the contributors summary with the expected summary.'
        contributors_summary = collection_services.get_collection_summary_by_id(collection_id)
        assert contributors_summary is not None
        self.assertEqual(expected, contributors_summary.contributors_summary)

    def test_contributor_summary(self) -> None:
        if False:
            while True:
                i = 10
        self.save_new_valid_collection(self.COLLECTION_0_ID, self.albert_id)
        self._check_contributors_summary(self.COLLECTION_0_ID, {self.albert_id: 1})
        changelist_cmds = [{'cmd': collection_domain.CMD_EDIT_COLLECTION_PROPERTY, 'property_name': 'title', 'new_value': 'Collection Bob title'}]
        collection_services.update_collection(self.bob_id, self.COLLECTION_0_ID, changelist_cmds, 'Changed title.')
        self._check_contributors_summary(self.COLLECTION_0_ID, {self.albert_id: 1, self.bob_id: 1})
        collection_services.update_collection(self.bob_id, self.COLLECTION_0_ID, changelist_cmds, 'Changed title.')
        self._check_contributors_summary(self.COLLECTION_0_ID, {self.albert_id: 1, self.bob_id: 2})
        collection_services.update_collection(self.albert_id, self.COLLECTION_0_ID, changelist_cmds, 'Changed title.')
        self._check_contributors_summary(self.COLLECTION_0_ID, {self.albert_id: 2, self.bob_id: 2})

    def test_create_collection_summary_with_contributor_to_remove(self) -> None:
        if False:
            return 10
        self.save_new_valid_collection(self.COLLECTION_0_ID, self.albert_id)
        collection_services.update_collection(self.bob_id, self.COLLECTION_0_ID, [{'cmd': collection_domain.CMD_EDIT_COLLECTION_PROPERTY, 'property_name': 'title', 'new_value': 'Collection Bob title'}], 'Changed title.')
        collection_services.regenerate_collection_and_contributors_summaries(self.COLLECTION_0_ID)
        self._check_contributors_summary(self.COLLECTION_0_ID, {self.albert_id: 1, self.bob_id: 1})
        user_services.mark_user_for_deletion(self.bob_id)
        collection_services.regenerate_collection_and_contributors_summaries(self.COLLECTION_0_ID)
        self._check_contributors_summary(self.COLLECTION_0_ID, {self.albert_id: 1})

    def test_raises_error_when_collection_provided_with_no_last_updated_data(self) -> None:
        if False:
            return 10
        self.save_new_valid_collection('test_id', self.albert_id)
        collection = collection_services.get_collection_by_id('test_id')
        collection.last_updated = None
        with self.swap_to_always_return(collection_services, 'get_collection_by_id', collection):
            with self.assertRaisesRegex(Exception, 'No data available for when the collection was last_updated.'):
                collection_services.regenerate_collection_and_contributors_summaries('test_id')

    def test_raises_error_when_collection_provided_with_no_created_on_data(self) -> None:
        if False:
            i = 10
            return i + 15
        self.save_new_valid_collection('test_id', self.albert_id)
        collection = collection_services.get_collection_by_id('test_id')
        collection.created_on = None
        with self.swap_to_always_return(collection_services, 'get_collection_by_id', collection):
            with self.assertRaisesRegex(Exception, 'No data available for when the collection was created.'):
                collection_services.regenerate_collection_and_contributors_summaries('test_id')

class GetCollectionAndCollectionRightsTests(CollectionServicesUnitTests):

    def test_get_collection_and_collection_rights_object(self) -> None:
        if False:
            print('Hello World!')
        collection_id = self.COLLECTION_0_ID
        self.save_new_valid_collection(collection_id, self.owner_id, objective='The objective')
        (collection, collection_rights) = collection_services.get_collection_and_collection_rights_by_id(collection_id)
        assert collection_rights is not None
        assert collection is not None
        self.assertEqual(collection.id, collection_id)
        self.assertEqual(collection_rights.id, collection_id)
        (collection, collection_rights) = collection_services.get_collection_and_collection_rights_by_id('fake_id')
        self.assertIsNone(collection)
        self.assertIsNone(collection_rights)