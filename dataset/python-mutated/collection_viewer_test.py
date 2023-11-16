"""Tests for the page that allows learners to play through a collection."""
from __future__ import annotations
from core import feconf
from core.domain import collection_services
from core.domain import rights_manager
from core.domain import user_services
from core.tests import test_utils
from typing import Final

class CollectionViewerPermissionsTests(test_utils.GenericTestBase):
    """Test permissions for learners to view collections."""
    COLLECTION_ID: Final = 'cid'
    OTHER_EDITOR_EMAIL: Final = 'another@example.com'

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        'Before each individual test, create a dummy collection.'
        super().setUp()
        self.signup(self.EDITOR_EMAIL, self.EDITOR_USERNAME)
        self.editor_id = self.get_user_id_from_email(self.EDITOR_EMAIL)
        self.editor = user_services.get_user_actions_info(self.editor_id)
        self.signup(self.NEW_USER_EMAIL, self.NEW_USER_USERNAME)
        self.new_user_id = self.get_user_id_from_email(self.NEW_USER_EMAIL)
        self.save_new_valid_collection(self.COLLECTION_ID, self.editor_id)

    def test_unpublished_collections_are_invisible_to_logged_out_users(self) -> None:
        if False:
            print('Hello World!')
        self.get_html_response('%s/%s' % (feconf.COLLECTION_URL_PREFIX, self.COLLECTION_ID), expected_status_int=404)

    def test_unpublished_collections_are_invisible_to_unconnected_users(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.login(self.NEW_USER_EMAIL)
        self.get_html_response('%s/%s' % (feconf.COLLECTION_URL_PREFIX, self.COLLECTION_ID), expected_status_int=404)
        self.logout()

    def test_unpublished_collections_are_invisible_to_other_editors(self) -> None:
        if False:
            print('Hello World!')
        self.signup(self.OTHER_EDITOR_EMAIL, 'othereditorusername')
        self.save_new_valid_collection('cid2', self.OTHER_EDITOR_EMAIL)
        self.login(self.OTHER_EDITOR_EMAIL)
        self.get_html_response('%s/%s' % (feconf.COLLECTION_URL_PREFIX, self.COLLECTION_ID), expected_status_int=404)
        self.logout()

    def test_unpublished_collections_are_visible_to_their_editors(self) -> None:
        if False:
            while True:
                i = 10
        self.login(self.EDITOR_EMAIL)
        self.get_html_response('%s/%s' % (feconf.COLLECTION_URL_PREFIX, self.COLLECTION_ID))
        self.logout()

    def test_unpublished_collections_are_visible_to_admins(self) -> None:
        if False:
            while True:
                i = 10
        self.signup(self.MODERATOR_EMAIL, self.MODERATOR_USERNAME)
        self.set_moderators([self.MODERATOR_USERNAME])
        self.login(self.MODERATOR_EMAIL)
        self.get_html_response('%s/%s' % (feconf.COLLECTION_URL_PREFIX, self.COLLECTION_ID))
        self.logout()

    def test_published_collections_are_visible_to_logged_out_users(self) -> None:
        if False:
            i = 10
            return i + 15
        rights_manager.publish_collection(self.editor, self.COLLECTION_ID)
        self.get_html_response('%s/%s' % (feconf.COLLECTION_URL_PREFIX, self.COLLECTION_ID))

    def test_published_collections_are_visible_to_logged_in_users(self) -> None:
        if False:
            return 10
        rights_manager.publish_collection(self.editor, self.COLLECTION_ID)
        self.login(self.NEW_USER_EMAIL)
        self.get_html_response('%s/%s' % (feconf.COLLECTION_URL_PREFIX, self.COLLECTION_ID))

    def test_invalid_collection_error(self) -> None:
        if False:
            print('Hello World!')
        self.login(self.EDITOR_EMAIL)
        self.get_html_response('%s/%s' % (feconf.COLLECTION_URL_PREFIX, 'none'), expected_status_int=404)
        self.logout()

class CollectionViewerControllerEndToEndTests(test_utils.GenericTestBase):
    """Test the collection viewer controller using a sample collection."""

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.signup(self.VIEWER_EMAIL, self.VIEWER_USERNAME)
        self.viewer_id = self.get_user_id_from_email(self.VIEWER_EMAIL)

    def test_welcome_collection(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Test a learner's progression through the default collection."
        collection_services.load_demo('0')
        self.login(self.VIEWER_EMAIL)
        response_dict = self.get_json('%s/1' % feconf.COLLECTION_DATA_URL_PREFIX, expected_status_int=404)
        response_dict = self.get_json('%s/0' % feconf.COLLECTION_DATA_URL_PREFIX)
        collection_dict = response_dict['collection']
        self.assertEqual(collection_dict['objective'], 'To introduce collections using demo explorations.')
        self.assertEqual(collection_dict['category'], 'Welcome')
        self.assertEqual(collection_dict['title'], 'Introduction to Collections in Oppia')
        self.assertEqual(len(collection_dict['nodes']), 4)
        playthrough_dict = collection_dict['playthrough_dict']
        self.assertEqual(playthrough_dict['next_exploration_id'], '19')
        self.assertEqual(playthrough_dict['completed_exploration_ids'], [])
        collection_services.record_played_exploration_in_collection_context(self.viewer_id, '0', '19')
        response_dict = self.get_json('%s/0' % feconf.COLLECTION_DATA_URL_PREFIX)
        collection_dict = response_dict['collection']
        playthrough_dict = collection_dict['playthrough_dict']
        self.assertEqual(playthrough_dict['next_exploration_id'], '20')
        self.assertEqual(playthrough_dict['completed_exploration_ids'], ['19'])
        collection_services.record_played_exploration_in_collection_context(self.viewer_id, '0', '20')
        response_dict = self.get_json('%s/0' % feconf.COLLECTION_DATA_URL_PREFIX)
        collection_dict = response_dict['collection']
        playthrough_dict = collection_dict['playthrough_dict']
        self.assertEqual(playthrough_dict['next_exploration_id'], '21')
        self.assertEqual(playthrough_dict['completed_exploration_ids'], ['19', '20'])
        collection_services.record_played_exploration_in_collection_context(self.viewer_id, '0', '21')
        response_dict = self.get_json('%s/0' % feconf.COLLECTION_DATA_URL_PREFIX)
        collection_dict = response_dict['collection']
        playthrough_dict = collection_dict['playthrough_dict']
        self.assertEqual(playthrough_dict['next_exploration_id'], '0')
        self.assertEqual(playthrough_dict['completed_exploration_ids'], ['19', '20', '21'])
        collection_services.record_played_exploration_in_collection_context(self.viewer_id, '0', '0')
        response_dict = self.get_json('%s/0' % feconf.COLLECTION_DATA_URL_PREFIX)
        collection_dict = response_dict['collection']
        playthrough_dict = collection_dict['playthrough_dict']
        self.assertEqual(playthrough_dict['next_exploration_id'], None)
        self.assertEqual(playthrough_dict['completed_exploration_ids'], ['19', '20', '21', '0'])