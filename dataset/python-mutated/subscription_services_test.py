"""Tests for subscription management."""
from __future__ import annotations
from core import feconf
from core.domain import collection_domain
from core.domain import collection_services
from core.domain import exp_domain
from core.domain import exp_services
from core.domain import feedback_services
from core.domain import rights_domain
from core.domain import rights_manager
from core.domain import subscription_services
from core.domain import user_services
from core.platform import models
from core.tests import test_utils
from typing import Final, List
MYPY = False
if MYPY:
    from mypy_imports import user_models
(user_models,) = models.Registry.import_models([models.Names.USER])
COLLECTION_ID: Final = 'col_id'
COLLECTION_ID_2: Final = 'col_id_2'
EXP_ID: Final = 'exp_id'
EXP_ID_2: Final = 'exp_id_2'
FEEDBACK_THREAD_ID: Final = 'fthread_id'
FEEDBACK_THREAD_ID_2: Final = 'fthread_id_2'
USER_ID: Final = 'user_id'
USER_ID_2: Final = 'user_id_2'

class SubscriptionsTest(test_utils.GenericTestBase):
    """Tests for subscription management."""
    OWNER_2_EMAIL: Final = 'owner2@example.com'
    OWNER2_USERNAME: Final = 'owner2'

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.signup(self.OWNER_EMAIL, self.OWNER_USERNAME)
        self.signup(self.EDITOR_EMAIL, self.EDITOR_USERNAME)
        self.signup(self.VIEWER_EMAIL, self.VIEWER_USERNAME)
        self.signup(self.OWNER_2_EMAIL, self.OWNER2_USERNAME)
        self.owner_id = self.get_user_id_from_email(self.OWNER_EMAIL)
        self.editor_id = self.get_user_id_from_email(self.EDITOR_EMAIL)
        self.viewer_id = self.get_user_id_from_email(self.VIEWER_EMAIL)
        self.owner_2_id = self.get_user_id_from_email(self.OWNER_2_EMAIL)
        self.owner = user_services.get_user_actions_info(self.owner_id)

    def _get_thread_ids_subscribed_to(self, user_id: str) -> List[str]:
        if False:
            while True:
                i = 10
        'Returns the feedback thread ids to which the user corresponding to\n        the given user id is subscribed to.\n\n        Args:\n            user_id: str. The user id.\n\n        Returns:\n            List(str). The list containing all the feedback thread ids to\n            which the user is subscribed to.\n        '
        subscriptions_model = user_models.UserSubscriptionsModel.get(user_id, strict=False)
        if subscriptions_model:
            feedback_thread_ids: List[str] = subscriptions_model.general_feedback_thread_ids
            return feedback_thread_ids
        else:
            return []

    def _get_exploration_ids_subscribed_to(self, user_id: str) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Returns all the exploration ids of the explorations to which the user\n        has subscribed to.\n\n        Args:\n            user_id: str. The user id.\n\n        Returns:\n            List(str). The list containing all the exploration ids of the\n            explorations to which the user has subscribed to.\n        '
        subscriptions_model = user_models.UserSubscriptionsModel.get(user_id, strict=False)
        if subscriptions_model:
            exploration_ids: List[str] = subscriptions_model.exploration_ids
            return exploration_ids
        else:
            return []

    def _get_collection_ids_subscribed_to(self, user_id: str) -> List[str]:
        if False:
            print('Hello World!')
        'Returns all the collection ids of the collections to which the user\n        has subscribed to.\n\n        Args:\n            user_id: str. The user id.\n\n        Returns:\n            List(str). The list containing all the collection ids of the\n            collections to which the user has subscribed to.\n        '
        subscriptions_model = user_models.UserSubscriptionsModel.get(user_id, strict=False)
        if subscriptions_model:
            collection_ids: List[str] = subscriptions_model.collection_ids
            return collection_ids
        else:
            return []

    def test_subscribe_to_feedback_thread(self) -> None:
        if False:
            return 10
        self.assertEqual(self._get_thread_ids_subscribed_to(USER_ID), [])
        subscription_services.subscribe_to_thread(USER_ID, FEEDBACK_THREAD_ID)
        self.assertEqual(self._get_thread_ids_subscribed_to(USER_ID), [FEEDBACK_THREAD_ID])
        subscription_services.subscribe_to_thread(USER_ID, FEEDBACK_THREAD_ID)
        self.assertEqual(self._get_thread_ids_subscribed_to(USER_ID), [FEEDBACK_THREAD_ID])
        subscription_services.subscribe_to_thread(USER_ID, FEEDBACK_THREAD_ID_2)
        self.assertEqual(self._get_thread_ids_subscribed_to(USER_ID), [FEEDBACK_THREAD_ID, FEEDBACK_THREAD_ID_2])

    def test_subscribe_to_exploration(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(self._get_exploration_ids_subscribed_to(USER_ID), [])
        subscription_services.subscribe_to_exploration(USER_ID, EXP_ID)
        self.assertEqual(self._get_exploration_ids_subscribed_to(USER_ID), [EXP_ID])
        subscription_services.subscribe_to_exploration(USER_ID, EXP_ID)
        self.assertEqual(self._get_exploration_ids_subscribed_to(USER_ID), [EXP_ID])
        subscription_services.subscribe_to_exploration(USER_ID, EXP_ID_2)
        self.assertEqual(self._get_exploration_ids_subscribed_to(USER_ID), [EXP_ID, EXP_ID_2])

    def test_get_exploration_ids_subscribed_to(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(subscription_services.get_exploration_ids_subscribed_to(USER_ID), [])
        subscription_services.subscribe_to_exploration(USER_ID, EXP_ID)
        self.assertEqual(subscription_services.get_exploration_ids_subscribed_to(USER_ID), [EXP_ID])
        subscription_services.subscribe_to_exploration(USER_ID, EXP_ID_2)
        self.assertEqual(subscription_services.get_exploration_ids_subscribed_to(USER_ID), [EXP_ID, EXP_ID_2])

    def test_get_all_threads_subscribed_to(self) -> None:
        if False:
            return 10
        self.assertEqual(subscription_services.get_all_threads_subscribed_to(USER_ID), [])
        subscription_services.subscribe_to_thread(USER_ID, FEEDBACK_THREAD_ID)
        self.assertEqual(subscription_services.get_all_threads_subscribed_to(USER_ID), [FEEDBACK_THREAD_ID])
        subscription_services.subscribe_to_thread(USER_ID, FEEDBACK_THREAD_ID_2)
        self.assertEqual(subscription_services.get_all_threads_subscribed_to(USER_ID), [FEEDBACK_THREAD_ID, FEEDBACK_THREAD_ID_2])

    def test_thread_and_exp_subscriptions_are_tracked_individually(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(self._get_thread_ids_subscribed_to(USER_ID), [])
        subscription_services.subscribe_to_thread(USER_ID, FEEDBACK_THREAD_ID)
        subscription_services.subscribe_to_exploration(USER_ID, EXP_ID)
        self.assertEqual(self._get_thread_ids_subscribed_to(USER_ID), [FEEDBACK_THREAD_ID])
        self.assertEqual(self._get_exploration_ids_subscribed_to(USER_ID), [EXP_ID])

    def test_posting_to_feedback_thread_results_in_subscription(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        message_text = 'text'
        feedback_services.create_thread(feconf.ENTITY_TYPE_EXPLORATION, 'exp_id', self.viewer_id, 'subject', message_text)
        thread_ids_subscribed_to = self._get_thread_ids_subscribed_to(self.viewer_id)
        self.assertEqual(len(thread_ids_subscribed_to), 1)
        thread_id = thread_ids_subscribed_to[0]
        self.assertEqual(feedback_services.get_messages(thread_id)[0].text, message_text)
        new_message_text = 'new text'
        feedback_services.create_message(thread_id, self.editor_id, '', '', new_message_text)
        self.assertEqual(self._get_thread_ids_subscribed_to(self.viewer_id), [thread_id])
        self.assertEqual(self._get_thread_ids_subscribed_to(self.editor_id), [thread_id])

    def test_creating_exploration_results_in_subscription(self) -> None:
        if False:
            return 10
        self.assertEqual(self._get_exploration_ids_subscribed_to(USER_ID), [])
        exp_services.save_new_exploration(USER_ID, exp_domain.Exploration.create_default_exploration(EXP_ID))
        self.assertEqual(self._get_exploration_ids_subscribed_to(USER_ID), [EXP_ID])

    def test_adding_new_exploration_owner_or_editor_role_results_in_subscription(self) -> None:
        if False:
            print('Hello World!')
        exploration = exp_domain.Exploration.create_default_exploration(EXP_ID)
        exp_services.save_new_exploration(self.owner_id, exploration)
        self.assertEqual(self._get_exploration_ids_subscribed_to(self.owner_2_id), [])
        rights_manager.assign_role_for_exploration(self.owner, EXP_ID, self.owner_2_id, rights_domain.ROLE_OWNER)
        self.assertEqual(self._get_exploration_ids_subscribed_to(self.owner_2_id), [EXP_ID])
        self.assertEqual(self._get_exploration_ids_subscribed_to(self.editor_id), [])
        rights_manager.assign_role_for_exploration(self.owner, EXP_ID, self.editor_id, rights_domain.ROLE_EDITOR)
        self.assertEqual(self._get_exploration_ids_subscribed_to(self.editor_id), [EXP_ID])

    def test_adding_new_exploration_viewer_role_does_not_result_in_subscription(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        exploration = exp_domain.Exploration.create_default_exploration(EXP_ID)
        exp_services.save_new_exploration(self.owner_id, exploration)
        self.assertEqual(self._get_exploration_ids_subscribed_to(self.viewer_id), [])
        rights_manager.assign_role_for_exploration(self.owner, EXP_ID, self.viewer_id, rights_domain.ROLE_VIEWER)
        self.assertEqual(self._get_exploration_ids_subscribed_to(self.viewer_id), [])

    def test_deleting_exploration_does_not_delete_subscription(self) -> None:
        if False:
            print('Hello World!')
        exploration = exp_domain.Exploration.create_default_exploration(EXP_ID)
        exp_services.save_new_exploration(self.owner_id, exploration)
        self.assertEqual(self._get_exploration_ids_subscribed_to(self.owner_id), [EXP_ID])
        exp_services.delete_exploration(self.owner_id, EXP_ID)
        self.assertEqual(self._get_exploration_ids_subscribed_to(self.owner_id), [EXP_ID])

    def test_subscribe_to_collection(self) -> None:
        if False:
            return 10
        self.assertEqual(self._get_collection_ids_subscribed_to(USER_ID), [])
        subscription_services.subscribe_to_collection(USER_ID, COLLECTION_ID)
        self.assertEqual(self._get_collection_ids_subscribed_to(USER_ID), [COLLECTION_ID])
        subscription_services.subscribe_to_collection(USER_ID, COLLECTION_ID)
        self.assertEqual(self._get_collection_ids_subscribed_to(USER_ID), [COLLECTION_ID])
        subscription_services.subscribe_to_collection(USER_ID, COLLECTION_ID_2)
        self.assertEqual(self._get_collection_ids_subscribed_to(USER_ID), [COLLECTION_ID, COLLECTION_ID_2])

    def test_get_collection_ids_subscribed_to(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(subscription_services.get_collection_ids_subscribed_to(USER_ID), [])
        subscription_services.subscribe_to_collection(USER_ID, COLLECTION_ID)
        self.assertEqual(subscription_services.get_collection_ids_subscribed_to(USER_ID), [COLLECTION_ID])
        subscription_services.subscribe_to_collection(USER_ID, COLLECTION_ID_2)
        self.assertEqual(subscription_services.get_collection_ids_subscribed_to(USER_ID), [COLLECTION_ID, COLLECTION_ID_2])

    def test_creating_collection_results_in_subscription(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(self._get_collection_ids_subscribed_to(USER_ID), [])
        self.save_new_default_collection(COLLECTION_ID, USER_ID)
        self.assertEqual(self._get_collection_ids_subscribed_to(USER_ID), [COLLECTION_ID])

    def test_adding_new_collection_owner_or_editor_role_results_in_subscription(self) -> None:
        if False:
            while True:
                i = 10
        self.save_new_default_collection(COLLECTION_ID, self.owner_id)
        self.assertEqual(self._get_collection_ids_subscribed_to(self.owner_2_id), [])
        rights_manager.assign_role_for_collection(self.owner, COLLECTION_ID, self.owner_2_id, rights_domain.ROLE_OWNER)
        self.assertEqual(self._get_collection_ids_subscribed_to(self.owner_2_id), [COLLECTION_ID])
        self.assertEqual(self._get_collection_ids_subscribed_to(self.editor_id), [])
        rights_manager.assign_role_for_collection(self.owner, COLLECTION_ID, self.editor_id, rights_domain.ROLE_EDITOR)
        self.assertEqual(self._get_collection_ids_subscribed_to(self.editor_id), [COLLECTION_ID])

    def test_adding_new_collection_viewer_role_does_not_result_in_subscription(self) -> None:
        if False:
            while True:
                i = 10
        self.save_new_default_collection(COLLECTION_ID, self.owner_id)
        self.assertEqual(self._get_collection_ids_subscribed_to(self.viewer_id), [])
        rights_manager.assign_role_for_collection(self.owner, COLLECTION_ID, self.viewer_id, rights_domain.ROLE_VIEWER)
        self.assertEqual(self._get_collection_ids_subscribed_to(self.viewer_id), [])

    def test_deleting_collection_does_not_delete_subscription(self) -> None:
        if False:
            while True:
                i = 10
        self.save_new_default_collection(COLLECTION_ID, self.owner_id)
        self.assertEqual(self._get_collection_ids_subscribed_to(self.owner_id), [COLLECTION_ID])
        collection_services.delete_collection(self.owner_id, COLLECTION_ID)
        self.assertEqual(self._get_collection_ids_subscribed_to(self.owner_id), [COLLECTION_ID])

    def test_adding_exploration_to_collection_does_not_create_subscription(self) -> None:
        if False:
            print('Hello World!')
        self.save_new_default_collection(COLLECTION_ID, self.owner_id)
        self.assertEqual(self._get_collection_ids_subscribed_to(self.owner_id), [COLLECTION_ID])
        self.assertEqual(self._get_exploration_ids_subscribed_to(self.owner_id), [])
        self.save_new_valid_exploration(EXP_ID, self.owner_2_id)
        collection_services.update_collection(self.owner_id, COLLECTION_ID, [{'cmd': collection_domain.CMD_ADD_COLLECTION_NODE, 'exploration_id': EXP_ID}], 'Add new exploration to collection.')
        self.assertEqual(self._get_collection_ids_subscribed_to(self.owner_id), [COLLECTION_ID])
        self.assertEqual(self._get_exploration_ids_subscribed_to(self.owner_2_id), [EXP_ID])

class UserSubscriptionsTest(test_utils.GenericTestBase):
    """Tests for subscription management."""
    OWNER_2_EMAIL: Final = 'owner2@example.com'
    OWNER2_USERNAME: Final = 'owner2'

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.signup(self.OWNER_EMAIL, self.OWNER_USERNAME)
        self.signup(self.OWNER_2_EMAIL, self.OWNER2_USERNAME)
        self.owner_id = self.get_user_id_from_email(self.OWNER_EMAIL)
        self.owner_2_id = self.get_user_id_from_email(self.OWNER_2_EMAIL)

    def _get_all_subscribers_of_creator(self, user_id: str) -> List[str]:
        if False:
            while True:
                i = 10
        'Returns all the ids of the subscribers that have subscribed to the\n        creator.\n\n        Args:\n            user_id: str. The user id.\n\n        Returns:\n            List(str). The list containing all the ids of the subscribers that\n            have subscribed to the creator.\n        '
        subscribers_model = user_models.UserSubscribersModel.get(user_id, strict=False)
        if subscribers_model:
            subscriber_ids: List[str] = subscribers_model.subscriber_ids
            return subscriber_ids
        else:
            return []

    def _get_all_creators_subscribed_to(self, user_id: str) -> List[str]:
        if False:
            while True:
                i = 10
        'Returns the ids of the creators the given user has subscribed to.\n\n        Args:\n            user_id: str. The user id.\n\n        Returns:\n            List(str). The list containing all the creator ids the given user\n            has subscribed to.\n        '
        subscriptions_model = user_models.UserSubscriptionsModel.get(user_id, strict=False)
        if subscriptions_model:
            creator_ids: List[str] = subscriptions_model.creator_ids
            return creator_ids
        else:
            return []

    def test_exception_is_raised_when_user_self_subscribes(self) -> None:
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(Exception, 'User %s is not allowed to self subscribe.' % USER_ID):
            subscription_services.subscribe_to_creator(USER_ID, USER_ID)

    def test_subscribe_to_creator(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(self._get_all_subscribers_of_creator(self.owner_id), [])
        subscription_services.subscribe_to_creator(USER_ID, self.owner_id)
        self.assertEqual(self._get_all_subscribers_of_creator(self.owner_id), [USER_ID])
        self.assertEqual(self._get_all_creators_subscribed_to(USER_ID), [self.owner_id])
        subscription_services.subscribe_to_creator(USER_ID, self.owner_id)
        self.assertEqual(self._get_all_subscribers_of_creator(self.owner_id), [USER_ID])
        self.assertEqual(self._get_all_creators_subscribed_to(USER_ID), [self.owner_id])
        subscription_services.subscribe_to_creator(USER_ID_2, self.owner_id)
        self.assertEqual(self._get_all_subscribers_of_creator(self.owner_id), [USER_ID, USER_ID_2])
        self.assertEqual(self._get_all_creators_subscribed_to(USER_ID_2), [self.owner_id])

    def test_unsubscribe_from_creator(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(self._get_all_subscribers_of_creator(self.owner_id), [])
        subscription_services.subscribe_to_creator(USER_ID, self.owner_id)
        subscription_services.subscribe_to_creator(USER_ID_2, self.owner_id)
        self.assertEqual(self._get_all_subscribers_of_creator(self.owner_id), [USER_ID, USER_ID_2])
        self.assertEqual(self._get_all_creators_subscribed_to(USER_ID), [self.owner_id])
        self.assertEqual(self._get_all_creators_subscribed_to(USER_ID_2), [self.owner_id])
        subscription_services.unsubscribe_from_creator(USER_ID, self.owner_id)
        self.assertEqual(self._get_all_subscribers_of_creator(self.owner_id), [USER_ID_2])
        self.assertEqual(self._get_all_creators_subscribed_to(USER_ID), [])
        subscription_services.unsubscribe_from_creator(USER_ID, self.owner_id)
        self.assertEqual(self._get_all_subscribers_of_creator(self.owner_id), [USER_ID_2])
        self.assertEqual(self._get_all_creators_subscribed_to(USER_ID), [])
        subscription_services.unsubscribe_from_creator(USER_ID_2, self.owner_id)
        self.assertEqual(self._get_all_subscribers_of_creator(self.owner_id), [])
        self.assertEqual(self._get_all_creators_subscribed_to(USER_ID_2), [])

    def test_get_all_subscribers_of_creator(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(subscription_services.get_all_subscribers_of_creator(self.owner_id), [])
        subscription_services.subscribe_to_creator(USER_ID, self.owner_id)
        self.assertEqual(subscription_services.get_all_subscribers_of_creator(self.owner_id), [USER_ID])
        subscription_services.subscribe_to_creator(USER_ID_2, self.owner_id)
        self.assertEqual(subscription_services.get_all_subscribers_of_creator(self.owner_id), [USER_ID, USER_ID_2])

    def test_get_all_creators_subscribed_to(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(subscription_services.get_all_creators_subscribed_to(USER_ID), [])
        subscription_services.subscribe_to_creator(USER_ID, self.owner_id)
        self.assertEqual(subscription_services.get_all_creators_subscribed_to(USER_ID), [self.owner_id])
        subscription_services.subscribe_to_creator(USER_ID, self.owner_2_id)
        self.assertEqual(subscription_services.get_all_creators_subscribed_to(USER_ID), [self.owner_id, self.owner_2_id])