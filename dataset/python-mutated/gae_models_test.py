"""Tests for core.storage.email.gae_models."""
from __future__ import annotations
import datetime
import types
from core import feconf
from core import utils
from core.platform import models
from core.tests import test_utils
from typing import Final, Sequence
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import email_models
    from mypy_imports import user_models
(base_models, email_models, user_models) = models.Registry.import_models([models.Names.BASE_MODEL, models.Names.EMAIL, models.Names.USER])

class SentEmailModelUnitTests(test_utils.GenericTestBase):
    """Test the SentEmailModel class."""
    SENDER_ID: Final = 'sender_id'
    RECIPIENT_ID: Final = 'recipient_id'
    NONEXISTENT_USER_ID: Final = 'id_x'

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()

        def mock_generate_hash(unused_cls: email_models.SentEmailModel, unused_recipient_id: str, unused_email_subject: str, unused_email_body: str) -> str:
            if False:
                while True:
                    i = 10
            return 'Email Hash'
        self.generate_constant_hash_ctx = self.swap(email_models.SentEmailModel, '_generate_hash', types.MethodType(mock_generate_hash, email_models.SentEmailModel))
        with self.swap(email_models.SentEmailModel, '_generate_hash', types.MethodType(mock_generate_hash, email_models.SentEmailModel)):
            email_models.SentEmailModel.create('recipient_id', 'recipient@email.com', self.SENDER_ID, 'sender@email.com', feconf.EMAIL_INTENT_SIGNUP, 'Email Subject', 'Email Body', datetime.datetime.utcnow())

    def test_get_deletion_policy(self) -> None:
        if False:
            while True:
                i = 10
        self.assertEqual(email_models.SentEmailModel.get_deletion_policy(), base_models.DELETION_POLICY.DELETE)

    def test_has_reference_to_user_id(self) -> None:
        if False:
            return 10
        self.assertTrue(email_models.SentEmailModel.has_reference_to_user_id('recipient_id'))
        self.assertTrue(email_models.SentEmailModel.has_reference_to_user_id(self.SENDER_ID))
        self.assertFalse(email_models.SentEmailModel.has_reference_to_user_id(self.NONEXISTENT_USER_ID))

    def test_apply_deletion_policy_deletes_model_for_user_who_is_sender(self) -> None:
        if False:
            while True:
                i = 10
        email_models.SentEmailModel.apply_deletion_policy(self.SENDER_ID)
        self.assertIsNone(email_models.SentEmailModel.get_by_id(self.SENDER_ID))

    def test_apply_deletion_policy_deletes_model_for_user_who_is_recipient(self) -> None:
        if False:
            while True:
                i = 10
        email_models.SentEmailModel.apply_deletion_policy(self.RECIPIENT_ID)
        self.assertIsNone(email_models.SentEmailModel.get_by_id(self.RECIPIENT_ID))

    def test_apply_deletion_policy_raises_no_exception_for_nonexistent_user(self) -> None:
        if False:
            while True:
                i = 10
        email_models.SentEmailModel.apply_deletion_policy(self.NONEXISTENT_USER_ID)

    def test_saved_model_can_be_retrieved_with_same_hash(self) -> None:
        if False:
            while True:
                i = 10
        query = email_models.SentEmailModel.query()
        query = query.filter(email_models.SentEmailModel.email_hash == 'Email Hash')
        results: Sequence[email_models.SentEmailModel] = query.fetch(2)
        self.assertEqual(len(results), 1)
        query = email_models.SentEmailModel.query()
        query = query.filter(email_models.SentEmailModel.email_hash == 'Bad Email Hash')
        results = query.fetch(2)
        self.assertEqual(len(results), 0)

    def test_get_by_hash_works_correctly(self) -> None:
        if False:
            print('Hello World!')
        results = email_models.SentEmailModel.get_by_hash('Email Hash')
        self.assertEqual(len(results), 1)
        results = email_models.SentEmailModel.get_by_hash('Bad Email Hash')
        self.assertEqual(len(results), 0)

    def test_get_by_hash_returns_multiple_models_with_same_hash(self) -> None:
        if False:
            while True:
                i = 10
        with self.generate_constant_hash_ctx:
            email_models.SentEmailModel.create('recipient_id', 'recipient@email.com', self.SENDER_ID, 'sender@email.com', feconf.EMAIL_INTENT_SIGNUP, 'Email Subject', 'Email Body', datetime.datetime.utcnow())
            results = email_models.SentEmailModel.get_by_hash('Email Hash')
            self.assertEqual(len(results), 2)

    def test_get_by_hash_behavior_with_sent_datetime_lower_bound(self) -> None:
        if False:
            print('Hello World!')
        with self.generate_constant_hash_ctx:
            time_now = datetime.datetime.utcnow()
            email_models.SentEmailModel.create('recipient_id', 'recipient@email.com', self.SENDER_ID, 'sender@email.com', feconf.EMAIL_INTENT_SIGNUP, 'Email Subject', 'Email Body', datetime.datetime.utcnow())
        results = email_models.SentEmailModel.get_by_hash('Email Hash', sent_datetime_lower_bound=time_now)
        self.assertEqual(len(results), 1)
        time_now1 = datetime.datetime.utcnow()
        results = email_models.SentEmailModel.get_by_hash('Email Hash', sent_datetime_lower_bound=time_now1)
        self.assertEqual(len(results), 0)
        time_before = datetime.datetime.utcnow() - datetime.timedelta(minutes=10)
        results = email_models.SentEmailModel.get_by_hash('Email Hash', sent_datetime_lower_bound=time_before)
        self.assertEqual(len(results), 2)
        with self.assertRaisesRegex(Exception, "Expected datetime, received Not a datetime object of type <class 'str'>"):
            email_models.SentEmailModel.get_by_hash('Email Hash', sent_datetime_lower_bound='Not a datetime object')

    def test_get_export_policy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        expected_dict = {'created_on': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'last_updated': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'deleted': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'recipient_id': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'recipient_email': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'sender_id': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'sender_email': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'intent': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'subject': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'html_body': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'sent_datetime': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'email_hash': base_models.EXPORT_POLICY.NOT_APPLICABLE}
        model = email_models.SentEmailModel
        self.assertEqual(model.get_export_policy(), expected_dict)

    def test_get_model_association_to_user(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        model = email_models.SentEmailModel
        self.assertEqual(model.get_model_association_to_user(), base_models.MODEL_ASSOCIATION_TO_USER.NOT_CORRESPONDING_TO_USER)

    def test_check_duplicate_message(self) -> None:
        if False:
            while True:
                i = 10
        email_models.SentEmailModel.create('recipient_id', 'recipient@email.com', self.SENDER_ID, 'sender@email.com', feconf.EMAIL_INTENT_SIGNUP, 'Email Subject', 'Email Body', datetime.datetime.utcnow())
        self.assertTrue(email_models.SentEmailModel.check_duplicate_message('recipient_id', 'Email Subject', 'Email Body'))
        email_models.SentEmailModel.create('recipient_id2', 'recipient@email.com', self.SENDER_ID, 'sender@email.com', feconf.EMAIL_INTENT_SIGNUP, 'Email Subject', 'Email Body', datetime.datetime.utcnow() - datetime.timedelta(minutes=feconf.DUPLICATE_EMAIL_INTERVAL_MINS))
        self.assertFalse(email_models.SentEmailModel.check_duplicate_message('recipient_id2', 'Email Subject', 'Email Body'))

    def test_check_duplicate_messages_with_same_hash(self) -> None:
        if False:
            return 10

        def mock_convert_to_hash(input_string: str, max_length: int) -> str:
            if False:
                while True:
                    i = 10
            return 'some_poor_hash'
        swap_generate_hash = self.swap(utils, 'convert_to_hash', mock_convert_to_hash)
        with swap_generate_hash:
            email_models.SentEmailModel.create('recipient_id', 'recipient@email.com', self.SENDER_ID, 'sender@email.com', feconf.EMAIL_INTENT_SIGNUP, 'Email Subject', 'Email Body', datetime.datetime.utcnow())
            self.assertFalse(email_models.SentEmailModel.check_duplicate_message('recipient_id2', 'Email Subject2', 'Email Body2'))

    def test_raise_exception_by_mocking_collision(self) -> None:
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(Exception, 'The id generator for SentEmailModel is producing too many collisions.'):
            with self.swap(email_models.SentEmailModel, 'get_by_id', types.MethodType(lambda x, y: True, email_models.SentEmailModel)):
                email_models.SentEmailModel.create('recipient_id', 'recipient@email.com', 'sender_id', 'sender@email.com', feconf.EMAIL_INTENT_SIGNUP, 'Email Subject', 'Email Body', datetime.datetime.utcnow())

class BulkEmailModelUnitTests(test_utils.GenericTestBase):
    """Test the BulkEmailModel class."""
    SENDER_ID: Final = 'sender_id'
    NONEXISTENT_USER_ID: Final = 'id_x'

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        email_models.BulkEmailModel.create('instance_id', self.SENDER_ID, 'sender@email.com', feconf.BULK_EMAIL_INTENT_MARKETING, 'Email Subject', 'Email Body', datetime.datetime.utcnow())

    def test_get_deletion_policy(self) -> None:
        if False:
            print('Hello World!')
        self.assertEqual(email_models.BulkEmailModel.get_deletion_policy(), base_models.DELETION_POLICY.DELETE)

    def test_has_reference_to_user_id(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(email_models.BulkEmailModel.has_reference_to_user_id(self.SENDER_ID))
        self.assertFalse(email_models.BulkEmailModel.has_reference_to_user_id(self.NONEXISTENT_USER_ID))

    def test_apply_deletion_policy_deletes_model_for_user_who_is_sender(self) -> None:
        if False:
            return 10
        email_models.BulkEmailModel.apply_deletion_policy(self.SENDER_ID)
        self.assertIsNone(email_models.BulkEmailModel.get_by_id(self.SENDER_ID))

    def test_apply_deletion_policy_raises_no_exception_for_nonexistent_user(self) -> None:
        if False:
            return 10
        email_models.BulkEmailModel.apply_deletion_policy(self.NONEXISTENT_USER_ID)

    def test_get_export_policy(self) -> None:
        if False:
            return 10
        expected_dict = {'created_on': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'last_updated': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'deleted': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'sender_id': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'sender_email': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'recipient_ids': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'intent': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'subject': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'html_body': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'sent_datetime': base_models.EXPORT_POLICY.NOT_APPLICABLE}
        model = email_models.BulkEmailModel
        self.assertEqual(model.get_export_policy(), expected_dict)

    def test_get_model_association_to_user(self) -> None:
        if False:
            print('Hello World!')
        model = email_models.BulkEmailModel
        self.assertEqual(model.get_model_association_to_user(), base_models.MODEL_ASSOCIATION_TO_USER.NOT_CORRESPONDING_TO_USER)

class GenerateHashTests(test_utils.GenericTestBase):
    """Test that generating hash functionality works as expected."""

    def test_same_inputs_always_gives_same_hashes(self) -> None:
        if False:
            print('Hello World!')
        email_model_instance = email_models.SentEmailModel(id='exp_id.new_id', recipient_id='recipient_id', recipient_email='recipient@email.com', sender_id='sender_id', sender_email='sender@email.com', intent=feconf.EMAIL_INTENT_SIGNUP, subject='email_subject', html_body='email_html_body', sent_datetime=datetime.datetime.utcnow())
        email_model_instance.update_timestamps()
        email_model_instance.put()
        email_hash1 = email_model_instance.email_hash
        email_hash2 = email_model_instance.email_hash
        self.assertEqual(email_hash1, email_hash2)

    def test_different_inputs_give_different_hashes(self) -> None:
        if False:
            return 10
        email_model_instance = email_models.SentEmailModel(id='exp_id.new_id', recipient_id='recipient_id', recipient_email='recipient@email.com', sender_id='sender_id', sender_email='sender@email.com', intent=feconf.EMAIL_INTENT_SIGNUP, subject='email_subject', html_body='email_html_body', sent_datetime=datetime.datetime.utcnow())
        email_model_instance.update_timestamps()
        email_model_instance.put()
        email_model_instance2 = email_models.SentEmailModel(id='exp_id.new_id2', recipient_id='recipient_id', recipient_email='recipient@email.com', sender_id='sender_id', sender_email='sender@email.com', intent=feconf.EMAIL_INTENT_SIGNUP, subject='email_subject', html_body='email_html_body2', sent_datetime=datetime.datetime.utcnow())
        email_model_instance2.update_timestamps()
        email_model_instance2.put()
        email_hash1 = email_model_instance.email_hash
        email_hash2 = email_model_instance2.email_hash
        self.assertNotEqual(email_hash1, email_hash2)
        email_model_instance2 = email_models.SentEmailModel(id='exp_id.new_id2', recipient_id='recipient_id2', recipient_email='recipient@email.com', sender_id='sender_id', sender_email='sender@email.com', intent=feconf.EMAIL_INTENT_SIGNUP, subject='email_subject', html_body='email_html_body', sent_datetime=datetime.datetime.utcnow())
        email_model_instance2.update_timestamps()
        email_model_instance2.put()
        email_hash2 = email_model_instance2.email_hash
        self.assertNotEqual(email_hash1, email_hash2)
        email_model_instance2 = email_models.SentEmailModel(id='exp_id.new_id2', recipient_id='recipient_id', recipient_email='recipient@email.com', sender_id='sender_id', sender_email='sender@email.com', intent=feconf.EMAIL_INTENT_SIGNUP, subject='email_subject2', html_body='email_html_body', sent_datetime=datetime.datetime.utcnow())
        email_model_instance2.update_timestamps()
        email_model_instance2.put()
        email_hash2 = email_model_instance2.email_hash
        self.assertNotEqual(email_hash1, email_hash2)
        email_model_instance2 = email_models.SentEmailModel(id='exp_id.new_id2', recipient_id='recipient_id2', recipient_email='recipient@email.com', sender_id='sender_id', sender_email='sender@email.com', intent=feconf.EMAIL_INTENT_SIGNUP, subject='email_subject2', html_body='email_html_body2', sent_datetime=datetime.datetime.utcnow())
        email_model_instance2.update_timestamps()
        email_model_instance2.put()
        email_hash2 = email_model_instance2.email_hash
        self.assertNotEqual(email_hash1, email_hash2)