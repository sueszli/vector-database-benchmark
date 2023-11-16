"""Models for the content of sent emails."""
from __future__ import annotations
import datetime
from core import feconf
from core import utils
from core.platform import models
from typing import Dict, Optional, Sequence
MYPY = False
if MYPY:
    from mypy_imports import base_models
    from mypy_imports import datastore_services
(base_models,) = models.Registry.import_models([models.Names.BASE_MODEL])
datastore_services = models.Registry.import_datastore_services()

class SentEmailModel(base_models.BaseModel):
    """Records the content and metadata of an email sent from Oppia.

    This model is read-only; entries cannot be modified once created. The
    id/key of instances of this class has the form '[intent].[random hash]'.
    """
    recipient_id = datastore_services.StringProperty(required=True, indexed=True)
    recipient_email = datastore_services.StringProperty(required=True)
    sender_id = datastore_services.StringProperty(required=True, indexed=True)
    sender_email = datastore_services.StringProperty(required=True)
    intent = datastore_services.StringProperty(required=True, indexed=True, choices=[feconf.EMAIL_INTENT_SIGNUP, feconf.EMAIL_INTENT_MARKETING, feconf.EMAIL_INTENT_DAILY_BATCH, feconf.EMAIL_INTENT_EDITOR_ROLE_NOTIFICATION, feconf.EMAIL_INTENT_FEEDBACK_MESSAGE_NOTIFICATION, feconf.EMAIL_INTENT_SUBSCRIPTION_NOTIFICATION, feconf.EMAIL_INTENT_SUGGESTION_NOTIFICATION, feconf.EMAIL_INTENT_UNPUBLISH_EXPLORATION, feconf.EMAIL_INTENT_DELETE_EXPLORATION, feconf.EMAIL_INTENT_REPORT_BAD_CONTENT, feconf.EMAIL_INTENT_QUERY_STATUS_NOTIFICATION, feconf.EMAIL_INTENT_ONBOARD_REVIEWER, feconf.EMAIL_INTENT_REMOVE_REVIEWER, feconf.EMAIL_INTENT_ADDRESS_CONTRIBUTOR_DASHBOARD_SUGGESTIONS, feconf.EMAIL_INTENT_REVIEW_CREATOR_DASHBOARD_SUGGESTIONS, feconf.EMAIL_INTENT_REVIEW_CONTRIBUTOR_DASHBOARD_SUGGESTIONS, feconf.EMAIL_INTENT_ADD_CONTRIBUTOR_DASHBOARD_REVIEWERS, feconf.EMAIL_INTENT_ACCOUNT_DELETED, feconf.BULK_EMAIL_INTENT_TEST, feconf.EMAIL_INTENT_NOTIFY_CONTRIBUTOR_DASHBOARD_ACHIEVEMENTS, feconf.EMAIL_INTENT_ML_JOB_FAILURE])
    subject = datastore_services.TextProperty(required=True)
    html_body = datastore_services.TextProperty(required=True)
    sent_datetime = datastore_services.DateTimeProperty(required=True, indexed=True)
    email_hash = datastore_services.StringProperty(indexed=True)

    @staticmethod
    def get_deletion_policy() -> base_models.DELETION_POLICY:
        if False:
            for i in range(10):
                print('nop')
        'Model contains data corresponding to a user: recipient_id,\n        recipient_email, sender_id, and sender_email.\n        '
        return base_models.DELETION_POLICY.DELETE

    @staticmethod
    def get_model_association_to_user() -> base_models.MODEL_ASSOCIATION_TO_USER:
        if False:
            while True:
                i = 10
        'Users already have access to this data since emails were sent\n        to them.\n        '
        return base_models.MODEL_ASSOCIATION_TO_USER.NOT_CORRESPONDING_TO_USER

    @classmethod
    def get_export_policy(cls) -> Dict[str, base_models.EXPORT_POLICY]:
        if False:
            print('Hello World!')
        "Model contains data corresponding to a user, but this isn't exported\n        because users already have access to noteworthy details of this data\n        (since emails were sent to them).\n        "
        return dict(super(cls, cls).get_export_policy(), **{'recipient_id': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'recipient_email': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'sender_id': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'sender_email': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'intent': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'subject': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'html_body': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'sent_datetime': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'email_hash': base_models.EXPORT_POLICY.NOT_APPLICABLE})

    @classmethod
    def apply_deletion_policy(cls, user_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Delete instances of SentEmailModel for the user.\n\n        Args:\n            user_id: str. The ID of the user whose data should be deleted.\n        '
        keys = cls.query(datastore_services.any_of(cls.recipient_id == user_id, cls.sender_id == user_id)).fetch(keys_only=True)
        datastore_services.delete_multi(keys)

    @classmethod
    def has_reference_to_user_id(cls, user_id: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Check whether SentEmailModel exists for user.\n\n        Args:\n            user_id: str. The ID of the user whose data should be checked.\n\n        Returns:\n            bool. Whether any models refer to the given user ID.\n        '
        return cls.query(datastore_services.any_of(cls.recipient_id == user_id, cls.sender_id == user_id)).get(keys_only=True) is not None

    @classmethod
    def _generate_id(cls, intent: str) -> str:
        if False:
            while True:
                i = 10
        'Generates an ID for a new SentEmailModel instance.\n\n        Args:\n            intent: str. The intent string, i.e. the purpose of the email.\n                Valid intent strings are defined in feconf.py.\n\n        Returns:\n            str. The newly-generated ID for the SentEmailModel instance.\n\n        Raises:\n            Exception. The id generator for SentEmailModel is producing\n                too many collisions.\n        '
        id_prefix = '%s.' % intent
        for _ in range(base_models.MAX_RETRIES):
            new_id = '%s.%s' % (id_prefix, utils.convert_to_hash(str(utils.get_random_int(base_models.RAND_RANGE)), base_models.ID_LENGTH))
            if not cls.get_by_id(new_id):
                return new_id
        raise Exception('The id generator for SentEmailModel is producing too many collisions.')

    @classmethod
    def create(cls, recipient_id: str, recipient_email: str, sender_id: str, sender_email: str, intent: str, subject: str, html_body: str, sent_datetime: datetime.datetime) -> None:
        if False:
            while True:
                i = 10
        'Creates a new SentEmailModel entry.\n\n        Args:\n            recipient_id: str. The user ID of the email recipient.\n            recipient_email: str. The email address of the recipient.\n            sender_id: str. The user ID of the email sender.\n            sender_email: str. The email address used to send the notification.\n            intent: str. The intent string, i.e. the purpose of the email.\n            subject: str. The subject line of the email.\n            html_body: str. The HTML content of the email body.\n            sent_datetime: datetime.datetime. The datetime the email was sent,\n                in UTC.\n        '
        instance_id = cls._generate_id(intent)
        email_model_instance = cls(id=instance_id, recipient_id=recipient_id, recipient_email=recipient_email, sender_id=sender_id, sender_email=sender_email, intent=intent, subject=subject, html_body=html_body, sent_datetime=sent_datetime)
        email_model_instance.update_timestamps()
        email_model_instance.put()

    def _pre_put_hook(self) -> None:
        if False:
            return 10
        'Operations to perform just before the model is `put` into storage.'
        super()._pre_put_hook()
        self.email_hash = self._generate_hash(self.recipient_id, self.subject, self.html_body)

    @classmethod
    def get_by_hash(cls, email_hash: str, sent_datetime_lower_bound: Optional[datetime.datetime]=None) -> Sequence[SentEmailModel]:
        if False:
            i = 10
            return i + 15
        'Returns all messages with a given email_hash.\n\n        This also takes an optional sent_datetime_lower_bound argument,\n        which is a datetime instance. If this is given, only\n        SentEmailModel instances sent after sent_datetime_lower_bound\n        should be returned.\n\n        Args:\n            email_hash: str. The hash value of the email.\n            sent_datetime_lower_bound: datetime.datetime. The lower bound on\n                sent_datetime of the email to be searched.\n\n        Returns:\n            list(SentEmailModel). A list of emails which have the given hash\n            value and sent more recently than sent_datetime_lower_bound.\n\n        Raises:\n            Exception. The sent_datetime_lower_bound is not a valid\n                datetime.datetime.\n        '
        if sent_datetime_lower_bound is not None:
            if not isinstance(sent_datetime_lower_bound, datetime.datetime):
                raise Exception('Expected datetime, received %s of type %s' % (sent_datetime_lower_bound, type(sent_datetime_lower_bound)))
        query = cls.query().filter(cls.email_hash == email_hash)
        if sent_datetime_lower_bound is not None:
            query = query.filter(cls.sent_datetime > sent_datetime_lower_bound)
        return query.fetch()

    @classmethod
    def _generate_hash(cls, recipient_id: str, email_subject: str, email_body: str) -> str:
        if False:
            print('Hello World!')
        'Generate hash for a given recipient_id, email_subject and cleaned\n        email_body.\n\n        Args:\n            recipient_id: str. The user ID of the email recipient.\n            email_subject: str. The subject line of the email.\n            email_body: str. The HTML content of the email body.\n\n        Returns:\n            str. The generated hash value of the given email.\n        '
        hash_value = utils.convert_to_hash(recipient_id + email_subject + email_body, 100)
        return hash_value

    @classmethod
    def check_duplicate_message(cls, recipient_id: str, email_subject: str, email_body: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Check for a given recipient_id, email_subject and cleaned\n        email_body, whether a similar message has been sent in the last\n        DUPLICATE_EMAIL_INTERVAL_MINS.\n\n        Args:\n            recipient_id: str. The user ID of the email recipient.\n            email_subject: str. The subject line of the email.\n            email_body: str. The HTML content of the email body.\n\n        Returns:\n            bool. Whether a similar message has been sent to the same recipient\n            in the last DUPLICATE_EMAIL_INTERVAL_MINS.\n        '
        email_hash = cls._generate_hash(recipient_id, email_subject, email_body)
        datetime_now = datetime.datetime.utcnow()
        time_interval = datetime.timedelta(minutes=feconf.DUPLICATE_EMAIL_INTERVAL_MINS)
        sent_datetime_lower_bound = datetime_now - time_interval
        messages = cls.get_by_hash(email_hash, sent_datetime_lower_bound=sent_datetime_lower_bound)
        for message in messages:
            if message.recipient_id == recipient_id and message.subject == email_subject and (message.html_body == email_body):
                return True
        return False

class BulkEmailModel(base_models.BaseModel):
    """Records the content of an email sent from Oppia to multiple users.

    This model is read-only; entries cannot be modified once created. The
    id/key of instances of this model is randomly generated string of
    length 12.

    The recipient IDs are not stored in this model. But, we store all
    bulk emails that are sent to a particular user in UserBulkEmailsModel.
    """
    sender_id = datastore_services.StringProperty(required=True, indexed=True)
    sender_email = datastore_services.StringProperty(required=True)
    intent = datastore_services.StringProperty(required=True, indexed=True, choices=[feconf.BULK_EMAIL_INTENT_MARKETING, feconf.BULK_EMAIL_INTENT_IMPROVE_EXPLORATION, feconf.BULK_EMAIL_INTENT_CREATE_EXPLORATION, feconf.BULK_EMAIL_INTENT_CREATOR_REENGAGEMENT, feconf.BULK_EMAIL_INTENT_LEARNER_REENGAGEMENT, feconf.EMAIL_INTENT_NOTIFY_CURRICULUM_ADMINS_CHAPTERS, feconf.BULK_EMAIL_INTENT_ML_JOB_FAILURE])
    subject = datastore_services.TextProperty(required=True)
    html_body = datastore_services.TextProperty(required=True)
    sent_datetime = datastore_services.DateTimeProperty(required=True, indexed=True)
    recipient_ids = datastore_services.JsonProperty(default=[], compressed=True)

    @staticmethod
    def get_deletion_policy() -> base_models.DELETION_POLICY:
        if False:
            return 10
        'Model contains data corresponding to a user: sender_id, and\n        sender_email.\n        '
        return base_models.DELETION_POLICY.DELETE

    @staticmethod
    def get_model_association_to_user() -> base_models.MODEL_ASSOCIATION_TO_USER:
        if False:
            while True:
                i = 10
        'Users already have access to this data since the emails were sent\n        to them.\n        '
        return base_models.MODEL_ASSOCIATION_TO_USER.NOT_CORRESPONDING_TO_USER

    @classmethod
    def get_export_policy(cls) -> Dict[str, base_models.EXPORT_POLICY]:
        if False:
            for i in range(10):
                print('nop')
        "Model contains data corresponding to a user, but this isn't exported\n        because users already have access to noteworthy details of this data\n        (since emails were sent to them).\n        "
        return dict(super(cls, cls).get_export_policy(), **{'sender_id': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'sender_email': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'recipient_ids': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'intent': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'subject': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'html_body': base_models.EXPORT_POLICY.NOT_APPLICABLE, 'sent_datetime': base_models.EXPORT_POLICY.NOT_APPLICABLE})

    @classmethod
    def apply_deletion_policy(cls, user_id: str) -> None:
        if False:
            return 10
        'Delete instances of BulkEmailModel for the user.\n\n        Args:\n            user_id: str. The ID of the user whose data should be deleted.\n        '
        keys = cls.query(datastore_services.any_of(cls.sender_id == user_id)).fetch(keys_only=True)
        datastore_services.delete_multi(keys)

    @classmethod
    def has_reference_to_user_id(cls, user_id: str) -> bool:
        if False:
            return 10
        'Check whether BulkEmailModel exists for user.\n\n        Args:\n            user_id: str. The ID of the user whose data should be checked.\n\n        Returns:\n            bool. Whether any models refer to the given user ID.\n        '
        return cls.query(cls.sender_id == user_id).get(keys_only=True) is not None

    @classmethod
    def create(cls, instance_id: str, sender_id: str, sender_email: str, intent: str, subject: str, html_body: str, sent_datetime: datetime.datetime) -> None:
        if False:
            i = 10
            return i + 15
        'Creates a new BulkEmailModel entry.\n\n        Args:\n            instance_id: str. The ID of the instance.\n            sender_id: str. The user ID of the email sender.\n            sender_email: str. The email address used to send the notification.\n            intent: str. The intent string, i.e. the purpose of the email.\n            subject: str. The subject line of the email.\n            html_body: str. The HTML content of the email body.\n            sent_datetime: datetime.datetime. The date and time the email\n                was sent, in UTC.\n        '
        email_model_instance = cls(id=instance_id, sender_id=sender_id, sender_email=sender_email, intent=intent, subject=subject, html_body=html_body, sent_datetime=sent_datetime)
        email_model_instance.update_timestamps()
        email_model_instance.put()