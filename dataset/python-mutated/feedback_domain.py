"""Domain objects for feedback models."""
from __future__ import annotations
import datetime
from core import utils
from typing import Dict, List, Optional, TypedDict

class FeedbackThreadDict(TypedDict):
    """Dict for FeedbackThread object."""
    last_updated_msecs: float
    original_author_id: str
    state_name: Optional[str]
    status: str
    subject: str
    summary: str
    thread_id: str
    message_count: int
    last_nonempty_message_text: Optional[str]
    last_nonempty_message_author_id: Optional[str]

class FeedbackMessageDict(TypedDict):
    """Dict for FeedbackMessage object."""
    author_id: str
    created_on_msecs: float
    entity_type: str
    entity_id: str
    message_id: int
    text: str
    updated_status: str
    updated_subject: str

class FeedbackThreadSummaryDict(TypedDict):
    """Dict for FeedbackThreadSummary object."""
    status: str
    original_author_id: str
    last_updated_msecs: float
    last_message_text: str
    total_message_count: int
    last_message_is_read: bool
    second_last_message_is_read: bool
    author_last_message: str
    author_second_last_message: Optional[str]
    exploration_title: str
    exploration_id: str
    thread_id: str

class FeedbackThread:
    """Domain object for a feedback thread.

    Attributes:
        thread_id: str. The feedback thread ID.
        entity_type: str. The type of entity the feedback thread is linked to.
        entity_id: str. The id of the entity.
        state_name: str|None. The name of the state associated with
            the feedback thread or None, if no state is associated.
        original_author_id: str. The ID of the original author.
        status: str. The current status of the thread. Status should
            be one of core.storage.feedback.gae_models.STATUS_CHOICES.
        subject: str. The subject of the feedback thread.
        summary: str. A summary of the feedback thread.
        has_suggestion: bool. Whether the feedback thread includes a
            suggestion.
        message_count: int. The number of messages posted onto the thread.
        created_on: datetime.datetime. The time when the feedback thread was
            created.
        last_updated: datetime.datetime. The time when the feedback thread
            was last updated.
        last_nonempty_message_text: str|None. Cached text of the last message in
            the thread with non-empty content, or None if there is no such
            message.
        last_nonempty_message_author_id: str|None. Cached ID for the user of the
            last message in the thread with non-empty content, or None if the
            message was made anonymously or if there is no such message.
    """

    def __init__(self, thread_id: str, entity_type: str, entity_id: str, state_name: Optional[str], original_author_id: str, status: str, subject: str, summary: str, has_suggestion: bool, message_count: int, created_on: datetime.datetime, last_updated: datetime.datetime, last_nonempty_message_text: Optional[str]=None, last_nonempty_message_author_id: Optional[str]=None) -> None:
        if False:
            return 10
        'Initializes a FeedbackThread object.'
        self.id = thread_id
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.state_name = state_name
        self.original_author_id = original_author_id
        self.status = status
        self.subject = subject
        self.summary = summary
        self.has_suggestion = has_suggestion
        self.message_count = message_count
        self.created_on = created_on
        self.last_updated = last_updated
        self.last_nonempty_message_text = last_nonempty_message_text
        self.last_nonempty_message_author_id = last_nonempty_message_author_id

    def to_dict(self) -> FeedbackThreadDict:
        if False:
            return 10
        'Returns a dict representation of this FeedbackThread object.\n\n        Returns:\n            dict. A dict representation of the FeedbackThread object.\n        '
        return {'last_updated_msecs': utils.get_time_in_millisecs(self.last_updated), 'original_author_id': self.original_author_id, 'state_name': self.state_name, 'status': self.status, 'subject': self.subject, 'summary': self.summary, 'thread_id': self.id, 'message_count': self.message_count, 'last_nonempty_message_text': self.last_nonempty_message_text, 'last_nonempty_message_author_id': self.last_nonempty_message_author_id}

    def _get_full_message_id(self, message_id: int) -> str:
        if False:
            return 10
        'Returns the full id of the message.\n\n        Args:\n            message_id: int. The id of the message for which we have to fetch\n                the complete message id.\n\n        Returns:\n            str. The full id corresponding to the given message id.\n        '
        return '.'.join([self.id, str(message_id)])

    def get_last_two_message_ids(self) -> List[Optional[str]]:
        if False:
            return 10
        'Returns the full message ids of the last two messages of the thread.\n        If the thread has only one message, the id of the second last message is\n        None.\n\n        Returns:\n            list(str|None). The ids of the last two messages of the thread. If\n            the message does not exist, None is returned.\n        '
        return [self._get_full_message_id(i) if i >= 0 else None for i in range(self.message_count - 1, self.message_count - 3, -1)]

class FeedbackMessage:
    """Domain object for a feedback message.

    Attributes:
        full_message_id: str. The ID of the feedback message.
        thread_id: str. The ID of the feedback thread containing this
            message.
        message_id: int. The ID of the feedback thread message.
        author_id: str. The ID of the message's author.
        updated_status: str. The new status of the feedback thread.
        updated_subject: str. The new feedback thread subject.
        text: str. The text for the full feedback thread message.
        created_on: datetime.datetime. The time when the feedback message was
            created.
        last_updated: datetime.datetime. The time when the feedback message
            was last updated.
        received_via_email: bool. Whether the feedback was received via email.
    """

    def __init__(self, full_message_id: str, thread_id: str, message_id: int, author_id: str, updated_status: str, updated_subject: str, text: str, created_on: datetime.datetime, last_updated: datetime.datetime, received_via_email: bool) -> None:
        if False:
            i = 10
            return i + 15
        self.id = full_message_id
        self.thread_id = thread_id
        self.message_id = message_id
        self.author_id = author_id
        self.updated_status = updated_status
        self.updated_subject = updated_subject
        self.text = text
        self.created_on = created_on
        self.last_updated = last_updated
        self.received_via_email = received_via_email

    @property
    def entity_id(self) -> str:
        if False:
            while True:
                i = 10
        'Returns the entity ID corresponding to this FeedbackMessage instance.\n\n        Returns:\n            str. The entity_id.\n        '
        return self.id.split('.')[1]

    @property
    def entity_type(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns the entity type corresponding to this FeedbackMessage\n        instance.\n\n        Returns:\n            str. The entity_type.\n        '
        return self.id.split('.')[0]

    def to_dict(self) -> FeedbackMessageDict:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representation of this FeedbackMessage object.\n\n        Returns:\n            dict. Dict representation of the FeedbackMessage object.\n        '
        return {'author_id': self.author_id, 'created_on_msecs': utils.get_time_in_millisecs(self.created_on), 'entity_type': self.entity_type, 'entity_id': self.entity_id, 'message_id': self.message_id, 'text': self.text, 'updated_status': self.updated_status, 'updated_subject': self.updated_subject}

class FullyQualifiedMessageIdentifier:
    """Domain object representing the full identifier of a message in a
    feedback thread.

    Attributes:
        thread_id: str. The ID of the thread.
        message_id: int. The ID of a message beloning to the thread.
    """

    def __init__(self, thread_id: str, message_id: int) -> None:
        if False:
            print('Hello World!')
        self.thread_id = thread_id
        self.message_id = message_id

class FeedbackAnalytics:
    """Domain object representing feedback analytics for a specific entity.

    Attributes:
        entity_type: str. The type of entity the feedback thread is linked to.
        entity_id: str. The id of the entity.
        num_open_threads: int. The number of open threads associated with the
            entity.
        num_total_threads: int. The total number of threads associated with the
            entity (regardless of status).
    """

    def __init__(self, entity_type: str, entity_id: str, num_open_threads: int, num_total_threads: int) -> None:
        if False:
            while True:
                i = 10
        'Initializes a FeedbackAnalytics object.'
        self.id = entity_id
        self.entity_type = entity_type
        self.num_open_threads = num_open_threads
        self.num_total_threads = num_total_threads

    def to_dict(self) -> Dict[str, int]:
        if False:
            while True:
                i = 10
        'Returns the numbers of threads in the FeedbackAnalytics object.\n\n        Attributes:\n            dict. Dict representation of the numbers of threads in the\n                FeedbackAnalytics object.\n        '
        return {'num_open_threads': self.num_open_threads, 'num_total_threads': self.num_total_threads}

class FeedbackMessageReferenceDict(TypedDict):
    """Dict for FeedbackMessageReference object."""
    entity_type: str
    entity_id: str
    thread_id: str
    message_id: int

class FeedbackMessageReference:
    """Domain object for feedback message references.

    Attributes:
        entity_type: str. The type of entity the feedback thread is linked to.
        entity_id: str. The id of the entity.
        thread_id: str. The ID of the feedback thread.
        message_id: int. The ID of the feedback thread message.
    """

    def __init__(self, entity_type: str, entity_id: str, thread_id: str, message_id: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initializes FeedbackMessageReference object.'
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.thread_id = thread_id
        self.message_id = message_id

    def to_dict(self) -> FeedbackMessageReferenceDict:
        if False:
            return 10
        'Returns dict representation of the FeedbackMessageReference object.\n\n        Returns:\n            dict. Dict representation of the FeedbackMessageReference object.\n        '
        return {'entity_type': self.entity_type, 'entity_id': self.entity_id, 'thread_id': self.thread_id, 'message_id': self.message_id}

class FeedbackThreadSummary:
    """Domain object for the summary of a particular thread.

    Attributes:
        status: str. The status of the thread.
        original_author_id: str. The id of the original author of the thread.
        last_updated: datetime.datetime. When was the thread last updated.
        last_message_text: str. The text of the last message.
        total_message_count: int. The total number of messages in the thread.
        last_message_is_read: bool. Whether the last message is read by the
            user.
        second_last_message_is_read: bool. Whether the second last message is
            read by the user,
        author_last_message: str. The name of the author of the last message.
        author_second_last_message: str. The name of the author of the second
            last message and None if no second-to-last message exists.
        exploration_title: str. The title of the exploration to which
            exploration belongs.
        exploration_id: str. The id of the exploration associated to the thread.
        thread_id: str. The id of the thread this dict is describing.
    """

    def __init__(self, status: str, original_author_id: str, last_updated: datetime.datetime, last_message_text: str, total_message_count: int, last_message_is_read: bool, second_last_message_is_read: bool, author_last_message: str, author_second_last_message: Optional[str], exploration_title: str, exploration_id: str, thread_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.status = status
        self.original_author_id = original_author_id
        self.last_updated = last_updated
        self.last_message_text = last_message_text
        self.total_message_count = total_message_count
        self.last_message_is_read = last_message_is_read
        self.second_last_message_is_read = second_last_message_is_read
        self.author_last_message = author_last_message
        self.author_second_last_message = author_second_last_message
        self.exploration_title = exploration_title
        self.exploration_id = exploration_id
        self.thread_id = thread_id

    def to_dict(self) -> FeedbackThreadSummaryDict:
        if False:
            print('Hello World!')
        'Returns dict representation of the FeedbackThreadSummary object.\n\n        Returns:\n            dict. Dict representation of the FeedbackThreadSummary object.\n        '
        return {'status': self.status, 'original_author_id': self.original_author_id, 'last_updated_msecs': utils.get_time_in_millisecs(self.last_updated), 'last_message_text': self.last_message_text, 'total_message_count': self.total_message_count, 'last_message_is_read': self.last_message_is_read, 'second_last_message_is_read': self.second_last_message_is_read, 'author_last_message': self.author_last_message, 'author_second_last_message': self.author_second_last_message, 'exploration_title': self.exploration_title, 'exploration_id': self.exploration_id, 'thread_id': self.thread_id}