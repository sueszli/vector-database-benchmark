from typing import Iterable, List, Optional, Sequence, Union, cast
from django.utils.translation import gettext as _
from zerver.lib.exceptions import JsonableError
from zerver.lib.string_validation import check_stream_topic
from zerver.models import Realm, Stream, UserProfile, get_user_by_id_in_realm_including_cross_realm, get_user_including_cross_realm

def get_user_profiles(emails: Iterable[str], realm: Realm) -> List[UserProfile]:
    if False:
        i = 10
        return i + 15
    user_profiles: List[UserProfile] = []
    for email in emails:
        try:
            user_profile = get_user_including_cross_realm(email, realm)
        except UserProfile.DoesNotExist:
            raise JsonableError(_("Invalid email '{email}'").format(email=email))
        user_profiles.append(user_profile)
    return user_profiles

def get_user_profiles_by_ids(user_ids: Iterable[int], realm: Realm) -> List[UserProfile]:
    if False:
        print('Hello World!')
    user_profiles: List[UserProfile] = []
    for user_id in user_ids:
        try:
            user_profile = get_user_by_id_in_realm_including_cross_realm(user_id, realm)
        except UserProfile.DoesNotExist:
            raise JsonableError(_('Invalid user ID {user_id}').format(user_id=user_id))
        user_profiles.append(user_profile)
    return user_profiles

class Addressee:

    def __init__(self, msg_type: str, user_profiles: Optional[Sequence[UserProfile]]=None, stream: Optional[Stream]=None, stream_name: Optional[str]=None, stream_id: Optional[int]=None, topic: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        assert msg_type in ['stream', 'private']
        if msg_type == 'stream' and topic is None:
            raise JsonableError(_('Missing topic'))
        self._msg_type = msg_type
        self._user_profiles = user_profiles
        self._stream = stream
        self._stream_name = stream_name
        self._stream_id = stream_id
        self._topic = topic

    def is_stream(self) -> bool:
        if False:
            while True:
                i = 10
        return self._msg_type == 'stream'

    def is_private(self) -> bool:
        if False:
            while True:
                i = 10
        return self._msg_type == 'private'

    def user_profiles(self) -> Sequence[UserProfile]:
        if False:
            return 10
        assert self.is_private()
        assert self._user_profiles is not None
        return self._user_profiles

    def stream(self) -> Optional[Stream]:
        if False:
            i = 10
            return i + 15
        assert self.is_stream()
        return self._stream

    def stream_name(self) -> Optional[str]:
        if False:
            print('Hello World!')
        assert self.is_stream()
        return self._stream_name

    def stream_id(self) -> Optional[int]:
        if False:
            return 10
        assert self.is_stream()
        return self._stream_id

    def topic(self) -> str:
        if False:
            i = 10
            return i + 15
        assert self.is_stream()
        assert self._topic is not None
        return self._topic

    @staticmethod
    def legacy_build(sender: UserProfile, recipient_type_name: str, message_to: Union[Sequence[int], Sequence[str]], topic_name: Optional[str], realm: Optional[Realm]=None) -> 'Addressee':
        if False:
            for i in range(10):
                print('nop')
        if realm is None:
            realm = sender.realm
        if recipient_type_name == 'stream':
            if len(message_to) > 1:
                raise JsonableError(_('Cannot send to multiple streams'))
            if message_to:
                stream_name_or_id = message_to[0]
            elif sender.default_sending_stream_id:
                stream_name_or_id = sender.default_sending_stream_id
            else:
                raise JsonableError(_('Missing stream'))
            if topic_name is None:
                raise JsonableError(_('Missing topic'))
            if isinstance(stream_name_or_id, int):
                return Addressee.for_stream_id(stream_name_or_id, topic_name)
            return Addressee.for_stream_name(stream_name_or_id, topic_name)
        elif recipient_type_name == 'private':
            if not message_to:
                raise JsonableError(_('Message must have recipients'))
            if isinstance(message_to[0], str):
                emails = cast(Sequence[str], message_to)
                return Addressee.for_private(emails, realm)
            elif isinstance(message_to[0], int):
                user_ids = cast(Sequence[int], message_to)
                return Addressee.for_user_ids(user_ids=user_ids, realm=realm)
        else:
            raise JsonableError(_('Invalid message type'))

    @staticmethod
    def for_stream(stream: Stream, topic: str) -> 'Addressee':
        if False:
            print('Hello World!')
        topic = topic.strip()
        check_stream_topic(topic)
        return Addressee(msg_type='stream', stream=stream, topic=topic)

    @staticmethod
    def for_stream_name(stream_name: str, topic: str) -> 'Addressee':
        if False:
            print('Hello World!')
        topic = topic.strip()
        check_stream_topic(topic)
        return Addressee(msg_type='stream', stream_name=stream_name, topic=topic)

    @staticmethod
    def for_stream_id(stream_id: int, topic: str) -> 'Addressee':
        if False:
            for i in range(10):
                print('nop')
        topic = topic.strip()
        check_stream_topic(topic)
        return Addressee(msg_type='stream', stream_id=stream_id, topic=topic)

    @staticmethod
    def for_private(emails: Sequence[str], realm: Realm) -> 'Addressee':
        if False:
            for i in range(10):
                print('nop')
        assert len(emails) > 0
        user_profiles = get_user_profiles(emails, realm)
        return Addressee(msg_type='private', user_profiles=user_profiles)

    @staticmethod
    def for_user_ids(user_ids: Sequence[int], realm: Realm) -> 'Addressee':
        if False:
            for i in range(10):
                print('nop')
        assert len(user_ids) > 0
        user_profiles = get_user_profiles_by_ids(user_ids, realm)
        return Addressee(msg_type='private', user_profiles=user_profiles)

    @staticmethod
    def for_user_profile(user_profile: UserProfile) -> 'Addressee':
        if False:
            for i in range(10):
                print('nop')
        user_profiles = [user_profile]
        return Addressee(msg_type='private', user_profiles=user_profiles)