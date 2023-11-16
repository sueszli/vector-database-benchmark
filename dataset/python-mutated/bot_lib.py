import importlib
import json
from typing import Any, Callable, Dict, Optional
from django.conf import settings
from django.utils.translation import gettext as _
from zulip_bots.lib import BotIdentity, RateLimit
from zerver.actions.message_send import internal_send_huddle_message, internal_send_private_message, internal_send_stream_message_by_name
from zerver.lib.bot_config import ConfigError, get_bot_config
from zerver.lib.bot_storage import get_bot_storage, is_key_in_bot_storage, remove_bot_storage, set_bot_storage
from zerver.lib.integrations import EMBEDDED_BOTS
from zerver.lib.topic import get_topic_from_message_info
from zerver.models import UserProfile, get_active_user

def get_bot_handler(service_name: str) -> Any:
    if False:
        i = 10
        return i + 15
    configured_service = ''
    for embedded_bot_service in EMBEDDED_BOTS:
        if service_name == embedded_bot_service.name:
            configured_service = embedded_bot_service.name
    if not configured_service:
        return None
    bot_module_name = f'zulip_bots.bots.{configured_service}.{configured_service}'
    bot_module: Any = importlib.import_module(bot_module_name)
    return bot_module.handler_class()

class StateHandler:
    storage_size_limit: int = settings.USER_STATE_SIZE_LIMIT

    def __init__(self, user_profile: UserProfile) -> None:
        if False:
            while True:
                i = 10
        self.user_profile = user_profile
        self.marshal: Callable[[object], str] = lambda obj: json.dumps(obj)
        self.demarshal: Callable[[str], object] = lambda obj: json.loads(obj)

    def get(self, key: str) -> object:
        if False:
            while True:
                i = 10
        return self.demarshal(get_bot_storage(self.user_profile, key))

    def put(self, key: str, value: object) -> None:
        if False:
            while True:
                i = 10
        set_bot_storage(self.user_profile, [(key, self.marshal(value))])

    def remove(self, key: str) -> None:
        if False:
            print('Hello World!')
        remove_bot_storage(self.user_profile, [key])

    def contains(self, key: str) -> bool:
        if False:
            while True:
                i = 10
        return is_key_in_bot_storage(self.user_profile, key)

class EmbeddedBotQuitError(Exception):
    pass

class EmbeddedBotEmptyRecipientsListError(Exception):
    pass

class EmbeddedBotHandler:

    def __init__(self, user_profile: UserProfile) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.user_profile = user_profile
        self._rate_limit = RateLimit(20, 5)
        self.full_name = user_profile.full_name
        self.email = user_profile.email
        self.storage = StateHandler(user_profile)
        self.user_id = user_profile.id

    def identity(self) -> BotIdentity:
        if False:
            i = 10
            return i + 15
        return BotIdentity(self.full_name, self.email)

    def react(self, message: Dict[str, Any], emoji_name: str) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return {}

    def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        if not self._rate_limit.is_legal():
            self._rate_limit.show_error_and_exit()
        if message['type'] == 'stream':
            message_id = internal_send_stream_message_by_name(self.user_profile.realm, self.user_profile, message['to'], message['topic'], message['content'])
            return {'id': message_id}
        assert message['type'] == 'private'
        recipients = ','.join(message['to']).split(',')
        if len(message['to']) == 0:
            raise EmbeddedBotEmptyRecipientsListError(_('Message must have recipients!'))
        elif len(message['to']) == 1:
            recipient_user = get_active_user(recipients[0], self.user_profile.realm)
            message_id = internal_send_private_message(self.user_profile, recipient_user, message['content'])
        else:
            message_id = internal_send_huddle_message(self.user_profile.realm, self.user_profile, recipients, message['content'])
        return {'id': message_id}

    def send_reply(self, message: Dict[str, Any], response: str, widget_content: Optional[str]=None) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        if message['type'] == 'private':
            result = self.send_message(dict(type='private', to=[x['email'] for x in message['display_recipient']], content=response, sender_email=message['sender_email']))
        else:
            result = self.send_message(dict(type='stream', to=message['display_recipient'], topic=get_topic_from_message_info(message), content=response, sender_email=message['sender_email']))
        return {'id': result['id']}

    def update_message(self, message: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        pass

    def get_config_info(self, bot_name: str, optional: bool=False) -> Dict[str, str]:
        if False:
            i = 10
            return i + 15
        try:
            return get_bot_config(self.user_profile)
        except ConfigError:
            if optional:
                return {}
            raise

    def quit(self, message: str='') -> None:
        if False:
            for i in range(10):
                print('nop')
        raise EmbeddedBotQuitError(message)