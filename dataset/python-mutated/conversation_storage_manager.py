import threading
from langchain.chat_models.openai import ChatOpenAI
from lwe.core.logger import Logger
from lwe.core import constants
import lwe.core.util as util
from lwe.core.function_cache import FunctionCache
from lwe.core.token_manager import TokenManager
from lwe.backends.api.orm import Orm
from lwe.backends.api.conversation import ConversationManager
from lwe.backends.api.message import MessageManager
from lwe.backends.api.request import ApiRequest
from langchain.adapters.openai import convert_dict_to_message

class ConversationStorageManager:
    """Manage conversation storage."""

    def __init__(self, config, function_manager, current_user=None, conversation_id=None, provider=None, model_name=None, preset_name=None, provider_manager=None, orm=None):
        if False:
            for i in range(10):
                print('nop')
        self.config = config
        self.log = Logger(self.__class__.__name__, self.config)
        self.function_manager = function_manager
        self.current_user = current_user
        self.conversation_id = conversation_id
        self.provider = provider
        self.model_name = model_name or constants.API_BACKEND_DEFAULT_MODEL
        self.preset_name = preset_name or ''
        self.provider_manager = provider_manager
        self.function_cache = FunctionCache(self.config, self.function_manager)
        self.token_manager = TokenManager(self.config, self.provider, self.model_name, self.function_cache)
        self.orm = orm or Orm(self.config)
        self.conversation = ConversationManager(config, self.orm)
        self.message = MessageManager(config, self.orm)

    def store_conversation_messages(self, new_messages, response_content=None, title=None):
        if False:
            return 10
        '\n        Store conversation messages.\n\n        :param new_messages: New messages\n        :type new_messages: list\n        :param response_content: Response content\n        :type response_content: str\n        :param title: Title\n        :type title: str, optional\n        :returns: success, conversation or response_content, message\n        :rtype: tuple\n        '
        self.log.debug(f"Storing conversation messages for conversation: {self.conversation_id or 'new'}")
        if self.current_user:
            (success, response, user_message) = self.add_new_messages_to_conversation(new_messages, title)
            if not success:
                return (success, response, user_message)
            (conversation, last_message) = response
            if conversation.title:
                self.log.debug(f'Conversation {conversation.id} already has title: {conversation.title}')
            else:
                self.gen_title(conversation)
            return (True, conversation, 'Conversation updated with new messages')
        else:
            return (True, response_content, 'No current user, conversation not saved')

    def create_new_conversation_if_needed(self, title=None):
        if False:
            while True:
                i = 10
        "\n        Create new conversation if it doesn't exist.\n\n        :param title: Conversation title, defaults to None\n        :type title: str, optional\n        :returns: Conversation object\n        :rtype: Conversation\n        "
        if self.conversation_id:
            (success, conversation, message) = self.conversation.get_conversation(self.conversation_id)
            if not success:
                raise Exception(message)
        else:
            (success, conversation, message) = self.conversation.add_conversation(self.current_user.id, title=title)
            self.conversation_id = conversation.id
            if not success:
                raise Exception(message)
        return conversation

    def add_new_messages_to_conversation(self, new_messages, title=None):
        if False:
            i = 10
            return i + 15
        'Add new messages to a conversation.\n\n        :param new_messages: New messages\n        :type new_messages: list\n        :param title: Conversation title, defaults to None\n        :type title: str, optional\n        :returns: Conversation, last message\n        :rtype: tuple\n        '
        conversation = self.create_new_conversation_if_needed(title)
        last_message = None
        for m in new_messages:
            (success, last_message, user_message) = self.add_message(m['role'], m['message'], m['message_type'], m['message_metadata'])
            if not success:
                raise Exception(user_message)
        return (True, (conversation, last_message), f'Added new messages to conversation {conversation.id}')

    def add_message(self, role, message, message_type, metadata):
        if False:
            i = 10
            return i + 15
        '\n        Add a new message to a conversation.\n\n        :param role: Message role\n        :type role: str\n        :param message: Message content\n        :type message: str\n        :param message_type: Message type\n        :type message_type: str\n        :param metadata: Message metadata\n        :type metadata: dict\n        :returns: success, added message, user message\n        :rtype: tuple\n        '
        return self.message.add_message(self.conversation_id, role, message, message_type, metadata, self.provider.name, self.model_name, self.preset_name)

    def gen_title_thread(self, conversation_id):
        if False:
            print('Hello World!')
        '\n        Generate the title for a conversation in a separate thread.\n\n        :param conversation_id: Conversation ID\n        :type conversation_id: int\n        '
        self.log.info(f'Generating title for conversation {conversation_id}')
        message_manager = MessageManager(self.config, self.orm)
        conversation_manager = ConversationManager(self.config, self.orm)
        (success, messages, user_message) = message_manager.get_messages(conversation_id, limit=2)
        if not success:
            self.log.warning(f'Failed to generate title for conversation: {user_message}')
            return
        user_content = messages[1]['message'][:constants.TITLE_GENERATION_MAX_CHARACTERS]
        new_messages = [self.message.build_message('system', constants.DEFAULT_TITLE_GENERATION_SYSTEM_PROMPT), self.message.build_message('user', '%s: %s' % (constants.DEFAULT_TITLE_GENERATION_USER_PROMPT, user_content))]
        new_messages = util.transform_messages_to_chat_messages(new_messages)
        new_messages = [convert_dict_to_message(m) for m in new_messages]
        title_provider_name = self.config.get('backend_options.title_generation.provider')
        if title_provider_name:
            provider = self.provider_manager.get_provider_from_name(title_provider_name)
            if not provider:
                raise RuntimeError(f'Failed to load title provider: {title_provider_name}')
            llm = provider.make_llm()
        else:
            llm = ChatOpenAI(model_name=constants.API_BACKEND_DEFAULT_MODEL, temperature=0)
        try:
            result = llm(new_messages)
            request = ApiRequest(orm=self.orm)
            title = request.extract_message_content(result)['message']
            title = title.replace('\n', ', ').strip().strip('\'"')
            self.log.info(f'Title generated for conversation {conversation_id}: {title}')
            (success, conversation, user_message) = conversation_manager.edit_conversation_title(conversation_id, title)
            if success:
                self.log.debug(f'Title saved for conversation {conversation_id}')
        except ValueError as e:
            self.log.warning(f'Failed to generate title for conversation: {str(e)}')

    def gen_title(self, conversation):
        if False:
            return 10
        '\n        Generate the title for a conversation.\n\n        :param conversation: Conversation\n        :type conversation: Conversation\n        '
        conversation_id = conversation.id
        database = self.config.get('database')
        if database.startswith('sqlite') and ':memory:' in database:
            self.gen_title_thread(conversation_id)
        else:
            thread = threading.Thread(target=self.gen_title_thread, args=(conversation_id,))
            thread.start()

    def get_conversation_token_count(self):
        if False:
            while True:
                i = 10
        'Get token count for conversation.\n\n        :returns: Number of tokens\n        :rtype: int\n        '
        (success, old_messages, user_message) = self.message.get_messages(self.conversation_id)
        if not success:
            raise Exception(user_message)
        tokens = self.token_manager.get_num_tokens_from_messages(old_messages)
        return tokens