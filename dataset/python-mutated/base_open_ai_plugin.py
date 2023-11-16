"""Handles loading of plugins."""
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar
from auto_gpt_plugin_template import AutoGPTPluginTemplate
PromptGenerator = TypeVar('PromptGenerator')

class Message(TypedDict):
    role: str
    content: str

class BaseOpenAIPlugin(AutoGPTPluginTemplate):
    """
    This is a BaseOpenAIPlugin class for generating AutoGPT plugins.
    """

    def __init__(self, manifests_specs_clients: dict):
        if False:
            print('Hello World!')
        self._name = manifests_specs_clients['manifest']['name_for_model']
        self._version = manifests_specs_clients['manifest']['schema_version']
        self._description = manifests_specs_clients['manifest']['description_for_model']
        self._client = manifests_specs_clients['client']
        self._manifest = manifests_specs_clients['manifest']
        self._openapi_spec = manifests_specs_clients['openapi_spec']

    def can_handle_on_response(self) -> bool:
        if False:
            i = 10
            return i + 15
        'This method is called to check that the plugin can\n        handle the on_response method.\n        Returns:\n            bool: True if the plugin can handle the on_response method.'
        return False

    def on_response(self, response: str, *args, **kwargs) -> str:
        if False:
            while True:
                i = 10
        'This method is called when a response is received from the model.'
        return response

    def can_handle_post_prompt(self) -> bool:
        if False:
            print('Hello World!')
        'This method is called to check that the plugin can\n        handle the post_prompt method.\n        Returns:\n            bool: True if the plugin can handle the post_prompt method.'
        return False

    def post_prompt(self, prompt: PromptGenerator) -> PromptGenerator:
        if False:
            for i in range(10):
                print('nop')
        'This method is called just after the generate_prompt is called,\n            but actually before the prompt is generated.\n        Args:\n            prompt (PromptGenerator): The prompt generator.\n        Returns:\n            PromptGenerator: The prompt generator.\n        '
        return prompt

    def can_handle_on_planning(self) -> bool:
        if False:
            i = 10
            return i + 15
        'This method is called to check that the plugin can\n        handle the on_planning method.\n        Returns:\n            bool: True if the plugin can handle the on_planning method.'
        return False

    def on_planning(self, prompt: PromptGenerator, messages: List[Message]) -> Optional[str]:
        if False:
            print('Hello World!')
        'This method is called before the planning chat completion is done.\n        Args:\n            prompt (PromptGenerator): The prompt generator.\n            messages (List[str]): The list of messages.\n        '

    def can_handle_post_planning(self) -> bool:
        if False:
            return 10
        'This method is called to check that the plugin can\n        handle the post_planning method.\n        Returns:\n            bool: True if the plugin can handle the post_planning method.'
        return False

    def post_planning(self, response: str) -> str:
        if False:
            return 10
        'This method is called after the planning chat completion is done.\n        Args:\n            response (str): The response.\n        Returns:\n            str: The resulting response.\n        '
        return response

    def can_handle_pre_instruction(self) -> bool:
        if False:
            i = 10
            return i + 15
        'This method is called to check that the plugin can\n        handle the pre_instruction method.\n        Returns:\n            bool: True if the plugin can handle the pre_instruction method.'
        return False

    def pre_instruction(self, messages: List[Message]) -> List[Message]:
        if False:
            for i in range(10):
                print('nop')
        'This method is called before the instruction chat is done.\n        Args:\n            messages (List[Message]): The list of context messages.\n        Returns:\n            List[Message]: The resulting list of messages.\n        '
        return messages

    def can_handle_on_instruction(self) -> bool:
        if False:
            print('Hello World!')
        'This method is called to check that the plugin can\n        handle the on_instruction method.\n        Returns:\n            bool: True if the plugin can handle the on_instruction method.'
        return False

    def on_instruction(self, messages: List[Message]) -> Optional[str]:
        if False:
            print('Hello World!')
        'This method is called when the instruction chat is done.\n        Args:\n            messages (List[Message]): The list of context messages.\n        Returns:\n            Optional[str]: The resulting message.\n        '

    def can_handle_post_instruction(self) -> bool:
        if False:
            return 10
        'This method is called to check that the plugin can\n        handle the post_instruction method.\n        Returns:\n            bool: True if the plugin can handle the post_instruction method.'
        return False

    def post_instruction(self, response: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'This method is called after the instruction chat is done.\n        Args:\n            response (str): The response.\n        Returns:\n            str: The resulting response.\n        '
        return response

    def can_handle_pre_command(self) -> bool:
        if False:
            i = 10
            return i + 15
        'This method is called to check that the plugin can\n        handle the pre_command method.\n        Returns:\n            bool: True if the plugin can handle the pre_command method.'
        return False

    def pre_command(self, command_name: str, arguments: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        'This method is called before the command is executed.\n        Args:\n            command_name (str): The command name.\n            arguments (Dict[str, Any]): The arguments.\n        Returns:\n            Tuple[str, Dict[str, Any]]: The command name and the arguments.\n        '
        return (command_name, arguments)

    def can_handle_post_command(self) -> bool:
        if False:
            print('Hello World!')
        'This method is called to check that the plugin can\n        handle the post_command method.\n        Returns:\n            bool: True if the plugin can handle the post_command method.'
        return False

    def post_command(self, command_name: str, response: str) -> str:
        if False:
            i = 10
            return i + 15
        'This method is called after the command is executed.\n        Args:\n            command_name (str): The command name.\n            response (str): The response.\n        Returns:\n            str: The resulting response.\n        '
        return response

    def can_handle_chat_completion(self, messages: Dict[Any, Any], model: str, temperature: float, max_tokens: int) -> bool:
        if False:
            print('Hello World!')
        'This method is called to check that the plugin can\n          handle the chat_completion method.\n        Args:\n            messages (List[Message]): The messages.\n            model (str): The model name.\n            temperature (float): The temperature.\n            max_tokens (int): The max tokens.\n          Returns:\n              bool: True if the plugin can handle the chat_completion method.'
        return False

    def handle_chat_completion(self, messages: List[Message], model: str, temperature: float, max_tokens: int) -> str:
        if False:
            print('Hello World!')
        'This method is called when the chat completion is done.\n        Args:\n            messages (List[Message]): The messages.\n            model (str): The model name.\n            temperature (float): The temperature.\n            max_tokens (int): The max tokens.\n        Returns:\n            str: The resulting response.\n        '

    def can_handle_text_embedding(self, text: str) -> bool:
        if False:
            return 10
        'This method is called to check that the plugin can\n          handle the text_embedding method.\n\n        Args:\n            text (str): The text to be convert to embedding.\n        Returns:\n            bool: True if the plugin can handle the text_embedding method.'
        return False

    def handle_text_embedding(self, text: str) -> list[float]:
        if False:
            i = 10
            return i + 15
        'This method is called to create a text embedding.\n\n        Args:\n            text (str): The text to be convert to embedding.\n        Returns:\n            list[float]: The created embedding vector.\n        '

    def can_handle_user_input(self, user_input: str) -> bool:
        if False:
            print('Hello World!')
        'This method is called to check that the plugin can\n        handle the user_input method.\n\n        Args:\n            user_input (str): The user input.\n\n        Returns:\n            bool: True if the plugin can handle the user_input method.'
        return False

    def user_input(self, user_input: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'This method is called to request user input to the user.\n\n        Args:\n            user_input (str): The question or prompt to ask the user.\n\n        Returns:\n            str: The user input.\n        '

    def can_handle_report(self) -> bool:
        if False:
            while True:
                i = 10
        'This method is called to check that the plugin can\n        handle the report method.\n\n        Returns:\n            bool: True if the plugin can handle the report method.'
        return False

    def report(self, message: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'This method is called to report a message to the user.\n\n        Args:\n            message (str): The message to report.\n        '