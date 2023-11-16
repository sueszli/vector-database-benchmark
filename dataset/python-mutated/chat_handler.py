import json
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional
import typer
from click import BadArgumentUsage
from ..config import cfg
from ..role import SystemRole
from .handler import Handler
CHAT_CACHE_LENGTH = int(cfg.get('CHAT_CACHE_LENGTH'))
CHAT_CACHE_PATH = Path(cfg.get('CHAT_CACHE_PATH'))

class ChatSession:
    """
    This class is used as a decorator for OpenAI chat API requests.
    The ChatSession class caches chat messages and keeps track of the
    conversation history. It is designed to store cached messages
    in a specified directory and in JSON format.
    """

    def __init__(self, length: int, storage_path: Path):
        if False:
            return 10
        '\n        Initialize the ChatSession decorator.\n\n        :param length: Integer, maximum number of cached messages to keep.\n        '
        self.length = length
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        if False:
            i = 10
            return i + 15
        '\n        The Cache decorator.\n\n        :param func: The chat function to cache.\n        :return: Wrapped function with chat caching.\n        '

        def wrapper(*args: Any, **kwargs: Any) -> Generator[str, None, None]:
            if False:
                i = 10
                return i + 15
            chat_id = kwargs.pop('chat_id', None)
            messages = kwargs['messages']
            if not chat_id:
                yield from func(*args, **kwargs)
                return
            old_messages = self._read(chat_id)
            for message in messages:
                old_messages.append(message)
            kwargs['messages'] = old_messages
            response_text = ''
            for word in func(*args, **kwargs):
                response_text += word
                yield word
            old_messages.append({'role': 'assistant', 'content': response_text})
            self._write(kwargs['messages'], chat_id)
        return wrapper

    def _read(self, chat_id: str) -> List[Dict[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        file_path = self.storage_path / chat_id
        if not file_path.exists():
            return []
        parsed_cache = json.loads(file_path.read_text())
        return parsed_cache if isinstance(parsed_cache, list) else []

    def _write(self, messages: List[Dict[str, str]], chat_id: str) -> None:
        if False:
            i = 10
            return i + 15
        file_path = self.storage_path / chat_id
        json.dump(messages[-self.length:], file_path.open('w'))

    def invalidate(self, chat_id: str) -> None:
        if False:
            i = 10
            return i + 15
        file_path = self.storage_path / chat_id
        file_path.unlink(missing_ok=True)

    def get_messages(self, chat_id: str) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        messages = self._read(chat_id)
        return [f"{message['role']}: {message['content']}" for message in messages]

    def exists(self, chat_id: Optional[str]) -> bool:
        if False:
            i = 10
            return i + 15
        return bool(chat_id and bool(self._read(chat_id)))

    def list(self) -> List[Path]:
        if False:
            for i in range(10):
                print('nop')
        files = self.storage_path.glob('*')
        return sorted(files, key=lambda f: f.stat().st_mtime)

class ChatHandler(Handler):
    chat_session = ChatSession(CHAT_CACHE_LENGTH, CHAT_CACHE_PATH)

    def __init__(self, chat_id: str, role: SystemRole) -> None:
        if False:
            print('Hello World!')
        super().__init__(role)
        self.chat_id = chat_id
        self.role = role
        if chat_id == 'temp':
            self.chat_session.invalidate(chat_id)
        self.validate()

    @classmethod
    def list_ids(cls, value: str) -> None:
        if False:
            return 10
        if not value:
            return
        for chat_id in cls.chat_session.list():
            typer.echo(chat_id)
        raise typer.Exit()

    @property
    def initiated(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.chat_session.exists(self.chat_id)

    @property
    def initial_message(self) -> str:
        if False:
            print('Hello World!')
        chat_history = self.chat_session.get_messages(self.chat_id)
        index = 1 if cfg.get('SYSTEM_ROLES') == 'true' else 0
        return chat_history[index] if chat_history else ''

    @property
    def is_same_role(self) -> bool:
        if False:
            while True:
                i = 10
        return self.role.same_role(self.initial_message)

    @classmethod
    def show_messages_callback(cls, chat_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not chat_id:
            return
        cls.show_messages(chat_id)
        raise typer.Exit()

    @classmethod
    def show_messages(cls, chat_id: str) -> None:
        if False:
            return 10
        for (index, message) in enumerate(cls.chat_session.get_messages(chat_id)):
            if message.startswith('user:'):
                message = '\n'.join(message.splitlines()[:-1])
            color = 'magenta' if index % 2 == 0 else 'green'
            typer.secho(message, fg=color)

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.initiated:
            chat_role_name = self.role.get_role_name(self.initial_message)
            if not chat_role_name:
                raise BadArgumentUsage(f'Could not determine chat role of "{self.chat_id}"')
            if self.role.name == 'default':
                self.role = SystemRole.get(chat_role_name)
            elif not self.is_same_role:
                raise BadArgumentUsage(f'Cant change chat role to "{self.role.name}" since it was initiated as "{chat_role_name}" chat.')

    def make_prompt(self, prompt: str) -> str:
        if False:
            i = 10
            return i + 15
        prompt = prompt.strip()
        return self.role.make_prompt(prompt, not self.initiated)

    def make_messages(self, prompt: str) -> List[Dict[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        messages = []
        if not self.initiated and cfg.get('SYSTEM_ROLES') == 'true':
            messages.append({'role': 'system', 'content': self.role.role})
        messages.append({'role': 'user', 'content': prompt})
        return messages

    @chat_session
    def get_completion(self, **kwargs: Any) -> Generator[str, None, None]:
        if False:
            i = 10
            return i + 15
        yield from super().get_completion(**kwargs)