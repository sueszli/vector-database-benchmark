from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, TypeVar
from xiaogpt.config import Config
T = TypeVar('T', bound='BaseBot')

class BaseBot(ABC):

    @abstractmethod
    async def ask(self, query: str, **options: Any) -> str:
        pass

    @abstractmethod
    async def ask_stream(self, query: str, **options: Any) -> AsyncGenerator[str, None]:
        pass

    @classmethod
    @abstractmethod
    def from_config(cls: type[T], config: Config) -> T:
        if False:
            return 10
        pass

    @abstractmethod
    def has_history(self) -> bool:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def change_prompt(self, new_prompt: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

class ChatHistoryMixin:
    history: list[tuple[str, str]]

    def has_history(self) -> bool:
        if False:
            return 10
        return bool(self.history)

    def change_prompt(self, new_prompt: str) -> None:
        if False:
            print('Hello World!')
        if self.history:
            print(self.history)
            self.history[0][0] = new_prompt

    def get_messages(self) -> list[dict]:
        if False:
            for i in range(10):
                print('nop')
        ms = []
        for h in self.history:
            ms.append({'role': 'user', 'content': h[0]})
            ms.append({'role': 'assistant', 'content': h[1]})
        return ms

    def add_message(self, query: str, message: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.history.append([f'{query}', message])
        first_history = self.history.pop(0)
        self.history = [first_history] + self.history[-5:]