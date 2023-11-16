from pathlib import Path
from typing import Dict, List
from ..config import cfg
from ..role import SystemRole
from .handler import Handler
CHAT_CACHE_LENGTH = int(cfg.get('CHAT_CACHE_LENGTH'))
CHAT_CACHE_PATH = Path(cfg.get('CHAT_CACHE_PATH'))

class DefaultHandler(Handler):

    def __init__(self, role: SystemRole) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(role)
        self.role = role

    def make_prompt(self, prompt: str) -> str:
        if False:
            i = 10
            return i + 15
        prompt = prompt.strip()
        return self.role.make_prompt(prompt, initial=True)

    def make_messages(self, prompt: str) -> List[Dict[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        messages = []
        if cfg.get('SYSTEM_ROLES') == 'true':
            messages.append({'role': 'system', 'content': self.role.role})
        messages.append({'role': 'user', 'content': prompt})
        return messages