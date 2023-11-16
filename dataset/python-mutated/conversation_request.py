from typing import Dict
from .conversation_writer import ConversationWriter

class ConversationRequest:

    def __init__(self, template: str, instantiation: Dict[str, str], writer: ConversationWriter=None):
        if False:
            while True:
                i = 10
        self._template = template
        self._instantiation = instantiation
        self._writer = writer

    @property
    def template(self) -> str:
        if False:
            while True:
                i = 10
        return self._template

    @property
    def instantiation_parameters(self) -> Dict[str, str]:
        if False:
            print('Hello World!')
        return self._instantiation

    @property
    def writer(self) -> ConversationWriter:
        if False:
            return 10
        return self._writer