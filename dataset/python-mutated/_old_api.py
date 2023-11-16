from __future__ import annotations
from typing import TYPE_CHECKING
from typing_extensions import override
from .._utils import LazyProxy
from .._exceptions import OpenAIError
INSTRUCTIONS = '\n\nYou tried to access openai.{symbol}, but this is no longer supported in openai>=1.0.0 - see the README at https://github.com/openai/openai-python for the API.\n\nYou can run `openai migrate` to automatically upgrade your codebase to use the 1.0.0 interface. \n\nAlternatively, you can pin your installation to the old version, e.g. `pip install openai==0.28`\n\nA detailed migration guide is available here: https://github.com/openai/openai-python/discussions/742\n'

class APIRemovedInV1(OpenAIError):

    def __init__(self, *, symbol: str) -> None:
        if False:
            return 10
        super().__init__(INSTRUCTIONS.format(symbol=symbol))

class APIRemovedInV1Proxy(LazyProxy[None]):

    def __init__(self, *, symbol: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._symbol = symbol

    @override
    def __load__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise APIRemovedInV1(symbol=self._symbol)
SYMBOLS = ['Edit', 'File', 'Audio', 'Image', 'Model', 'Engine', 'Customer', 'FineTune', 'Embedding', 'Completion', 'Deployment', 'Moderation', 'ErrorObject', 'FineTuningJob', 'ChatCompletion']
if TYPE_CHECKING:
    __all__: list[str] = []
else:
    __all__ = SYMBOLS
__locals = locals()
for symbol in SYMBOLS:
    __locals[symbol] = APIRemovedInV1Proxy(symbol=symbol)