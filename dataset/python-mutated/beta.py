from __future__ import annotations
from typing import TYPE_CHECKING
from .threads import Threads, AsyncThreads, ThreadsWithRawResponse, AsyncThreadsWithRawResponse
from .assistants import Assistants, AsyncAssistants, AssistantsWithRawResponse, AsyncAssistantsWithRawResponse
from ..._resource import SyncAPIResource, AsyncAPIResource
if TYPE_CHECKING:
    from ..._client import OpenAI, AsyncOpenAI
__all__ = ['Beta', 'AsyncBeta']

class Beta(SyncAPIResource):
    assistants: Assistants
    threads: Threads
    with_raw_response: BetaWithRawResponse

    def __init__(self, client: OpenAI) -> None:
        if False:
            while True:
                i = 10
        super().__init__(client)
        self.assistants = Assistants(client)
        self.threads = Threads(client)
        self.with_raw_response = BetaWithRawResponse(self)

class AsyncBeta(AsyncAPIResource):
    assistants: AsyncAssistants
    threads: AsyncThreads
    with_raw_response: AsyncBetaWithRawResponse

    def __init__(self, client: AsyncOpenAI) -> None:
        if False:
            return 10
        super().__init__(client)
        self.assistants = AsyncAssistants(client)
        self.threads = AsyncThreads(client)
        self.with_raw_response = AsyncBetaWithRawResponse(self)

class BetaWithRawResponse:

    def __init__(self, beta: Beta) -> None:
        if False:
            i = 10
            return i + 15
        self.assistants = AssistantsWithRawResponse(beta.assistants)
        self.threads = ThreadsWithRawResponse(beta.threads)

class AsyncBetaWithRawResponse:

    def __init__(self, beta: AsyncBeta) -> None:
        if False:
            while True:
                i = 10
        self.assistants = AsyncAssistantsWithRawResponse(beta.assistants)
        self.threads = AsyncThreadsWithRawResponse(beta.threads)