import logging
from typing import Callable, Any, Tuple, Generic, TypeVar, Generator, Awaitable
from ..exceptions import AzureError
from ._poller import _SansIONoPolling
PollingReturnType_co = TypeVar('PollingReturnType_co', covariant=True)
DeserializationCallbackType = Any
_LOGGER = logging.getLogger(__name__)

class AsyncPollingMethod(Generic[PollingReturnType_co]):
    """ABC class for polling method."""

    def initialize(self, client: Any, initial_response: Any, deserialization_callback: DeserializationCallbackType) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError('This method needs to be implemented')

    async def run(self) -> None:
        raise NotImplementedError('This method needs to be implemented')

    def status(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('This method needs to be implemented')

    def finished(self) -> bool:
        if False:
            return 10
        raise NotImplementedError('This method needs to be implemented')

    def resource(self) -> PollingReturnType_co:
        if False:
            return 10
        raise NotImplementedError('This method needs to be implemented')

    def get_continuation_token(self) -> str:
        if False:
            print('Hello World!')
        raise TypeError("Polling method '{}' doesn't support get_continuation_token".format(self.__class__.__name__))

    @classmethod
    def from_continuation_token(cls, continuation_token: str, **kwargs: Any) -> Tuple[Any, Any, DeserializationCallbackType]:
        if False:
            for i in range(10):
                print('nop')
        raise TypeError("Polling method '{}' doesn't support from_continuation_token".format(cls.__name__))

class AsyncNoPolling(_SansIONoPolling[PollingReturnType_co], AsyncPollingMethod[PollingReturnType_co]):
    """An empty async poller that returns the deserialized initial response."""

    async def run(self) -> None:
        """Empty run, no polling.
        Just override initial run to add "async"
        """

async def async_poller(client: Any, initial_response: Any, deserialization_callback: Callable[[Any], PollingReturnType_co], polling_method: AsyncPollingMethod[PollingReturnType_co]) -> PollingReturnType_co:
    """Async Poller for long running operations.

    .. deprecated:: 1.5.0
       Use :class:`AsyncLROPoller` instead.

    :param client: A pipeline service client.
    :type client: ~azure.core.PipelineClient
    :param initial_response: The initial call response
    :type initial_response: ~azure.core.pipeline.PipelineResponse
    :param deserialization_callback: A callback that takes a Response and return a deserialized object.
                                     If a subclass of Model is given, this passes "deserialize" as callback.
    :type deserialization_callback: callable or msrest.serialization.Model
    :param polling_method: The polling strategy to adopt
    :type polling_method: ~azure.core.polling.PollingMethod
    :return: The final resource at the end of the polling.
    :rtype: any or None
    """
    poller = AsyncLROPoller(client, initial_response, deserialization_callback, polling_method)
    return await poller

class AsyncLROPoller(Generic[PollingReturnType_co], Awaitable[PollingReturnType_co]):
    """Async poller for long running operations.

    :param client: A pipeline service client
    :type client: ~azure.core.PipelineClient
    :param initial_response: The initial call response
    :type initial_response: ~azure.core.pipeline.PipelineResponse
    :param deserialization_callback: A callback that takes a Response and return a deserialized object.
                                     If a subclass of Model is given, this passes "deserialize" as callback.
    :type deserialization_callback: callable or msrest.serialization.Model
    :param polling_method: The polling strategy to adopt
    :type polling_method: ~azure.core.polling.AsyncPollingMethod
    """

    def __init__(self, client: Any, initial_response: Any, deserialization_callback: Callable[[Any], PollingReturnType_co], polling_method: AsyncPollingMethod[PollingReturnType_co]):
        if False:
            while True:
                i = 10
        self._polling_method = polling_method
        self._done = False
        try:
            deserialization_callback = deserialization_callback.deserialize
        except AttributeError:
            pass
        self._polling_method.initialize(client, initial_response, deserialization_callback)

    def polling_method(self) -> AsyncPollingMethod[PollingReturnType_co]:
        if False:
            i = 10
            return i + 15
        'Return the polling method associated to this poller.\n\n        :return: The polling method associated to this poller.\n        :rtype: ~azure.core.polling.AsyncPollingMethod\n        '
        return self._polling_method

    def continuation_token(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return a continuation token that allows to restart the poller later.\n\n        :returns: An opaque continuation token\n        :rtype: str\n        '
        return self._polling_method.get_continuation_token()

    @classmethod
    def from_continuation_token(cls, polling_method: AsyncPollingMethod[PollingReturnType_co], continuation_token: str, **kwargs: Any) -> 'AsyncLROPoller[PollingReturnType_co]':
        if False:
            return 10
        (client, initial_response, deserialization_callback) = polling_method.from_continuation_token(continuation_token, **kwargs)
        return cls(client, initial_response, deserialization_callback, polling_method)

    def status(self) -> str:
        if False:
            print('Hello World!')
        'Returns the current status string.\n\n        :returns: The current status string\n        :rtype: str\n        '
        return self._polling_method.status()

    async def result(self) -> PollingReturnType_co:
        """Return the result of the long running operation.

        :returns: The deserialized resource of the long running operation, if one is available.
        :rtype: any or None
        :raises ~azure.core.exceptions.HttpResponseError: Server problem with the query.
        """
        await self.wait()
        return self._polling_method.resource()

    def __await__(self) -> Generator[Any, None, PollingReturnType_co]:
        if False:
            print('Hello World!')
        return self.result().__await__()

    async def wait(self) -> None:
        """Wait on the long running operation.

        :raises ~azure.core.exceptions.HttpResponseError: Server problem with the query.
        """
        try:
            await self._polling_method.run()
        except AzureError as error:
            if not error.continuation_token:
                try:
                    error.continuation_token = self.continuation_token()
                except Exception as err:
                    _LOGGER.warning('Unable to retrieve continuation token: %s', err)
                    error.continuation_token = None
            raise
        self._done = True

    def done(self) -> bool:
        if False:
            return 10
        "Check status of the long running operation.\n\n        :returns: 'True' if the process has completed, else 'False'.\n        :rtype: bool\n        "
        return self._done