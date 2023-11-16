import base64
import logging
import threading
import uuid
from typing import TypeVar, Generic, Any, Callable, Optional, Tuple, List
from azure.core.exceptions import AzureError
from azure.core.tracing.decorator import distributed_trace
from azure.core.tracing.common import with_current_context
PollingReturnType_co = TypeVar('PollingReturnType_co', covariant=True)
DeserializationCallbackType = Any
_LOGGER = logging.getLogger(__name__)

class PollingMethod(Generic[PollingReturnType_co]):
    """ABC class for polling method."""

    def initialize(self, client: Any, initial_response: Any, deserialization_callback: DeserializationCallbackType) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError('This method needs to be implemented')

    def run(self) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('This method needs to be implemented')

    def status(self) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('This method needs to be implemented')

    def finished(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('This method needs to be implemented')

    def resource(self) -> PollingReturnType_co:
        if False:
            print('Hello World!')
        raise NotImplementedError('This method needs to be implemented')

    def get_continuation_token(self) -> str:
        if False:
            return 10
        raise TypeError("Polling method '{}' doesn't support get_continuation_token".format(self.__class__.__name__))

    @classmethod
    def from_continuation_token(cls, continuation_token: str, **kwargs: Any) -> Tuple[Any, Any, DeserializationCallbackType]:
        if False:
            for i in range(10):
                print('nop')
        raise TypeError("Polling method '{}' doesn't support from_continuation_token".format(cls.__name__))

class _SansIONoPolling(Generic[PollingReturnType_co]):
    _deserialization_callback: Callable[[Any], PollingReturnType_co]
    'Deserialization callback passed during initialization'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._initial_response = None

    def initialize(self, _: Any, initial_response: Any, deserialization_callback: Callable[[Any], PollingReturnType_co]) -> None:
        if False:
            print('Hello World!')
        self._initial_response = initial_response
        self._deserialization_callback = deserialization_callback

    def status(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return the current status.\n\n        :rtype: str\n        :return: The current status\n        '
        return 'succeeded'

    def finished(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is this polling finished?\n\n        :rtype: bool\n        :return: Whether this polling is finished\n        '
        return True

    def resource(self) -> PollingReturnType_co:
        if False:
            print('Hello World!')
        return self._deserialization_callback(self._initial_response)

    def get_continuation_token(self) -> str:
        if False:
            print('Hello World!')
        import pickle
        return base64.b64encode(pickle.dumps(self._initial_response)).decode('ascii')

    @classmethod
    def from_continuation_token(cls, continuation_token: str, **kwargs: Any) -> Tuple[Any, Any, Callable[[Any], PollingReturnType_co]]:
        if False:
            print('Hello World!')
        try:
            deserialization_callback = kwargs['deserialization_callback']
        except KeyError:
            raise ValueError("Need kwarg 'deserialization_callback' to be recreated from continuation_token") from None
        import pickle
        initial_response = pickle.loads(base64.b64decode(continuation_token))
        return (None, initial_response, deserialization_callback)

class NoPolling(_SansIONoPolling[PollingReturnType_co], PollingMethod[PollingReturnType_co]):
    """An empty poller that returns the deserialized initial response."""

    def run(self) -> None:
        if False:
            while True:
                i = 10
        'Empty run, no polling.'

class LROPoller(Generic[PollingReturnType_co]):
    """Poller for long running operations.

    :param client: A pipeline service client
    :type client: ~azure.core.PipelineClient
    :param initial_response: The initial call response
    :type initial_response: ~azure.core.pipeline.PipelineResponse
    :param deserialization_callback: A callback that takes a Response and return a deserialized object.
                                     If a subclass of Model is given, this passes "deserialize" as callback.
    :type deserialization_callback: callable or msrest.serialization.Model
    :param polling_method: The polling strategy to adopt
    :type polling_method: ~azure.core.polling.PollingMethod
    """

    def __init__(self, client: Any, initial_response: Any, deserialization_callback: Callable[[Any], PollingReturnType_co], polling_method: PollingMethod[PollingReturnType_co]) -> None:
        if False:
            while True:
                i = 10
        self._callbacks: List[Callable] = []
        self._polling_method = polling_method
        try:
            deserialization_callback = deserialization_callback.deserialize
        except AttributeError:
            pass
        self._polling_method.initialize(client, initial_response, deserialization_callback)
        self._thread = None
        self._done = threading.Event()
        self._exception = None
        if self._polling_method.finished():
            self._done.set()
        else:
            self._thread = threading.Thread(target=with_current_context(self._start), name='LROPoller({})'.format(uuid.uuid4()))
            self._thread.daemon = True
            self._thread.start()

    def _start(self):
        if False:
            print('Hello World!')
        'Start the long running operation.\n        On completion, runs any callbacks.\n        '
        try:
            self._polling_method.run()
        except AzureError as error:
            if not error.continuation_token:
                try:
                    error.continuation_token = self.continuation_token()
                except Exception as err:
                    _LOGGER.warning('Unable to retrieve continuation token: %s', err)
                    error.continuation_token = None
            self._exception = error
        except Exception as error:
            self._exception = error
        finally:
            self._done.set()
        (callbacks, self._callbacks) = (self._callbacks, [])
        while callbacks:
            for call in callbacks:
                call(self._polling_method)
            (callbacks, self._callbacks) = (self._callbacks, [])

    def polling_method(self) -> PollingMethod[PollingReturnType_co]:
        if False:
            for i in range(10):
                print('nop')
        'Return the polling method associated to this poller.\n\n        :return: The polling method\n        :rtype: ~azure.core.polling.PollingMethod\n        '
        return self._polling_method

    def continuation_token(self) -> str:
        if False:
            print('Hello World!')
        'Return a continuation token that allows to restart the poller later.\n\n        :returns: An opaque continuation token\n        :rtype: str\n        '
        return self._polling_method.get_continuation_token()

    @classmethod
    def from_continuation_token(cls, polling_method: PollingMethod[PollingReturnType_co], continuation_token: str, **kwargs: Any) -> 'LROPoller[PollingReturnType_co]':
        if False:
            print('Hello World!')
        (client, initial_response, deserialization_callback) = polling_method.from_continuation_token(continuation_token, **kwargs)
        return cls(client, initial_response, deserialization_callback, polling_method)

    def status(self) -> str:
        if False:
            i = 10
            return i + 15
        'Returns the current status string.\n\n        :returns: The current status string\n        :rtype: str\n        '
        return self._polling_method.status()

    def result(self, timeout: Optional[float]=None) -> PollingReturnType_co:
        if False:
            print('Hello World!')
        'Return the result of the long running operation, or\n        the result available after the specified timeout.\n\n        :param float timeout: Period of time to wait before getting back control.\n        :returns: The deserialized resource of the long running operation, if one is available.\n        :rtype: any or None\n        :raises ~azure.core.exceptions.HttpResponseError: Server problem with the query.\n        '
        self.wait(timeout)
        return self._polling_method.resource()

    @distributed_trace
    def wait(self, timeout: Optional[float]=None) -> None:
        if False:
            print('Hello World!')
        'Wait on the long running operation for a specified length\n        of time. You can check if this call as ended with timeout with the\n        "done()" method.\n\n        :param float timeout: Period of time to wait for the long running\n         operation to complete (in seconds).\n        :raises ~azure.core.exceptions.HttpResponseError: Server problem with the query.\n        '
        if self._thread is None:
            return
        self._thread.join(timeout=timeout)
        try:
            raise self._exception
        except TypeError:
            pass

    def done(self) -> bool:
        if False:
            i = 10
            return i + 15
        "Check status of the long running operation.\n\n        :returns: 'True' if the process has completed, else 'False'.\n        :rtype: bool\n        "
        return self._thread is None or not self._thread.is_alive()

    def add_done_callback(self, func: Callable) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add callback function to be run once the long running operation\n        has completed - regardless of the status of the operation.\n\n        :param callable func: Callback function that takes at least one\n         argument, a completed LongRunningOperation.\n        '
        if self._done.is_set():
            func(self._polling_method)
        self._callbacks.append(func)

    def remove_done_callback(self, func: Callable) -> None:
        if False:
            while True:
                i = 10
        'Remove a callback from the long running operation.\n\n        :param callable func: The function to be removed from the callbacks.\n        :raises ValueError: if the long running operation has already completed.\n        '
        if self._done is None or self._done.is_set():
            raise ValueError('Process is complete.')
        self._callbacks = [c for c in self._callbacks if c != func]