"""``PTransform`` for reading from and writing to Web APIs."""
import abc
from typing import TypeVar
RequestT = TypeVar('RequestT')
ResponseT = TypeVar('ResponseT')

class UserCodeExecutionException(Exception):
    """Base class for errors related to calling Web APIs."""

class UserCodeQuotaException(UserCodeExecutionException):
    """Extends ``UserCodeExecutionException`` to signal specifically that
  the Web API client encountered a Quota or API overuse related error.
  """

class UserCodeTimeoutException(UserCodeExecutionException):
    """Extends ``UserCodeExecutionException`` to signal a user code timeout."""

class Caller(metaclass=abc.ABCMeta):
    """Interfaces user custom code intended for API calls."""

    @abc.abstractmethod
    def call(self, request: RequestT) -> ResponseT:
        if False:
            i = 10
            return i + 15
        'Calls a Web API with the ``RequestT``  and returns a\n    ``ResponseT``. ``RequestResponseIO`` expects implementations of the\n    call method to throw either a ``UserCodeExecutionException``,\n    ``UserCodeQuotaException``, or ``UserCodeTimeoutException``.\n    '
        pass

class SetupTeardown(metaclass=abc.ABCMeta):
    """Interfaces user custom code to set up and teardown the API clients.
    Called by ``RequestResponseIO`` within its DoFn's setup and teardown
    methods.
    """

    @abc.abstractmethod
    def setup(self) -> None:
        if False:
            while True:
                i = 10
        "Called during the DoFn's setup lifecycle method."
        pass

    @abc.abstractmethod
    def teardown(self) -> None:
        if False:
            return 10
        "Called during the DoFn's teardown lifecycle method."
        pass