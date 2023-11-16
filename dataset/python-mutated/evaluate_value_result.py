from typing import Any, Generator, Generic, Optional, Sequence, TypeVar
import dagster._check as check
from .errors import EvaluationError
T = TypeVar('T')

class EvaluateValueResult(Generic[T]):
    success: Optional[bool]
    value: Optional[T]
    errors: Optional[Sequence[EvaluationError]]

    def __init__(self, success: Optional[bool], value: T, errors: Optional[Sequence[EvaluationError]]):
        if False:
            print('Hello World!')
        self.success = check.opt_bool_param(success, 'success')
        self.value = value
        self.errors = check.opt_sequence_param(errors, 'errors', of_type=EvaluationError)

    @staticmethod
    def for_error(error: EvaluationError) -> 'EvaluateValueResult[Any]':
        if False:
            while True:
                i = 10
        return EvaluateValueResult(False, None, [error])

    @staticmethod
    def for_errors(errors: Sequence[EvaluationError]) -> 'EvaluateValueResult[Any]':
        if False:
            while True:
                i = 10
        return EvaluateValueResult(False, None, errors)

    @staticmethod
    def for_value(value: T) -> 'EvaluateValueResult[T]':
        if False:
            for i in range(10):
                print('nop')
        return EvaluateValueResult(True, value, None)

    def errors_at_level(self, *levels: str) -> Sequence[EvaluationError]:
        if False:
            print('Hello World!')
        return list(self._iterate_errors_at_level(list(levels)))

    def _iterate_errors_at_level(self, levels: Sequence[str]) -> Generator[EvaluationError, None, None]:
        if False:
            while True:
                i = 10
        check.sequence_param(levels, 'levels', of_type=str)
        for error in check.is_list(self.errors):
            if error.stack.levels == levels:
                yield error