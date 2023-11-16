"""Objects for holding onto the results produced by Apache Beam jobs."""
from __future__ import annotations
import heapq
from core import utils
from typing import Any, List, Tuple, Union
MAX_OUTPUT_CHARACTERS = 5000
TRUNCATED_MARK = '[TRUNCATED]'

class JobRunResult:
    """Encapsulates the result of a job run.

    The stdout and stderr are string values analogous to a program's stdout and
    stderr pipes (reserved for standard output and errors, respectively).
    """
    __slots__ = ('stdout', 'stderr')

    def __init__(self, stdout: str='', stderr: str=''):
        if False:
            return 10
        'Initializes a new JobRunResult instance.\n\n        Args:\n            stdout: str. The standard output from a job run.\n            stderr: str. The error output from a job run.\n\n        Raises:\n            ValueError. Both stdout and stderr are empty.\n            ValueError. JobRunResult exceeds maximum limit.\n        '
        if not stdout and (not stderr):
            raise ValueError('JobRunResult instances must not be empty')
        (self.stdout, self.stderr) = (stdout, stderr)
        if len(self.stdout) > MAX_OUTPUT_CHARACTERS:
            self.stdout = '%s%s' % (self.stdout[:MAX_OUTPUT_CHARACTERS], TRUNCATED_MARK)
        if len(self.stderr) > MAX_OUTPUT_CHARACTERS:
            self.stderr = '%s%s' % (self.stderr[:MAX_OUTPUT_CHARACTERS], TRUNCATED_MARK)

    @classmethod
    def as_stdout(cls, value: Union[str, int], use_repr: bool=False) -> JobRunResult:
        if False:
            print('Hello World!')
        "Returns a new JobRunResult with a stdout value.\n\n        Args:\n            value: *. The input value to convert into a stdout result. Types are\n                always casted to string using '%s' formatting.\n            use_repr: bool. Whether to use the `repr` of the value.\n\n        Returns:\n            JobRunResult. A JobRunResult with the given value as its stdout.\n        "
        str_value = ('%r' if use_repr else '%s') % (value,)
        return JobRunResult(stdout=str_value)

    @classmethod
    def as_stderr(cls, value: Union[str, int], use_repr: bool=False) -> JobRunResult:
        if False:
            for i in range(10):
                print('nop')
        "Returns a new JobRunResult with a stderr value.\n\n        Args:\n            value: *. The input value to convert into a stderr result. Types are\n                always casted to string using '%s' formatting.\n            use_repr: bool. Whether to use the `repr` of the value.\n\n        Returns:\n            JobRunResult. A JobRunResult with the given value as its stderr.\n        "
        str_value = ('%r' if use_repr else '%s') % (value,)
        return JobRunResult(stderr=str_value)

    @classmethod
    def accumulate(cls, results: List[JobRunResult]) -> List[JobRunResult]:
        if False:
            for i in range(10):
                print('nop')
        'Accumulates results into bigger ones that maintain the size limit.\n\n        The len_in_bytes() of each result is always less than MAX_OUTPUT_BYTES.\n\n        Args:\n            results: list(JobRunResult). The results to concatenate.\n\n        Returns:\n            list(JobRunResult). JobRunResult instances with stdout and stderr\n            values concatenated together with newline delimiters. Each\n            individual item maintains the size limit.\n        '
        if not results:
            return []
        results_heap: List[Tuple[int, int, JobRunResult]] = []
        for (i, result) in enumerate(results):
            heapq.heappush(results_heap, (len(result.stdout) + len(result.stderr), i, result))
        batches = []
        (latest_batch_size, _, smallest) = heapq.heappop(results_heap)
        batches.append([smallest])
        while results_heap:
            (result_size, _, next_smallest) = heapq.heappop(results_heap)
            padding = 2 if next_smallest.stdout and next_smallest.stderr else 1
            overall_size = latest_batch_size + padding + result_size
            if overall_size <= MAX_OUTPUT_CHARACTERS:
                latest_batch_size += padding + result_size
                batches[-1].append(next_smallest)
            else:
                latest_batch_size = result_size
                batches.append([next_smallest])
        batched_results = []
        for batch in batches:
            stdout = '\n'.join((r.stdout for r in batch if r.stdout))
            stderr = '\n'.join((r.stderr for r in batch if r.stderr))
            batched_results.append(JobRunResult(stdout=stdout, stderr=stderr))
        return batched_results

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return '%s(stdout=%s, stderr=%s)' % (self.__class__.__name__, utils.quoted(self.stdout), utils.quoted(self.stderr))

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        return hash((self.stdout, self.stderr))

    def __eq__(self, other: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return (self.stdout, self.stderr) == (other.stdout, other.stderr) if self.__class__ is other.__class__ else NotImplemented

    def __ne__(self, other: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return not self == other if self.__class__ is other.__class__ else NotImplemented

    def __getstate__(self) -> Tuple[str, str]:
        if False:
            print('Hello World!')
        'Called by pickle to get the value that uniquely defines self.'
        return (self.stdout, self.stderr)

    def __setstate__(self, state: Tuple[str, str]) -> None:
        if False:
            print('Hello World!')
        "Called by pickle to build an instance from __getstate__'s value."
        (self.stdout, self.stderr) = state