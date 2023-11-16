import contextlib
import os
import typing
import unittest
import unittest.mock
from typing import Iterator, Optional, Sequence
import tools.setup_helpers.cmake
import tools.setup_helpers.env
T = typing.TypeVar('T')

class TestCMake(unittest.TestCase):

    @unittest.mock.patch('multiprocessing.cpu_count')
    def test_build_jobs(self, mock_cpu_count: unittest.mock.MagicMock) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that the number of build jobs comes out correctly.'
        mock_cpu_count.return_value = 13
        cases = [(('8', True, False), ['-j', '8']), ((None, True, False), None), (('7', False, False), ['-j', '7']), ((None, False, False), ['-j', '13']), (('6', True, True), ['-j', '6']), ((None, True, True), None), (('11', False, True), ['/p:CL_MPCount=11']), ((None, False, True), ['/p:CL_MPCount=13'])]
        for ((max_jobs, use_ninja, is_windows), want) in cases:
            with self.subTest(MAX_JOBS=max_jobs, USE_NINJA=use_ninja, IS_WINDOWS=is_windows):
                with contextlib.ExitStack() as stack:
                    stack.enter_context(env_var('MAX_JOBS', max_jobs))
                    stack.enter_context(unittest.mock.patch.object(tools.setup_helpers.cmake, 'USE_NINJA', use_ninja))
                    stack.enter_context(unittest.mock.patch.object(tools.setup_helpers.cmake, 'IS_WINDOWS', is_windows))
                    cmake = tools.setup_helpers.cmake.CMake()
                    with unittest.mock.patch.object(cmake, 'run') as cmake_run:
                        cmake.build({})
                    cmake_run.assert_called_once()
                    (call,) = cmake_run.mock_calls
                    (build_args, _) = call.args
                if want is None:
                    self.assertNotIn('-j', build_args)
                else:
                    self.assert_contains_sequence(build_args, want)

    @staticmethod
    def assert_contains_sequence(sequence: Sequence[T], subsequence: Sequence[T]) -> None:
        if False:
            while True:
                i = 10
        'Raises an assertion if the subsequence is not contained in the sequence.'
        if len(subsequence) == 0:
            return
        for i in range(len(sequence) - len(subsequence) + 1):
            candidate = sequence[i:i + len(subsequence)]
            assert len(candidate) == len(subsequence)
            if candidate == subsequence:
                return
        raise AssertionError(f'{subsequence} not found in {sequence}')

@contextlib.contextmanager
def env_var(key: str, value: Optional[str]) -> Iterator[None]:
    if False:
        for i in range(10):
            print('nop')
    'Sets/clears an environment variable within a Python context.'
    previous_value = os.environ.get(key)
    set_env_var(key, value)
    try:
        yield
    finally:
        set_env_var(key, previous_value)

def set_env_var(key: str, value: Optional[str]) -> None:
    if False:
        while True:
            i = 10
    'Sets/clears an environment variable.'
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
if __name__ == '__main__':
    unittest.main()