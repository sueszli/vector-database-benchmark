import pytest
from ludwig.constants import TRIES
from ludwig.utils.error_handling_utils import default_retry

def test_default_retry_success():
    if False:
        return 10
    ctr = 0

    @default_retry()
    def flaky_function():
        if False:
            while True:
                i = 10
        nonlocal ctr
        if ctr < TRIES - 1:
            ctr += 1
            raise Exception(f'Ctr: {ctr} too low.')
        return
    flaky_function()

def test_default_retry_failure():
    if False:
        i = 10
        return i + 15
    ctr = 0

    @default_retry()
    def flaky_function():
        if False:
            for i in range(10):
                print('nop')
        nonlocal ctr
        if ctr < TRIES:
            ctr += 1
            raise Exception(f'Ctr: {ctr} too low.')
        return
    with pytest.raises(Exception):
        flaky_function()

def test_default_retry_success_custom_num_tries():
    if False:
        return 10
    CUSTOM_TRIES = 3
    ctr = 0

    @default_retry(tries=CUSTOM_TRIES)
    def flaky_function():
        if False:
            i = 10
            return i + 15
        nonlocal ctr
        if ctr < CUSTOM_TRIES - 1:
            ctr += 1
            raise Exception(f'Ctr: {ctr} too low.')
        return
    flaky_function()