from __future__ import annotations
import os
from airflow_breeze.utils.coertions import coerce_bool_value

def __get_default_bool_value(env_var: str) -> bool:
    if False:
        print('Hello World!')
    string_val = os.environ.get(env_var, '')
    return coerce_bool_value(string_val)
__verbose_value: bool = __get_default_bool_value('VERBOSE')

def set_verbose(verbose: bool):
    if False:
        return 10
    global __verbose_value
    __verbose_value = verbose

def get_verbose(verbose_override: bool | None=None) -> bool:
    if False:
        return 10
    if verbose_override is None:
        return __verbose_value
    return verbose_override
__dry_run_value: bool = __get_default_bool_value('DRY_RUN')

def set_dry_run(dry_run: bool):
    if False:
        print('Hello World!')
    global __dry_run_value
    __dry_run_value = dry_run

def get_dry_run(dry_run_override: bool | None=None) -> bool:
    if False:
        print('Hello World!')
    if dry_run_override is None:
        return __dry_run_value
    return dry_run_override
__forced_answer: str | None = None

def set_forced_answer(answer: str | None):
    if False:
        for i in range(10):
            print('nop')
    global __forced_answer
    __forced_answer = answer

def get_forced_answer(answer_override: str | None=None) -> str | None:
    if False:
        print('Hello World!')
    if answer_override is None:
        return __forced_answer
    return answer_override