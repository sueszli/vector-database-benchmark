import os
ENV_FLAG_IS_TEST = 'IS_TEST'

def _env_flag_enabled(name: str) -> bool:
    if False:
        while True:
            i = 10
    return os.getenv(name, default='False') == 'True'

def is_test() -> bool:
    if False:
        for i in range(10):
            print('nop')
    return _env_flag_enabled(ENV_FLAG_IS_TEST)