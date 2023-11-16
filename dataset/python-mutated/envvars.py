import os

def env_var_2_bool(env_var: object) -> bool:
    if False:
        i = 10
        return i + 15
    if isinstance(env_var, bool):
        return env_var
    if not isinstance(env_var, str):
        return False
    return env_var.lower().strip() == 'true'
GITHUB_ACTION = os.getenv('GITHUB_ACTION', '')
TEST_WITH_OPT_DEPS = env_var_2_bool(os.getenv('TEST_WITH_OPT_DEPS', 'true'))
RUN_TEST_OFFICIAL = env_var_2_bool(os.getenv('TEST_OFFICIAL'))