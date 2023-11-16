import os
from dagster import job, op, repository
if os.getenv('FOO') != 'BAR':
    raise Exception('Missing env var')

@op
def needs_env_var():
    if False:
        return 10
    if os.getenv('FOO_INSIDE_OP') != 'BAR_INSIDE_OP':
        raise Exception('Missing env var inside op')

@job
def needs_env_var_job():
    if False:
        for i in range(10):
            print('nop')
    needs_env_var()

@repository
def needs_env_var_repo():
    if False:
        return 10
    return [needs_env_var_job]