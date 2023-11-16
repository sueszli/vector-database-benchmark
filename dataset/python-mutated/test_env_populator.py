import os
import pathlib
import pytest
from robyn.env_populator import load_vars, parser

@pytest.mark.benchmark
def test_env_population(test_session, env_file):
    if False:
        return 10
    path = pathlib.Path(__file__).parent
    env_path = path / 'robyn.env'
    load_vars(variables=parser(config_path=env_path))
    PORT = os.environ['ROBYN_PORT']
    HOST = os.environ['ROBYN_HOST']
    assert PORT == '8080'
    assert HOST == '127.0.0.1'