import os
import pytest
from molecule.command.init import base

class CommandBase(base.Base):
    """CommandBase Class."""

@pytest.fixture()
def _base_class():
    if False:
        print('Hello World!')
    return CommandBase

@pytest.fixture()
def _instance(_base_class):
    if False:
        return 10
    return _base_class()

@pytest.fixture()
def _role_directory():
    if False:
        while True:
            i = 10
    return '.'

@pytest.fixture()
def _command_args():
    if False:
        return 10
    return {'dependency_name': 'galaxy', 'driver_name': 'default', 'provisioner_name': 'ansible', 'scenario_name': 'default', 'role_name': 'test-role', 'verifier_name': 'ansible'}

@pytest.fixture()
def _molecule_file(_role_directory):
    if False:
        i = 10
        return i + 15
    return os.path.join(_role_directory, 'test-role', 'molecule', 'default', 'molecule.yml')