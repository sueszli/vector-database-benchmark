"""Tests around detection whether cookiecutter templates are cached locally."""
import os
from pathlib import Path
import pytest
from cookiecutter import repository

@pytest.fixture
def template():
    if False:
        return 10
    'Fixture. Return simple string as template name.'
    return 'cookiecutter-pytest-plugin'

@pytest.fixture
def cloned_cookiecutter_path(user_config_data, template):
    if False:
        for i in range(10):
            print('nop')
    'Fixture. Create fake project directory in special user folder.'
    cookiecutters_dir = user_config_data['cookiecutters_dir']
    cloned_template_path = os.path.join(cookiecutters_dir, template)
    os.mkdir(cloned_template_path)
    Path(cloned_template_path, 'cookiecutter.json').touch()
    return cloned_template_path

def test_should_find_existing_cookiecutter(template, user_config_data, cloned_cookiecutter_path):
    if False:
        return 10
    '\n    Should find folder created by `cloned_cookiecutter_path` and return it.\n\n    This folder is considered like previously cloned project directory.\n    '
    (project_dir, cleanup) = repository.determine_repo_dir(template=template, abbreviations={}, clone_to_dir=user_config_data['cookiecutters_dir'], checkout=None, no_input=True)
    assert cloned_cookiecutter_path == project_dir
    assert not cleanup