"""
Tests for configdialog.py
"""
import os.path as osp
import tempfile
import pytest
from spyder.plugins.projects.api import EmptyProject

@pytest.fixture
def project(qtbot):
    if False:
        return 10
    'Set up ProjectPreferences.'
    project_dir = tempfile.mkdtemp() + osp.sep + '.spyproject'
    project = EmptyProject(project_dir, None)
    return project

def test_projects_preferences(project):
    if False:
        print('Hello World!')
    'Run Project Preferences.'
    assert project
if __name__ == '__main__':
    pytest.main()