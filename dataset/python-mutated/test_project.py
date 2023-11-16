from unittest.mock import create_autospec
from sqlalchemy.orm import Session
from superagi.models.project import Project

def test_find_by_org_id():
    if False:
        while True:
            i = 10
    session = create_autospec(Session)
    org_id = 123
    mock_project = Project(id=1, name='Test Project', organisation_id=org_id, description='Project for testing')
    session.query.return_value.filter.return_value.first.return_value = mock_project
    project = Project.find_by_org_id(session, org_id)
    assert project == mock_project

def test_find_by_id():
    if False:
        i = 10
        return i + 15
    session = create_autospec(Session)
    project_id = 123
    mock_project = Project(id=project_id, name='Test Project', organisation_id=1, description='Project for testing')
    session.query.return_value.filter.return_value.first.return_value = mock_project
    project = Project.find_by_id(session, project_id)
    assert project == mock_project