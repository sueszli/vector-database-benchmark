import pytest
from unittest.mock import patch, Mock
from superagi.tools.jira.get_projects import GetProjectsTool

@patch('superagi.tools.jira.get_projects.JiraTool.build_jira_instance')
def test_get_projects_tool(mock_build_jira_instance):
    if False:
        for i in range(10):
            print('nop')
    mock_jira_instance = Mock()
    mock_project_1 = Mock()
    mock_project_1.id = '123'
    mock_project_1.key = 'PRJ1'
    mock_project_1.name = 'Project 1'
    mock_projects = [mock_project_1]
    mock_jira_instance.projects.return_value = mock_projects
    mock_build_jira_instance.return_value = mock_jira_instance
    tool = GetProjectsTool()
    result = tool._execute()
    mock_jira_instance.projects.assert_called_once()
    assert 'Found 1 projects' in result
    assert '123' in result
    assert 'PRJ1' in result
    assert 'Project 1' in result