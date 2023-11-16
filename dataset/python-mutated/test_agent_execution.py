from unittest.mock import patch
from unittest import mock
import pytest
from fastapi.testclient import TestClient
from main import app
from superagi.models.agent_schedule import AgentSchedule
from datetime import datetime
client = TestClient(app)

@pytest.fixture
def mock_patch_schedule_input():
    if False:
        i = 10
        return i + 15
    return {'agent_id': 1, 'start_time': '2023-02-02 01:00:00', 'recurrence_interval': '2 Hours', 'expiry_date': '2023-12-30 01:00:00', 'expiry_runs': -1}

@pytest.fixture
def mock_schedule():
    if False:
        i = 10
        return i + 15
    return AgentSchedule(id=1, agent_id=1, status='SCHEDULED')

def test_schedule_existing_agent_already_scheduled(mock_patch_schedule_input, mock_schedule):
    if False:
        i = 10
        return i + 15
    with patch('superagi.controllers.agent_execution.db') as mock_db:
        mock_db.session.query.return_value.filter.return_value.first.return_value = mock_schedule
        response = client.post('agentexecutions/schedule', json=mock_patch_schedule_input)
        assert response.status_code == 201
        assert mock_schedule.start_time == datetime.strptime(mock_patch_schedule_input['start_time'], '%Y-%m-%d %H:%M:%S')
        assert mock_schedule.recurrence_interval == mock_patch_schedule_input['recurrence_interval']
        assert mock_schedule.expiry_date == datetime.strptime(mock_patch_schedule_input['expiry_date'], '%Y-%m-%d %H:%M:%S')
        assert mock_schedule.expiry_runs == mock_patch_schedule_input['expiry_runs']

def test_schedule_existing_agent_new_schedule(mock_patch_schedule_input, mock_schedule):
    if False:
        for i in range(10):
            print('nop')
    with patch('superagi.controllers.agent_execution.db') as mock_db:
        mock_db.session.query.return_value.filter.return_value.first.return_value = mock_schedule
        response = client.post('agentexecutions/schedule', json=mock_patch_schedule_input)
        assert response.status_code == 201
        assert response.json()['schedule_id'] is not None