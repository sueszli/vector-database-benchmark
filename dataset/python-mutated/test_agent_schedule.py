from unittest.mock import create_autospec
from sqlalchemy.orm import Session
from superagi.models.agent_schedule import AgentSchedule

def test_find_by_agent_id():
    if False:
        i = 10
        return i + 15
    session = create_autospec(Session)
    agent_id = 1
    mock_agent_schedule = AgentSchedule(id=1, agent_id=agent_id, start_time='2023-08-10 12:17:00', recurrence_interval='2 Minutes', expiry_runs=2)
    session.query.return_value.filter.return_value.first.return_value = mock_agent_schedule
    agent_schedule = AgentSchedule.find_by_agent_id(session, agent_id)
    assert agent_schedule == mock_agent_schedule