import pytest
from superagi.models.events import Event
from superagi.apm.analytics_helper import AnalyticsHelper
from unittest.mock import MagicMock

@pytest.fixture
def organisation_id():
    if False:
        return 10
    return 1

@pytest.fixture
def mock_session():
    if False:
        i = 10
        return i + 15
    return MagicMock()

@pytest.fixture
def analytics_helper(mock_session, organisation_id):
    if False:
        print('Hello World!')
    return AnalyticsHelper(mock_session, organisation_id)

def test_calculate_run_completed_metrics(analytics_helper, mock_session):
    if False:
        return 10
    mock_session.query().all.return_value = [MagicMock()]
    result = analytics_helper.calculate_run_completed_metrics()
    assert isinstance(result, dict)

def test_fetch_agent_data(analytics_helper, mock_session):
    if False:
        i = 10
        return i + 15
    mock_session.query().all.return_value = [MagicMock()]
    result = analytics_helper.fetch_agent_data()
    assert isinstance(result, dict)

def test_fetch_agent_runs(analytics_helper, mock_session):
    if False:
        while True:
            i = 10
    mock_session.query().all.return_value = [MagicMock()]
    result = analytics_helper.fetch_agent_runs(1)
    assert isinstance(result, list)

def test_get_active_runs(analytics_helper, mock_session):
    if False:
        return 10
    mock_session.query().all.return_value = [MagicMock()]
    result = analytics_helper.get_active_runs()
    assert isinstance(result, list)