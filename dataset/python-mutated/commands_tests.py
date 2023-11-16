import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import call, Mock, patch
from uuid import uuid4
import pytest
from flask import current_app
from flask_appbuilder.security.sqla.models import User
from flask_sqlalchemy import BaseQuery
from freezegun import freeze_time
from slack_sdk.errors import BotUserAccessError, SlackApiError, SlackClientConfigurationError, SlackClientError, SlackClientNotConnectedError, SlackObjectFormationError, SlackRequestError, SlackTokenRotationError
from sqlalchemy.sql import func
from superset import db
from superset.exceptions import SupersetException
from superset.models.core import Database
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.reports.commands.exceptions import AlertQueryError, AlertQueryInvalidTypeError, AlertQueryMultipleColumnsError, AlertQueryMultipleRowsError, ReportScheduleClientErrorsException, ReportScheduleCsvFailedError, ReportScheduleCsvTimeout, ReportScheduleNotFoundError, ReportSchedulePreviousWorkingError, ReportScheduleScreenshotFailedError, ReportScheduleScreenshotTimeout, ReportScheduleSystemErrorsException, ReportScheduleWorkingTimeoutError
from superset.reports.commands.execute import AsyncExecuteReportScheduleCommand, BaseReportState
from superset.reports.commands.log_prune import AsyncPruneReportScheduleLogCommand
from superset.reports.models import ReportDataFormat, ReportExecutionLog, ReportSchedule, ReportScheduleType, ReportScheduleValidatorType, ReportState
from superset.reports.notifications.exceptions import NotificationError, NotificationParamException
from superset.tasks.types import ExecutorType
from superset.utils.database import get_example_database
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from tests.integration_tests.fixtures.world_bank_dashboard import load_world_bank_dashboard_with_slices_module_scope, load_world_bank_data
from tests.integration_tests.reports.utils import cleanup_report_schedule, create_report_notification, CSV_FILE, DEFAULT_OWNER_EMAIL, SCREENSHOT_FILE, TEST_ID
from tests.integration_tests.test_app import app
pytestmark = pytest.mark.usefixtures('load_world_bank_dashboard_with_slices_module_scope')

def get_target_from_report_schedule(report_schedule: ReportSchedule) -> list[str]:
    if False:
        print('Hello World!')
    return [json.loads(recipient.recipient_config_json)['target'] for recipient in report_schedule.recipients]

def get_error_logs_query(report_schedule: ReportSchedule) -> BaseQuery:
    if False:
        for i in range(10):
            print('nop')
    return db.session.query(ReportExecutionLog).filter(ReportExecutionLog.report_schedule == report_schedule, ReportExecutionLog.state == ReportState.ERROR).order_by(ReportExecutionLog.end_dttm.desc())

def get_notification_error_sent_count(report_schedule: ReportSchedule) -> int:
    if False:
        i = 10
        return i + 15
    logs = get_error_logs_query(report_schedule).all()
    notification_sent_logs = [log.error_message for log in logs if log.error_message == 'Notification sent with error']
    return len(notification_sent_logs)

def assert_log(state: str, error_message: Optional[str]=None):
    if False:
        return 10
    db.session.commit()
    logs = db.session.query(ReportExecutionLog).all()
    if state == ReportState.ERROR:
        assert len(logs) == 3
    else:
        assert len(logs) == 2
    log_states = [log.state for log in logs]
    assert ReportState.WORKING in log_states
    assert state in log_states
    assert error_message in [log.error_message for log in logs]
    for log in logs:
        if log.state == ReportState.WORKING:
            assert log.value is None
            assert log.value_row_json is None

@contextmanager
def create_test_table_context(database: Database):
    if False:
        for i in range(10):
            print('nop')
    with database.get_sqla_engine_with_context() as engine:
        engine.execute('CREATE TABLE test_table AS SELECT 1 as first, 2 as second')
        engine.execute('INSERT INTO test_table (first, second) VALUES (1, 2)')
        engine.execute('INSERT INTO test_table (first, second) VALUES (3, 4)')
    yield db.session
    with database.get_sqla_engine_with_context() as engine:
        engine.execute('DROP TABLE test_table')

@pytest.fixture()
def create_report_email_chart():
    if False:
        return 10
    with app.app_context():
        chart = db.session.query(Slice).first()
        report_schedule = create_report_notification(email_target='target@email.com', chart=chart)
        yield report_schedule
        cleanup_report_schedule(report_schedule)

@pytest.fixture()
def create_report_email_chart_alpha_owner(get_user):
    if False:
        i = 10
        return i + 15
    with app.app_context():
        owners = [get_user('alpha')]
        chart = db.session.query(Slice).first()
        report_schedule = create_report_notification(email_target='target@email.com', chart=chart, owners=owners)
        yield report_schedule
        cleanup_report_schedule(report_schedule)

@pytest.fixture()
def create_report_email_chart_force_screenshot():
    if False:
        return 10
    with app.app_context():
        chart = db.session.query(Slice).first()
        report_schedule = create_report_notification(email_target='target@email.com', chart=chart, force_screenshot=True)
        yield report_schedule
        cleanup_report_schedule(report_schedule)

@pytest.fixture()
def create_report_email_chart_with_csv():
    if False:
        while True:
            i = 10
    with app.app_context():
        chart = db.session.query(Slice).first()
        chart.query_context = '{"mock": "query_context"}'
        report_schedule = create_report_notification(email_target='target@email.com', chart=chart, report_format=ReportDataFormat.DATA)
        yield report_schedule
        cleanup_report_schedule(report_schedule)

@pytest.fixture()
def create_report_email_chart_with_text():
    if False:
        i = 10
        return i + 15
    with app.app_context():
        chart = db.session.query(Slice).first()
        chart.query_context = '{"mock": "query_context"}'
        report_schedule = create_report_notification(email_target='target@email.com', chart=chart, report_format=ReportDataFormat.TEXT)
        yield report_schedule
        cleanup_report_schedule(report_schedule)

@pytest.fixture()
def create_report_email_chart_with_csv_no_query_context():
    if False:
        i = 10
        return i + 15
    with app.app_context():
        chart = db.session.query(Slice).first()
        chart.query_context = None
        report_schedule = create_report_notification(email_target='target@email.com', chart=chart, report_format=ReportDataFormat.DATA, name='report_csv_no_query_context')
        yield report_schedule
        cleanup_report_schedule(report_schedule)

@pytest.fixture()
def create_report_email_dashboard():
    if False:
        for i in range(10):
            print('nop')
    with app.app_context():
        dashboard = db.session.query(Dashboard).first()
        report_schedule = create_report_notification(email_target='target@email.com', dashboard=dashboard)
        yield report_schedule
        cleanup_report_schedule(report_schedule)

@pytest.fixture()
def create_report_email_dashboard_force_screenshot():
    if False:
        for i in range(10):
            print('nop')
    with app.app_context():
        dashboard = db.session.query(Dashboard).first()
        report_schedule = create_report_notification(email_target='target@email.com', dashboard=dashboard, force_screenshot=True)
        yield report_schedule
        cleanup_report_schedule(report_schedule)

@pytest.fixture()
def create_report_slack_chart():
    if False:
        i = 10
        return i + 15
    with app.app_context():
        chart = db.session.query(Slice).first()
        report_schedule = create_report_notification(slack_channel='slack_channel', chart=chart)
        yield report_schedule
        cleanup_report_schedule(report_schedule)

@pytest.fixture()
def create_report_slack_chart_with_csv():
    if False:
        i = 10
        return i + 15
    with app.app_context():
        chart = db.session.query(Slice).first()
        chart.query_context = '{"mock": "query_context"}'
        report_schedule = create_report_notification(slack_channel='slack_channel', chart=chart, report_format=ReportDataFormat.DATA)
        yield report_schedule
        cleanup_report_schedule(report_schedule)

@pytest.fixture()
def create_report_slack_chart_with_text():
    if False:
        for i in range(10):
            print('nop')
    with app.app_context():
        chart = db.session.query(Slice).first()
        chart.query_context = '{"mock": "query_context"}'
        report_schedule = create_report_notification(slack_channel='slack_channel', chart=chart, report_format=ReportDataFormat.TEXT)
        yield report_schedule
        cleanup_report_schedule(report_schedule)

@pytest.fixture()
def create_report_slack_chart_working():
    if False:
        i = 10
        return i + 15
    with app.app_context():
        chart = db.session.query(Slice).first()
        report_schedule = create_report_notification(slack_channel='slack_channel', chart=chart)
        report_schedule.last_state = ReportState.WORKING
        report_schedule.last_eval_dttm = datetime(2020, 1, 1, 0, 0)
        report_schedule.last_value = None
        report_schedule.last_value_row_json = None
        db.session.commit()
        log = ReportExecutionLog(scheduled_dttm=report_schedule.last_eval_dttm, start_dttm=report_schedule.last_eval_dttm, end_dttm=report_schedule.last_eval_dttm, value=report_schedule.last_value, value_row_json=report_schedule.last_value_row_json, state=ReportState.WORKING, report_schedule=report_schedule, uuid=uuid4())
        db.session.add(log)
        db.session.commit()
        yield report_schedule
        cleanup_report_schedule(report_schedule)

@pytest.fixture()
def create_alert_slack_chart_success():
    if False:
        i = 10
        return i + 15
    with app.app_context():
        chart = db.session.query(Slice).first()
        report_schedule = create_report_notification(slack_channel='slack_channel', chart=chart, report_type=ReportScheduleType.ALERT)
        report_schedule.last_state = ReportState.SUCCESS
        report_schedule.last_eval_dttm = datetime(2020, 1, 1, 0, 0)
        log = ReportExecutionLog(report_schedule=report_schedule, state=ReportState.SUCCESS, start_dttm=report_schedule.last_eval_dttm, end_dttm=report_schedule.last_eval_dttm, scheduled_dttm=report_schedule.last_eval_dttm)
        db.session.add(log)
        db.session.commit()
        yield report_schedule
        cleanup_report_schedule(report_schedule)

@pytest.fixture(params=['alert1'])
def create_alert_slack_chart_grace(request):
    if False:
        while True:
            i = 10
    param_config = {'alert1': {'sql': 'SELECT count(*) from test_table', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": "<", "threshold": 10}'}}
    with app.app_context():
        chart = db.session.query(Slice).first()
        example_database = get_example_database()
        with create_test_table_context(example_database):
            report_schedule = create_report_notification(slack_channel='slack_channel', chart=chart, report_type=ReportScheduleType.ALERT, database=example_database, sql=param_config[request.param]['sql'], validator_type=param_config[request.param]['validator_type'], validator_config_json=param_config[request.param]['validator_config_json'])
            report_schedule.last_state = ReportState.GRACE
            report_schedule.last_eval_dttm = datetime(2020, 1, 1, 0, 0)
            log = ReportExecutionLog(report_schedule=report_schedule, state=ReportState.SUCCESS, start_dttm=report_schedule.last_eval_dttm, end_dttm=report_schedule.last_eval_dttm, scheduled_dttm=report_schedule.last_eval_dttm)
            db.session.add(log)
            db.session.commit()
            yield report_schedule
            cleanup_report_schedule(report_schedule)

@pytest.fixture(params=['alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6', 'alert7', 'alert8'])
def create_alert_email_chart(request):
    if False:
        return 10
    param_config = {'alert1': {'sql': 'SELECT 10 as metric', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": ">", "threshold": 9}'}, 'alert2': {'sql': 'SELECT 10 as metric', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": ">=", "threshold": 10}'}, 'alert3': {'sql': 'SELECT 10 as metric', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": "<", "threshold": 11}'}, 'alert4': {'sql': 'SELECT 10 as metric', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": "<=", "threshold": 10}'}, 'alert5': {'sql': 'SELECT 10 as metric', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": "!=", "threshold": 11}'}, 'alert6': {'sql': "SELECT 'something' as metric", 'validator_type': ReportScheduleValidatorType.NOT_NULL, 'validator_config_json': '{}'}, 'alert7': {'sql': 'SELECT {{ 5 + 5 }} as metric', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": "!=", "threshold": 11}'}, 'alert8': {'sql': 'SELECT 55 as metric', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": ">", "threshold": 54.999}'}}
    with app.app_context():
        chart = db.session.query(Slice).first()
        example_database = get_example_database()
        with create_test_table_context(example_database):
            report_schedule = create_report_notification(email_target='target@email.com', chart=chart, report_type=ReportScheduleType.ALERT, database=example_database, sql=param_config[request.param]['sql'], validator_type=param_config[request.param]['validator_type'], validator_config_json=param_config[request.param]['validator_config_json'], force_screenshot=True)
            yield report_schedule
            cleanup_report_schedule(report_schedule)

@pytest.fixture(params=['alert1', 'alert2', 'alert3', 'alert4', 'alert5', 'alert6', 'alert7', 'alert8', 'alert9'])
def create_no_alert_email_chart(request):
    if False:
        i = 10
        return i + 15
    param_config = {'alert1': {'sql': 'SELECT 10 as metric', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": "<", "threshold": 10}'}, 'alert2': {'sql': 'SELECT 10 as metric', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": ">=", "threshold": 11}'}, 'alert3': {'sql': 'SELECT 10 as metric', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": "<", "threshold": 10}'}, 'alert4': {'sql': 'SELECT 10 as metric', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": "<=", "threshold": 9}'}, 'alert5': {'sql': 'SELECT 10 as metric', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": "!=", "threshold": 10}'}, 'alert6': {'sql': 'SELECT first from test_table where 1=0', 'validator_type': ReportScheduleValidatorType.NOT_NULL, 'validator_config_json': '{}'}, 'alert7': {'sql': 'SELECT first from test_table where 1=0', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": ">", "threshold": 0}'}, 'alert8': {'sql': 'SELECT Null as metric', 'validator_type': ReportScheduleValidatorType.NOT_NULL, 'validator_config_json': '{}'}, 'alert9': {'sql': 'SELECT Null as metric', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": ">", "threshold": 0}'}}
    with app.app_context():
        chart = db.session.query(Slice).first()
        example_database = get_example_database()
        with create_test_table_context(example_database):
            report_schedule = create_report_notification(email_target='target@email.com', chart=chart, report_type=ReportScheduleType.ALERT, database=example_database, sql=param_config[request.param]['sql'], validator_type=param_config[request.param]['validator_type'], validator_config_json=param_config[request.param]['validator_config_json'])
            yield report_schedule
            cleanup_report_schedule(report_schedule)

@pytest.fixture(params=['alert1', 'alert2'])
def create_mul_alert_email_chart(request):
    if False:
        return 10
    param_config = {'alert1': {'sql': 'SELECT first, second from test_table', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": "<", "threshold": 10}'}, 'alert2': {'sql': 'SELECT first from test_table', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": "<", "threshold": 10}'}}
    with app.app_context():
        chart = db.session.query(Slice).first()
        example_database = get_example_database()
        with create_test_table_context(example_database):
            report_schedule = create_report_notification(email_target='target@email.com', chart=chart, report_type=ReportScheduleType.ALERT, database=example_database, sql=param_config[request.param]['sql'], validator_type=param_config[request.param]['validator_type'], validator_config_json=param_config[request.param]['validator_config_json'])
            yield report_schedule
            cleanup_report_schedule(report_schedule)

@pytest.fixture(params=['alert1', 'alert2'])
def create_invalid_sql_alert_email_chart(request):
    if False:
        while True:
            i = 10
    param_config = {'alert1': {'sql': "SELECT 'string' ", 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": "<", "threshold": 10}'}, 'alert2': {'sql': 'SELECT first from foo_table', 'validator_type': ReportScheduleValidatorType.OPERATOR, 'validator_config_json': '{"op": "<", "threshold": 10}'}}
    with app.app_context():
        chart = db.session.query(Slice).first()
        example_database = get_example_database()
        with create_test_table_context(example_database):
            report_schedule = create_report_notification(email_target='target@email.com', chart=chart, report_type=ReportScheduleType.ALERT, database=example_database, sql=param_config[request.param]['sql'], validator_type=param_config[request.param]['validator_type'], validator_config_json=param_config[request.param]['validator_config_json'], grace_period=60 * 60)
            yield report_schedule
            cleanup_report_schedule(report_schedule)

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_chart')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def test_email_chart_report_schedule(screenshot_mock, email_mock, create_report_email_chart):
    if False:
        while True:
            i = 10
    '\n    ExecuteReport Command: Test chart email report schedule with screenshot\n    '
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart.id, datetime.utcnow()).run()
        notification_targets = get_target_from_report_schedule(create_report_email_chart)
        assert f'<a href="http://0.0.0.0:8080/explore/?form_data=%7B%22slice_id%22:+{create_report_email_chart.chart.id}%7D&force=false">Explore in Superset</a>' in email_mock.call_args[0][2]
        assert email_mock.call_args[0][0] == notification_targets[0]
        smtp_images = email_mock.call_args[1]['images']
        assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
        assert_log(ReportState.SUCCESS)

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_chart_alpha_owner')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def test_email_chart_report_schedule_alpha_owner(screenshot_mock, email_mock, create_report_email_chart_alpha_owner):
    if False:
        i = 10
        return i + 15
    '\n    ExecuteReport Command: Test chart email report schedule with screenshot\n    executed as the chart owner\n    '
    config_key = 'ALERT_REPORTS_EXECUTE_AS'
    original_config_value = app.config[config_key]
    app.config[config_key] = [ExecutorType.OWNER]
    username = ''

    def _screenshot_side_effect(user: User) -> Optional[bytes]:
        if False:
            for i in range(10):
                print('nop')
        nonlocal username
        username = user.username
        return SCREENSHOT_FILE
    screenshot_mock.side_effect = _screenshot_side_effect
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_alpha_owner.id, datetime.utcnow()).run()
        notification_targets = get_target_from_report_schedule(create_report_email_chart_alpha_owner)
        assert username == 'alpha'
        assert f'<a href="http://0.0.0.0:8080/explore/?form_data=%7B%22slice_id%22:+{create_report_email_chart_alpha_owner.chart.id}%7D&force=false">Explore in Superset</a>' in email_mock.call_args[0][2]
        assert email_mock.call_args[0][0] == notification_targets[0]
        smtp_images = email_mock.call_args[1]['images']
        assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
        assert_log(ReportState.SUCCESS)
    app.config[config_key] = original_config_value

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_chart_force_screenshot')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def test_email_chart_report_schedule_force_screenshot(screenshot_mock, email_mock, create_report_email_chart_force_screenshot):
    if False:
        while True:
            i = 10
    '\n    ExecuteReport Command: Test chart email report schedule with screenshot\n\n    In this test ``force_screenshot`` is true, and the screenshot URL should\n    reflect that.\n    '
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_force_screenshot.id, datetime.utcnow()).run()
        notification_targets = get_target_from_report_schedule(create_report_email_chart_force_screenshot)
        assert f'<a href="http://0.0.0.0:8080/explore/?form_data=%7B%22slice_id%22:+{create_report_email_chart_force_screenshot.chart.id}%7D&force=true">Explore in Superset</a>' in email_mock.call_args[0][2]
        assert email_mock.call_args[0][0] == notification_targets[0]
        smtp_images = email_mock.call_args[1]['images']
        assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
        assert_log(ReportState.SUCCESS)

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_alert_email_chart')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def test_email_chart_alert_schedule(screenshot_mock, email_mock, create_alert_email_chart):
    if False:
        for i in range(10):
            print('nop')
    '\n    ExecuteReport Command: Test chart email alert schedule with screenshot\n    '
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
        notification_targets = get_target_from_report_schedule(create_alert_email_chart)
        assert f'<a href="http://0.0.0.0:8080/explore/?form_data=%7B%22slice_id%22:+{create_alert_email_chart.chart.id}%7D&force=true">Explore in Superset</a>' in email_mock.call_args[0][2]
        assert email_mock.call_args[0][0] == notification_targets[0]
        smtp_images = email_mock.call_args[1]['images']
        assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
        assert_log(ReportState.SUCCESS)

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_chart')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def test_email_chart_report_dry_run(screenshot_mock, email_mock, create_report_email_chart):
    if False:
        print('Hello World!')
    '\n    ExecuteReport Command: Test chart email report schedule dry run\n    '
    screenshot_mock.return_value = SCREENSHOT_FILE
    app.config['ALERT_REPORTS_NOTIFICATION_DRY_RUN'] = True
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart.id, datetime.utcnow()).run()
        email_mock.assert_not_called()
    app.config['ALERT_REPORTS_NOTIFICATION_DRY_RUN'] = False

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_chart_with_csv')
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.csv.get_chart_csv_data')
def test_email_chart_report_schedule_with_csv(csv_mock, email_mock, mock_open, mock_urlopen, create_report_email_chart_with_csv):
    if False:
        return 10
    '\n    ExecuteReport Command: Test chart email report schedule with CSV\n    '
    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.return_value = 200
    response.read.return_value = CSV_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_with_csv.id, datetime.utcnow()).run()
        notification_targets = get_target_from_report_schedule(create_report_email_chart_with_csv)
        assert f'<a href="http://0.0.0.0:8080/explore/?form_data=%7B%22slice_id%22:+{create_report_email_chart_with_csv.chart.id}%7D&force=false">Explore in Superset</a>' in email_mock.call_args[0][2]
        assert email_mock.call_args[0][0] == notification_targets[0]
        smtp_images = email_mock.call_args[1]['data']
        assert smtp_images[list(smtp_images.keys())[0]] == CSV_FILE
        assert_log(ReportState.SUCCESS)

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_chart_with_csv_no_query_context')
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.csv.get_chart_csv_data')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def test_email_chart_report_schedule_with_csv_no_query_context(screenshot_mock, csv_mock, email_mock, mock_open, mock_urlopen, create_report_email_chart_with_csv_no_query_context):
    if False:
        for i in range(10):
            print('nop')
    '\n    ExecuteReport Command: Test chart email report schedule with CSV (no query context)\n    '
    screenshot_mock.return_value = SCREENSHOT_FILE
    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.return_value = 200
    response.read.return_value = CSV_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_with_csv_no_query_context.id, datetime.utcnow()).run()
        screenshot_mock.assert_called_once()

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_chart_with_text')
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.csv.get_chart_dataframe')
def test_email_chart_report_schedule_with_text(dataframe_mock, email_mock, mock_open, mock_urlopen, create_report_email_chart_with_text):
    if False:
        print('Hello World!')
    '\n    ExecuteReport Command: Test chart email report schedule with text\n    '
    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.return_value = 200
    response.read.return_value = json.dumps({'result': [{'data': {'t1': {0: 'c11', 1: 'c21'}, 't2': {0: 'c12', 1: 'c22'}, 't3__sum': {0: 'c13', 1: 'c23'}}, 'colnames': [('t1',), ('t2',), ('t3__sum',)], 'indexnames': [(0,), (1,)], 'coltypes': [1, 1]}]}).encode('utf-8')
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_with_text.id, datetime.utcnow()).run()
        table_html = '<table border="1" class="dataframe">\n  <thead>\n    <tr>\n      <th></th>\n      <th>t1</th>\n      <th>t2</th>\n      <th>t3__sum</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>c11</td>\n      <td>c12</td>\n      <td>c13</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>c21</td>\n      <td>c22</td>\n      <td>c23</td>\n    </tr>\n  </tbody>\n</table>'
        assert table_html in email_mock.call_args[0][2]
        assert_log(ReportState.SUCCESS)
    dt = datetime(2022, 1, 1).replace(tzinfo=timezone.utc)
    ts = datetime.timestamp(dt) * 1000
    response.read.return_value = json.dumps({'result': [{'data': {'t1': {0: 'c11', 1: 'c21'}, 't2__date': {0: ts, 1: ts}, 't3__sum': {0: 'c13', 1: 'c23'}}, 'colnames': [('t1',), ('t2__date',), ('t3__sum',)], 'indexnames': [(0,), (1,)], 'coltypes': [1, 2]}]}).encode('utf-8')
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_with_text.id, datetime.utcnow()).run()
        table_html = '<table border="1" class="dataframe">\n  <thead>\n    <tr>\n      <th></th>\n      <th>t1</th>\n      <th>t2__date</th>\n      <th>t3__sum</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>c11</td>\n      <td>2022-01-01</td>\n      <td>c13</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>c21</td>\n      <td>2022-01-01</td>\n      <td>c23</td>\n    </tr>\n  </tbody>\n</table>'
        assert table_html in email_mock.call_args[0][2]

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_dashboard')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.DashboardScreenshot.get_screenshot')
def test_email_dashboard_report_schedule(screenshot_mock, email_mock, create_report_email_dashboard):
    if False:
        while True:
            i = 10
    '\n    ExecuteReport Command: Test dashboard email report schedule\n    '
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        with patch.object(current_app.config['STATS_LOGGER'], 'gauge') as statsd_mock:
            AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_dashboard.id, datetime.utcnow()).run()
            notification_targets = get_target_from_report_schedule(create_report_email_dashboard)
            assert email_mock.call_args[0][0] == notification_targets[0]
            smtp_images = email_mock.call_args[1]['images']
            assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
            assert_log(ReportState.SUCCESS)
            statsd_mock.assert_called_once_with('reports.email.send.ok', 1)

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_dashboard_force_screenshot')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.DashboardScreenshot.get_screenshot')
def test_email_dashboard_report_schedule_force_screenshot(screenshot_mock, email_mock, create_report_email_dashboard_force_screenshot):
    if False:
        return 10
    '\n    ExecuteReport Command: Test dashboard email report schedule\n    '
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_dashboard_force_screenshot.id, datetime.utcnow()).run()
        notification_targets = get_target_from_report_schedule(create_report_email_dashboard_force_screenshot)
        assert email_mock.call_args[0][0] == notification_targets[0]
        smtp_images = email_mock.call_args[1]['images']
        assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
        assert_log(ReportState.SUCCESS)

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_slack_chart')
@patch('superset.reports.notifications.slack.WebClient.files_upload')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def test_slack_chart_report_schedule(screenshot_mock, file_upload_mock, create_report_slack_chart):
    if False:
        for i in range(10):
            print('nop')
    '\n    ExecuteReport Command: Test chart slack report schedule\n    '
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        with patch.object(current_app.config['STATS_LOGGER'], 'gauge') as statsd_mock:
            AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart.id, datetime.utcnow()).run()
            notification_targets = get_target_from_report_schedule(create_report_slack_chart)
            assert file_upload_mock.call_args[1]['channels'] == notification_targets[0]
            assert file_upload_mock.call_args[1]['file'] == SCREENSHOT_FILE
            assert_log(ReportState.SUCCESS)
            statsd_mock.assert_called_once_with('reports.slack.send.ok', 1)

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_slack_chart')
@patch('superset.reports.notifications.slack.WebClient')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def test_slack_chart_report_schedule_with_errors(screenshot_mock, web_client_mock, create_report_slack_chart):
    if False:
        i = 10
        return i + 15
    '\n    ExecuteReport Command: Test that all slack errors will\n    properly log something\n    '
    screenshot_mock.return_value = SCREENSHOT_FILE
    slack_errors = [BotUserAccessError(), SlackRequestError(), SlackClientConfigurationError(), SlackObjectFormationError(), SlackTokenRotationError(api_error='foo'), SlackClientNotConnectedError(), SlackClientError(), SlackApiError(message='foo', response='bar')]
    for (idx, er) in enumerate(slack_errors):
        web_client_mock.side_effect = er
        with pytest.raises(ReportScheduleClientErrorsException):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart.id, datetime.utcnow()).run()
        db.session.commit()
    notification_logs_count = get_notification_error_sent_count(create_report_slack_chart)
    error_logs = get_error_logs_query(create_report_slack_chart)
    assert error_logs.count() == (len(slack_errors) + notification_logs_count) * 2
    assert len([log.error_message for log in error_logs]) == error_logs.count()

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_slack_chart_with_csv')
@patch('superset.reports.notifications.slack.WebClient.files_upload')
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.utils.csv.get_chart_csv_data')
def test_slack_chart_report_schedule_with_csv(csv_mock, mock_open, mock_urlopen, file_upload_mock, create_report_slack_chart_with_csv):
    if False:
        print('Hello World!')
    '\n    ExecuteReport Command: Test chart slack report schedule with CSV\n    '
    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.return_value = 200
    response.read.return_value = CSV_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart_with_csv.id, datetime.utcnow()).run()
        notification_targets = get_target_from_report_schedule(create_report_slack_chart_with_csv)
        assert file_upload_mock.call_args[1]['channels'] == notification_targets[0]
        assert file_upload_mock.call_args[1]['file'] == CSV_FILE
        assert_log(ReportState.SUCCESS)

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_slack_chart_with_text')
@patch('superset.reports.notifications.slack.WebClient.chat_postMessage')
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.utils.csv.get_chart_dataframe')
def test_slack_chart_report_schedule_with_text(dataframe_mock, mock_open, mock_urlopen, post_message_mock, create_report_slack_chart_with_text):
    if False:
        return 10
    '\n    ExecuteReport Command: Test chart slack report schedule with text\n    '
    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.return_value = 200
    response.read.return_value = json.dumps({'result': [{'data': {'t1': {0: 'c11', 1: 'c21'}, 't2': {0: 'c12', 1: 'c22'}, 't3__sum': {0: 'c13', 1: 'c23'}}, 'colnames': [('t1',), ('t2',), ('t3__sum',)], 'indexnames': [(0,), (1,)]}]}).encode('utf-8')
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart_with_text.id, datetime.utcnow()).run()
        table_markdown = '|    | t1   | t2   | t3__sum   |\n|---:|:-----|:-----|:----------|\n|  0 | c11  | c12  | c13       |\n|  1 | c21  | c22  | c23       |'
        assert table_markdown in post_message_mock.call_args[1]['text']
        assert f'<http://0.0.0.0:8080/explore/?form_data=%7B%22slice_id%22:+{create_report_slack_chart_with_text.chart.id}%7D&force=false|Explore in Superset>' in post_message_mock.call_args[1]['text']
        assert_log(ReportState.SUCCESS)

@pytest.mark.usefixtures('create_report_slack_chart')
def test_report_schedule_not_found(create_report_slack_chart):
    if False:
        print('Hello World!')
    '\n    ExecuteReport Command: Test report schedule not found\n    '
    max_id = db.session.query(func.max(ReportSchedule.id)).scalar()
    with pytest.raises(ReportScheduleNotFoundError):
        AsyncExecuteReportScheduleCommand(TEST_ID, max_id + 1, datetime.utcnow()).run()

@pytest.mark.usefixtures('create_report_slack_chart_working')
def test_report_schedule_working(create_report_slack_chart_working):
    if False:
        for i in range(10):
            print('nop')
    '\n    ExecuteReport Command: Test report schedule still working\n    '
    with freeze_time('2020-01-01T00:00:00Z'):
        with pytest.raises(ReportSchedulePreviousWorkingError):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart_working.id, datetime.utcnow()).run()
        assert_log(ReportState.WORKING, error_message=ReportSchedulePreviousWorkingError.message)
        assert create_report_slack_chart_working.last_state == ReportState.WORKING

@pytest.mark.usefixtures('create_report_slack_chart_working')
def test_report_schedule_working_timeout(create_report_slack_chart_working):
    if False:
        return 10
    '\n    ExecuteReport Command: Test report schedule still working but should timed out\n    '
    current_time = create_report_slack_chart_working.last_eval_dttm + timedelta(seconds=create_report_slack_chart_working.working_timeout + 1)
    with freeze_time(current_time):
        with pytest.raises(ReportScheduleWorkingTimeoutError):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart_working.id, datetime.utcnow()).run()
    db.session.commit()
    logs = db.session.query(ReportExecutionLog).all()
    assert len(logs) == 2
    assert ReportScheduleWorkingTimeoutError.message in [log.error_message for log in logs]
    assert create_report_slack_chart_working.last_state == ReportState.ERROR

@pytest.mark.usefixtures('create_alert_slack_chart_success')
def test_report_schedule_success_grace(create_alert_slack_chart_success):
    if False:
        for i in range(10):
            print('nop')
    '\n    ExecuteReport Command: Test report schedule on success to grace\n    '
    current_time = create_alert_slack_chart_success.last_eval_dttm + timedelta(seconds=create_alert_slack_chart_success.grace_period - 10)
    with freeze_time(current_time):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_slack_chart_success.id, datetime.utcnow()).run()
    db.session.commit()
    assert create_alert_slack_chart_success.last_state == ReportState.GRACE

@pytest.mark.usefixtures('create_alert_slack_chart_grace')
@patch('superset.reports.notifications.slack.WebClient.files_upload')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def test_report_schedule_success_grace_end(screenshot_mock, file_upload_mock, create_alert_slack_chart_grace):
    if False:
        i = 10
        return i + 15
    '\n    ExecuteReport Command: Test report schedule on grace to noop\n    '
    screenshot_mock.return_value = SCREENSHOT_FILE
    current_time = create_alert_slack_chart_grace.last_eval_dttm + timedelta(seconds=create_alert_slack_chart_grace.grace_period + 1)
    with freeze_time(current_time):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_slack_chart_grace.id, datetime.utcnow()).run()
    db.session.commit()
    assert create_alert_slack_chart_grace.last_state == ReportState.SUCCESS

@pytest.mark.usefixtures('create_alert_email_chart')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def test_alert_limit_is_applied(screenshot_mock, email_mock, create_alert_email_chart):
    if False:
        while True:
            i = 10
    '\n    ExecuteReport Command: Test that all alerts apply a SQL limit to stmts\n    '
    screenshot_mock.return_value = SCREENSHOT_FILE
    with patch.object(create_alert_email_chart.database.db_engine_spec, 'execute', return_value=None) as execute_mock:
        with patch.object(create_alert_email_chart.database.db_engine_spec, 'fetch_data', return_value=None) as fetch_data_mock:
            AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
            assert 'LIMIT 2' in execute_mock.call_args[0][1]

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_dashboard')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.DashboardScreenshot.get_screenshot')
def test_email_dashboard_report_fails(screenshot_mock, email_mock, create_report_email_dashboard):
    if False:
        print('Hello World!')
    '\n    ExecuteReport Command: Test dashboard email report schedule notification fails\n    '
    from smtplib import SMTPException
    screenshot_mock.return_value = SCREENSHOT_FILE
    email_mock.side_effect = SMTPException('Could not connect to SMTP XPTO')
    with pytest.raises(ReportScheduleSystemErrorsException):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_dashboard.id, datetime.utcnow()).run()
    assert_log(ReportState.ERROR, error_message='Could not connect to SMTP XPTO')

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_dashboard')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.DashboardScreenshot.get_screenshot')
def test_email_dashboard_report_fails_uncaught_exception(screenshot_mock, email_mock, create_report_email_dashboard):
    if False:
        while True:
            i = 10
    '\n    ExecuteReport Command: Test dashboard email report schedule notification fails\n    and logs with uncaught exception\n    '
    from smtplib import SMTPException
    screenshot_mock.return_value = SCREENSHOT_FILE
    email_mock.side_effect = Exception('Uncaught exception')
    with pytest.raises(Exception):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_dashboard.id, datetime.utcnow()).run()
    assert_log(ReportState.ERROR, error_message='Uncaught exception')

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_alert_email_chart')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
@patch.dict('superset.extensions.feature_flag_manager._feature_flags', ALERTS_ATTACH_REPORTS=True)
def test_slack_chart_alert(screenshot_mock, email_mock, create_alert_email_chart):
    if False:
        i = 10
        return i + 15
    '\n    ExecuteReport Command: Test chart slack alert\n    '
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
        notification_targets = get_target_from_report_schedule(create_alert_email_chart)
        assert email_mock.call_args[0][0] == notification_targets[0]
        smtp_images = email_mock.call_args[1]['images']
        assert smtp_images[list(smtp_images.keys())[0]] == SCREENSHOT_FILE
        assert_log(ReportState.SUCCESS)

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_alert_email_chart')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch.dict('superset.extensions.feature_flag_manager._feature_flags', ALERTS_ATTACH_REPORTS=False)
def test_slack_chart_alert_no_attachment(email_mock, create_alert_email_chart):
    if False:
        print('Hello World!')
    '\n    ExecuteReport Command: Test chart slack alert\n    '
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
        notification_targets = get_target_from_report_schedule(create_alert_email_chart)
        assert email_mock.call_args[0][0] == notification_targets[0]
        assert email_mock.call_args[1]['images'] == {}
        assert_log(ReportState.SUCCESS)

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_slack_chart')
@patch('superset.reports.notifications.slack.WebClient')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def test_slack_token_callable_chart_report(screenshot_mock, slack_client_mock_class, create_report_slack_chart):
    if False:
        while True:
            i = 10
    '\n    ExecuteReport Command: Test chart slack alert (slack token callable)\n    '
    slack_client_mock_class.return_value = Mock()
    app.config['SLACK_API_TOKEN'] = Mock(return_value='cool_code')
    screenshot_mock.return_value = SCREENSHOT_FILE
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_slack_chart.id, datetime.utcnow()).run()
        app.config['SLACK_API_TOKEN'].assert_called_once()
        assert slack_client_mock_class.called_with(token='cool_code', proxy='')
        assert_log(ReportState.SUCCESS)

@pytest.mark.usefixtures('create_no_alert_email_chart')
def test_email_chart_no_alert(create_no_alert_email_chart):
    if False:
        while True:
            i = 10
    '\n    ExecuteReport Command: Test chart email no alert\n    '
    with freeze_time('2020-01-01T00:00:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_no_alert_email_chart.id, datetime.utcnow()).run()
    assert_log(ReportState.NOOP)

@pytest.mark.usefixtures('create_mul_alert_email_chart')
def test_email_mul_alert(create_mul_alert_email_chart):
    if False:
        print('Hello World!')
    '\n    ExecuteReport Command: Test chart email multiple rows\n    '
    with freeze_time('2020-01-01T00:00:00Z'):
        with pytest.raises((AlertQueryMultipleRowsError, AlertQueryMultipleColumnsError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_mul_alert_email_chart.id, datetime.utcnow()).run()

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_alert_email_chart')
@patch('superset.reports.notifications.email.send_email_smtp')
def test_soft_timeout_alert(email_mock, create_alert_email_chart):
    if False:
        while True:
            i = 10
    '\n    ExecuteReport Command: Test soft timeout on alert queries\n    '
    from celery.exceptions import SoftTimeLimitExceeded
    from superset.reports.commands.exceptions import AlertQueryTimeout
    with patch.object(create_alert_email_chart.database.db_engine_spec, 'execute', return_value=None) as execute_mock:
        execute_mock.side_effect = SoftTimeLimitExceeded()
        with pytest.raises(AlertQueryTimeout):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
    notification_targets = get_target_from_report_schedule(create_alert_email_chart)
    assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
    assert_log(ReportState.ERROR, error_message='A timeout occurred while executing the query.')

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_alert_email_chart')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
@patch.dict('superset.extensions.feature_flag_manager._feature_flags', ALERTS_ATTACH_REPORTS=True)
def test_soft_timeout_screenshot(screenshot_mock, email_mock, create_alert_email_chart):
    if False:
        for i in range(10):
            print('nop')
    '\n    ExecuteReport Command: Test soft timeout on screenshot\n    '
    from celery.exceptions import SoftTimeLimitExceeded
    screenshot_mock.side_effect = SoftTimeLimitExceeded()
    with pytest.raises(ReportScheduleScreenshotTimeout):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
    assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
    assert_log(ReportState.ERROR, error_message='A timeout occurred while taking a screenshot.')

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_chart_with_csv')
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.csv.get_chart_csv_data')
def test_soft_timeout_csv(csv_mock, email_mock, mock_open, mock_urlopen, create_report_email_chart_with_csv):
    if False:
        print('Hello World!')
    '\n    ExecuteReport Command: Test fail on generating csv\n    '
    from celery.exceptions import SoftTimeLimitExceeded
    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.side_effect = SoftTimeLimitExceeded()
    with pytest.raises(ReportScheduleCsvTimeout):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_with_csv.id, datetime.utcnow()).run()
    notification_targets = get_target_from_report_schedule(create_report_email_chart_with_csv)
    assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
    assert_log(ReportState.ERROR, error_message='A timeout occurred while generating a csv.')

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_chart_with_csv')
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.csv.get_chart_csv_data')
def test_generate_no_csv(csv_mock, email_mock, mock_open, mock_urlopen, create_report_email_chart_with_csv):
    if False:
        while True:
            i = 10
    '\n    ExecuteReport Command: Test fail on generating csv\n    '
    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.return_value = 200
    response.read.return_value = None
    with pytest.raises(ReportScheduleCsvFailedError):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_with_csv.id, datetime.utcnow()).run()
    notification_targets = get_target_from_report_schedule(create_report_email_chart_with_csv)
    assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
    assert_log(ReportState.ERROR, error_message='Report Schedule execution failed when generating a csv.')

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_chart')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def test_fail_screenshot(screenshot_mock, email_mock, create_report_email_chart):
    if False:
        print('Hello World!')
    '\n    ExecuteReport Command: Test soft timeout on screenshot\n    '
    from celery.exceptions import SoftTimeLimitExceeded
    from superset.reports.commands.exceptions import AlertQueryTimeout
    screenshot_mock.side_effect = Exception('Unexpected error')
    with pytest.raises(ReportScheduleScreenshotFailedError):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart.id, datetime.utcnow()).run()
    notification_targets = get_target_from_report_schedule(create_report_email_chart)
    assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
    assert_log(ReportState.ERROR, error_message='Failed taking a screenshot Unexpected error')

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_chart_with_csv')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.csv.urllib.request.urlopen')
@patch('superset.utils.csv.urllib.request.OpenerDirector.open')
@patch('superset.utils.csv.get_chart_csv_data')
def test_fail_csv(csv_mock, mock_open, mock_urlopen, email_mock, create_report_email_chart_with_csv):
    if False:
        print('Hello World!')
    '\n    ExecuteReport Command: Test error on csv\n    '
    response = Mock()
    mock_open.return_value = response
    mock_urlopen.return_value = response
    mock_urlopen.return_value.getcode.return_value = 500
    with pytest.raises(ReportScheduleCsvFailedError):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_report_email_chart_with_csv.id, datetime.utcnow()).run()
    get_target_from_report_schedule(create_report_email_chart_with_csv)
    assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
    assert_log(ReportState.ERROR, error_message='Failed generating csv <urlopen error 500>')

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_alert_email_chart')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch.dict('superset.extensions.feature_flag_manager._feature_flags', ALERTS_ATTACH_REPORTS=False)
def test_email_disable_screenshot(email_mock, create_alert_email_chart):
    if False:
        return 10
    '\n    ExecuteReport Command: Test soft timeout on screenshot\n    '
    AsyncExecuteReportScheduleCommand(TEST_ID, create_alert_email_chart.id, datetime.utcnow()).run()
    notification_targets = get_target_from_report_schedule(create_alert_email_chart)
    assert email_mock.call_args[0][0] == notification_targets[0]
    assert email_mock.call_args[1]['images'] == {}
    assert_log(ReportState.SUCCESS)

@pytest.mark.usefixtures('create_invalid_sql_alert_email_chart')
@patch('superset.reports.notifications.email.send_email_smtp')
def test_invalid_sql_alert(email_mock, create_invalid_sql_alert_email_chart):
    if False:
        print('Hello World!')
    '\n    ExecuteReport Command: Test alert with invalid SQL statements\n    '
    with freeze_time('2020-01-01T00:00:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
        assert_log(ReportState.ERROR)

@pytest.mark.usefixtures('create_invalid_sql_alert_email_chart')
@patch('superset.reports.notifications.email.send_email_smtp')
def test_grace_period_error(email_mock, create_invalid_sql_alert_email_chart):
    if False:
        print('Hello World!')
    '\n    ExecuteReport Command: Test alert grace period on error\n    '
    with freeze_time('2020-01-01T00:00:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
        assert email_mock.call_args[0][0] == DEFAULT_OWNER_EMAIL
        assert get_notification_error_sent_count(create_invalid_sql_alert_email_chart) == 1
    with freeze_time('2020-01-01T00:30:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
        assert get_notification_error_sent_count(create_invalid_sql_alert_email_chart) == 1
    with freeze_time('2020-01-01T01:30:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
        assert get_notification_error_sent_count(create_invalid_sql_alert_email_chart) == 2

@pytest.mark.usefixtures('create_invalid_sql_alert_email_chart')
@patch('superset.reports.notifications.email.send_email_smtp')
@patch('superset.utils.screenshots.ChartScreenshot.get_screenshot')
def test_grace_period_error_flap(screenshot_mock, email_mock, create_invalid_sql_alert_email_chart):
    if False:
        i = 10
        return i + 15
    '\n    ExecuteReport Command: Test alert grace period on error\n    '
    with freeze_time('2020-01-01T00:00:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
        assert get_notification_error_sent_count(create_invalid_sql_alert_email_chart) == 1
    with freeze_time('2020-01-01T00:30:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
        assert get_notification_error_sent_count(create_invalid_sql_alert_email_chart) == 1
    create_invalid_sql_alert_email_chart.sql = 'SELECT 1 AS metric'
    create_invalid_sql_alert_email_chart.grace_period = 0
    db.session.merge(create_invalid_sql_alert_email_chart)
    db.session.commit()
    with freeze_time('2020-01-01T00:31:00Z'):
        AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
    create_invalid_sql_alert_email_chart.sql = "SELECT 'first'"
    create_invalid_sql_alert_email_chart.grace_period = 10
    db.session.merge(create_invalid_sql_alert_email_chart)
    db.session.commit()
    with freeze_time('2020-01-01T00:32:00Z'):
        with pytest.raises((AlertQueryError, AlertQueryInvalidTypeError)):
            AsyncExecuteReportScheduleCommand(TEST_ID, create_invalid_sql_alert_email_chart.id, datetime.utcnow()).run()
        db.session.commit()
        assert get_notification_error_sent_count(create_invalid_sql_alert_email_chart) == 2

@pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'create_report_email_dashboard')
@patch('superset.daos.report.ReportScheduleDAO.bulk_delete_logs')
def test_prune_log_soft_time_out(bulk_delete_logs, create_report_email_dashboard):
    if False:
        print('Hello World!')
    from celery.exceptions import SoftTimeLimitExceeded
    bulk_delete_logs.side_effect = SoftTimeLimitExceeded()
    with pytest.raises(SoftTimeLimitExceeded) as excinfo:
        AsyncPruneReportScheduleLogCommand().run()
    assert str(excinfo.value) == 'SoftTimeLimitExceeded()'

@patch('superset.reports.commands.execute.logger')
@patch('superset.reports.commands.execute.create_notification')
def test__send_with_client_errors(notification_mock, logger_mock):
    if False:
        for i in range(10):
            print('nop')
    notification_content = 'I am some content'
    recipients = ['test@foo.com']
    notification_mock.return_value.send.side_effect = NotificationParamException()
    with pytest.raises(ReportScheduleClientErrorsException) as excinfo:
        BaseReportState._send(BaseReportState, notification_content, recipients)
    assert excinfo.errisinstance(SupersetException)
    logger_mock.warning.assert_called_with("SupersetError(message='', error_type=<SupersetErrorType.REPORT_NOTIFICATION_ERROR: 'REPORT_NOTIFICATION_ERROR'>, level=<ErrorLevel.WARNING: 'warning'>, extra=None)")

@patch('superset.reports.commands.execute.logger')
@patch('superset.reports.commands.execute.create_notification')
def test__send_with_multiple_errors(notification_mock, logger_mock):
    if False:
        print('Hello World!')
    notification_content = 'I am some content'
    recipients = ['test@foo.com', 'test2@bar.com']
    notification_mock.return_value.send.side_effect = [NotificationParamException(), NotificationError()]
    with pytest.raises(ReportScheduleSystemErrorsException) as excinfo:
        BaseReportState._send(BaseReportState, notification_content, recipients)
    assert excinfo.errisinstance(SupersetException)
    logger_mock.warning.assert_has_calls([call("SupersetError(message='', error_type=<SupersetErrorType.REPORT_NOTIFICATION_ERROR: 'REPORT_NOTIFICATION_ERROR'>, level=<ErrorLevel.WARNING: 'warning'>, extra=None)"), call("SupersetError(message='', error_type=<SupersetErrorType.REPORT_NOTIFICATION_ERROR: 'REPORT_NOTIFICATION_ERROR'>, level=<ErrorLevel.ERROR: 'error'>, extra=None)")])

@patch('superset.reports.commands.execute.logger')
@patch('superset.reports.commands.execute.create_notification')
def test__send_with_server_errors(notification_mock, logger_mock):
    if False:
        i = 10
        return i + 15
    notification_content = 'I am some content'
    recipients = ['test@foo.com']
    notification_mock.return_value.send.side_effect = NotificationError()
    with pytest.raises(ReportScheduleSystemErrorsException) as excinfo:
        BaseReportState._send(BaseReportState, notification_content, recipients)
    assert excinfo.errisinstance(SupersetException)
    logger_mock.warning.assert_called_with("SupersetError(message='', error_type=<SupersetErrorType.REPORT_NOTIFICATION_ERROR: 'REPORT_NOTIFICATION_ERROR'>, level=<ErrorLevel.ERROR: 'error'>, extra=None)")