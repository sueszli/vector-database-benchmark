from __future__ import annotations
import os
from pathlib import Path
from unittest import mock
from airflow_breeze.commands.ci_commands import workflow_info
TEST_PR_INFO_DIR = Path(__file__).parent / 'test_pr_info_files'

def test_pr_info():
    if False:
        print('Hello World!')
    with mock.patch.dict(os.environ, {'AIRFLOW_SELF_HOSTED_RUNNER': ''}):
        json_string = (TEST_PR_INFO_DIR / 'pr_github_context.json').read_text()
        wi = workflow_info(json_string)
        assert wi.pull_request_labels == ['area:providers', 'area:dev-tools', 'area:logging', 'kind:documentation']
        assert wi.target_repo == 'apache/airflow'
        assert wi.head_repo == 'test/airflow'
        assert wi.event_name == 'pull_request'
        assert wi.pr_number == 26004
        assert wi.get_runs_on() == '["ubuntu-22.04"]'
        assert wi.is_canary_run() == 'false'
        assert wi.run_coverage() == 'false'

def test_push_info():
    if False:
        while True:
            i = 10
    with mock.patch.dict(os.environ, {'AIRFLOW_SELF_HOSTED_RUNNER': ''}):
        json_string = (TEST_PR_INFO_DIR / 'push_github_context.json').read_text()
        wi = workflow_info(json_string)
        assert wi.pull_request_labels == []
        assert wi.target_repo == 'apache/airflow'
        assert wi.head_repo == 'apache/airflow'
        assert wi.event_name == 'push'
        assert wi.pr_number is None
        assert wi.get_runs_on() == '["ubuntu-22.04"]'
        assert wi.is_canary_run() == 'true'
        assert wi.run_coverage() == 'true'

def test_schedule():
    if False:
        while True:
            i = 10
    with mock.patch.dict(os.environ, {'AIRFLOW_SELF_HOSTED_RUNNER': ''}):
        json_string = (TEST_PR_INFO_DIR / 'schedule_github_context.json').read_text()
        wi = workflow_info(json_string)
        assert wi.pull_request_labels == []
        assert wi.target_repo == 'apache/airflow'
        assert wi.head_repo == 'apache/airflow'
        assert wi.event_name == 'schedule'
        assert wi.pr_number is None
        assert wi.get_runs_on() == '["ubuntu-22.04"]'
        assert wi.is_canary_run() == 'false'
        assert wi.run_coverage() == 'false'

def test_runs_on_self_hosted():
    if False:
        i = 10
        return i + 15
    with mock.patch.dict(os.environ, {'AIRFLOW_SELF_HOSTED_RUNNER': 'true'}):
        json_string = (TEST_PR_INFO_DIR / 'simple_pr.json').read_text()
        wi = workflow_info(json_string)
        assert wi.pull_request_labels == ['another']
        assert wi.target_repo == 'apache/airflow'
        assert wi.head_repo == 'apache/airflow'
        assert wi.event_name == 'pull_request'
        assert wi.pr_number == 1234
        assert wi.get_runs_on() == '["self-hosted", "Linux", "X64"]'
        assert wi.is_canary_run() == 'false'
        assert wi.run_coverage() == 'false'

def test_runs_on_forced_public_runner():
    if False:
        i = 10
        return i + 15
    with mock.patch.dict(os.environ, {'AIRFLOW_SELF_HOSTED_RUNNER': 'true'}):
        json_string = (TEST_PR_INFO_DIR / 'self_hosted_forced_pr.json').read_text()
        wi = workflow_info(json_string)
        assert wi.pull_request_labels == ['use public runners', 'another']
        assert wi.target_repo == 'apache/airflow'
        assert wi.head_repo == 'apache/airflow'
        assert wi.event_name == 'pull_request'
        assert wi.pr_number == 1234
        assert wi.get_runs_on() == '["ubuntu-22.04"]'
        assert wi.is_canary_run() == 'false'
        assert wi.run_coverage() == 'false'

def test_runs_on_simple_pr_other_repo():
    if False:
        return 10
    with mock.patch.dict(os.environ, {'AIRFLOW_SELF_HOSTED_RUNNER': ''}):
        json_string = (TEST_PR_INFO_DIR / 'simple_pr_different_repo.json').read_text()
        wi = workflow_info(json_string)
        assert wi.pull_request_labels == ['another']
        assert wi.target_repo == 'apache/airflow'
        assert wi.head_repo == 'test/airflow'
        assert wi.event_name == 'pull_request'
        assert wi.pr_number == 1234
        assert wi.get_runs_on() == '["ubuntu-22.04"]'
        assert wi.is_canary_run() == 'false'
        assert wi.run_coverage() == 'false'

def test_runs_on_push_other_branch():
    if False:
        return 10
    with mock.patch.dict(os.environ, {'AIRFLOW_SELF_HOSTED_RUNNER': 'true'}):
        json_string = (TEST_PR_INFO_DIR / 'push_other_branch.json').read_text()
        wi = workflow_info(json_string)
        assert wi.pull_request_labels == []
        assert wi.target_repo == 'apache/airflow'
        assert wi.head_repo == 'apache/airflow'
        assert wi.event_name == 'push'
        assert wi.pr_number is None
        assert wi.get_runs_on() == '["self-hosted", "Linux", "X64"]'
        assert wi.is_canary_run() == 'false'
        assert wi.run_coverage() == 'false'

def test_runs_on_push_v_test_branch():
    if False:
        return 10
    with mock.patch.dict(os.environ, {'AIRFLOW_SELF_HOSTED_RUNNER': 'true'}):
        json_string = (TEST_PR_INFO_DIR / 'push_v_test_branch.json').read_text()
        wi = workflow_info(json_string)
        assert wi.pull_request_labels == []
        assert wi.target_repo == 'apache/airflow'
        assert wi.head_repo == 'apache/airflow'
        assert wi.event_name == 'push'
        assert wi.pr_number is None
        assert wi.get_runs_on() == '["self-hosted", "Linux", "X64"]'
        assert wi.is_canary_run() == 'true'
        assert wi.run_coverage() == 'false'