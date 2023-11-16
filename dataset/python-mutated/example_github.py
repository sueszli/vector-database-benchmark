from __future__ import annotations
import logging
import os
from datetime import datetime
from typing import Any
from github import GithubException
from airflow.exceptions import AirflowException
from airflow.models.dag import DAG
from airflow.providers.github.operators.github import GithubOperator
from airflow.providers.github.sensors.github import GithubSensor, GithubTagSensor
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
DAG_ID = 'example_github_operator'
with DAG(DAG_ID, start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    tag_sensor = GithubTagSensor(task_id='example_tag_sensor', tag_name='v1.0', repository_name='apache/airflow', timeout=60, poke_interval=10)

    def tag_checker(repo: Any, tag_name: str) -> bool | None:
        if False:
            print('Hello World!')
        result = None
        try:
            if repo is not None and tag_name is not None:
                all_tags = [x.name for x in repo.get_tags()]
                result = tag_name in all_tags
        except GithubException as github_error:
            raise AirflowException(f'Failed to execute GithubSensor, error: {github_error}')
        except Exception as e:
            raise AirflowException(f'GitHub operator error: {e}')
        return result
    github_sensor = GithubSensor(task_id='example_sensor', method_name='get_repo', method_params={'full_name_or_id': 'apache/airflow'}, result_processor=lambda repo: tag_checker(repo, 'v1.0'), timeout=60, poke_interval=10)
    github_list_repos = GithubOperator(task_id='github_list_repos', github_method='get_user', result_processor=lambda user: logging.info(list(user.get_repos())))
    list_repo_tags = GithubOperator(task_id='list_repo_tags', github_method='get_repo', github_method_args={'full_name_or_id': 'apache/airflow'}, result_processor=lambda repo: logging.info(list(repo.get_tags())))
from tests.system.utils import get_test_run
test_run = get_test_run(dag)