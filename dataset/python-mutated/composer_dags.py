"""Standalone script to pause/unpause all the dags in the specific environment."""
from __future__ import annotations
import argparse
import json
import logging
import re
import subprocess
import sys
from typing import Any
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DAG:
    """Provides necessary utils for Composer DAGs."""
    COMPOSER_AF_VERSION_RE = re.compile('composer-([0-9]+).([0-9]+).([0-9]+).*-airflow-([0-9]+).([0-9]+).([0-9]+).*')

    @staticmethod
    def get_list_of_dags(project_name: str, environment: str, location: str, sdk_endpoint: str, airflow_version: tuple[int]) -> list[str]:
        if False:
            i = 10
            return i + 15
        'Retrieves the list of dags for particular project.'
        sub_command = 'list_dags' if airflow_version < (2, 0, 0) else 'dags list'
        command = f'CLOUDSDK_API_ENDPOINT_OVERRIDES_COMPOSER={sdk_endpoint} gcloud composer environments run {environment} --project={project_name} --location={location} {sub_command}'
        command_output = DAG._run_shell_command_locally_once(command=command)[1]
        if airflow_version < (2, 0, 0):
            command_output_parsed = command_output.split()
            return command_output_parsed[command_output_parsed.index('DAGS') + 2:len(command_output_parsed) - 1]
        else:
            list_of_dags = []
            for line in command_output.split('\n'):
                if re.compile('[a-z_]+|[a-z]+|[a-z]+|[a-z_]+').findall(line):
                    list_of_dags.append(line.split()[0])
            return list_of_dags[1:-1]

    @staticmethod
    def _run_shell_command_locally_once(command: str, command_input: str=None, log_command: bool=True) -> tuple[int, str]:
        if False:
            i = 10
            return i + 15
        'Executes shell command and returns its output.'
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        if log_command:
            logger.info('Executing shell command: %s', command)
        (res, _) = p.communicate(input=command_input)
        if p.returncode:
            logged_command = f' "{command}"' if log_command else ''
            error_message = f'Failed to run shell command{logged_command}, details: {res}'
            logger.error(error_message)
            sys.exit(1)
        return (p.returncode, str(res.decode().strip('\n')))

    @staticmethod
    def pause_dag(project_name: str, environment: str, location: str, sdk_endpoint: str, dag_id: str, airflow_version: list[int]) -> str:
        if False:
            i = 10
            return i + 15
        'Pause specific DAG in the given environment.'
        sub_command = 'pause' if airflow_version < (2, 0, 0) else 'dags pause'
        command = f'CLOUDSDK_API_ENDPOINT_OVERRIDES_COMPOSER={sdk_endpoint} gcloud composer environments run {environment} --project={project_name} --location={location} {sub_command} -- {dag_id}'
        command_output = DAG._run_shell_command_locally_once(command=command)
        if command_output[0] == 1:
            logger.info(command_output[1])
            logger.info('Error pausing DAG %s, Retrying...', dag_id)
            command_output = DAG._run_shell_command_locally_once(command=command)
            if command_output[0] == 1:
                logger.info('Unable to pause DAG %s', dag_id)
        logger.info(command_output[1])

    @staticmethod
    def unpause_dag(project_name: str, environment: str, location: str, sdk_endpoint: str, dag_id: str, airflow_version: list[int]) -> str:
        if False:
            i = 10
            return i + 15
        'UnPause specific DAG in the given environment.'
        sub_command = 'unpause' if airflow_version < (2, 0, 0) else 'dags unpause'
        command = f'CLOUDSDK_API_ENDPOINT_OVERRIDES_COMPOSER={sdk_endpoint} gcloud composer environments run {environment} --project={project_name} --location={location} {sub_command} -- {dag_id}'
        command_output = DAG._run_shell_command_locally_once(command=command)
        if command_output[0] == 1:
            logger.info(command_output[1])
            logger.info('Error Unpausing DAG %s, Retrying...', dag_id)
            command_output = DAG._run_shell_command_locally_once(command=command)
            if command_output[0] == 1:
                logger.info('Unable to Unpause DAG %s', dag_id)
        logger.info(command_output[1])

    @staticmethod
    def describe_environment(project_name: str, environment: str, location: str, sdk_endpoint: str) -> Any:
        if False:
            return 10
        'Returns the given environment json object to parse necessary details.'
        logger.info('*** Fetching details of the environment: %s...', environment)
        command = f'CLOUDSDK_API_ENDPOINT_OVERRIDES_COMPOSER={sdk_endpoint} gcloud composer environments describe {environment} --project={project_name} --location={location} --format json'
        environment_json = json.loads(DAG._run_shell_command_locally_once(command)[1])
        logger.info('Environment Info:\n %s', environment_json['name'])
        return environment_json

def main(project_name: str, environment: str, location: str, operation: str, sdk_endpoint=str) -> int:
    if False:
        while True:
            i = 10
    logger.info('DAG Pause/UnPause Script for Cloud Composer')
    environment_info = DAG.describe_environment(project_name=project_name, environment=environment, location=location, sdk_endpoint=sdk_endpoint)
    versions = DAG.COMPOSER_AF_VERSION_RE.match(environment_info['config']['softwareConfig']['imageVersion']).groups()
    logger.info('Image version: %s', environment_info['config']['softwareConfig']['imageVersion'])
    airflow_version = (int(versions[3]), int(versions[4]), int(versions[5]))
    list_of_dags = DAG.get_list_of_dags(project_name=project_name, environment=environment, location=location, sdk_endpoint=sdk_endpoint, airflow_version=airflow_version)
    logger.info('List of dags : %s', list_of_dags)
    if operation == 'pause':
        for dag in list_of_dags:
            if dag == 'airflow_monitoring':
                continue
            DAG.pause_dag(project_name=project_name, environment=environment, location=location, sdk_endpoint=sdk_endpoint, dag_id=dag, airflow_version=airflow_version)
    else:
        for dag in list_of_dags:
            DAG.unpause_dag(project_name=project_name, environment=environment, location=location, sdk_endpoint=sdk_endpoint, dag_id=dag, airflow_version=airflow_version)
    return 0

def parse_arguments() -> dict[Any, Any]:
    if False:
        return 10
    'Parses command line parameters.'
    argument_parser = argparse.ArgumentParser(usage='Script to Pause/UnPause DAGs in Cloud Composer Environment \n')
    argument_parser.add_argument('--operation', type=str, choices=['pause', 'unpause'])
    argument_parser.add_argument('--project', type=str, required=True)
    argument_parser.add_argument('--environment', type=str, required=True)
    argument_parser.add_argument('--location', type=str, required=True)
    argument_parser.add_argument('--sdk_endpoint', type=str, default='https://composer.googleapis.com/', required=False)
    return argument_parser.parse_args()
if __name__ == '__main__':
    args = parse_arguments()
    exit(main(args.project, args.environment, args.location, args.operation, args.sdk_endpoint))