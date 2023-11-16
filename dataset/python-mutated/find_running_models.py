"""
Purpose: Displays a list of running Amazon Lookout for Vision
models across all accessible AWS Regions in the commercial
AWS partition. For accurate results, install the latest Boto3
client.
"""
import logging
import boto3
from boto3.session import Session
from botocore.exceptions import ClientError, EndpointConnectionError
logger = logging.getLogger(__name__)

def find_running_models_in_project(lfv_client, project_name):
    if False:
        i = 10
        return i + 15
    '\n    Gets a list of running models in a project.\n    :param lookoutvision_client: A Boto3 Amazon Lookout for Vision client.\n    param project_name: The name of the project that you want to use.\n    return: A list of running models. Empty if no models are\n    running in the project.\n    '
    logger.info('Finding running models in project: %s', project_name)
    running_models = []
    paginator = lfv_client.get_paginator('list_models')
    page_iterator = paginator.paginate(ProjectName=project_name)
    for page in page_iterator:
        for model in page['Models']:
            model_description = lfv_client.describe_model(ProjectName=project_name, ModelVersion=model['ModelVersion'])
            logger.info('Checking: %s', model_description['ModelDescription']['ModelArn'])
            if model_description['ModelDescription']['Status'] == 'HOSTED':
                running_model = {'Project': project_name, 'ARN': model_description['ModelDescription']['ModelArn'], 'Version': model_description['ModelDescription']['ModelVersion']}
                running_models.append(running_model)
                logger.info('Running model ARN: %s Version %s', model_description['ModelDescription']['ModelArn'], model_description['ModelDescription']['ModelVersion'])
    logger.info('Done finding running models in project: %s', project_name)
    return running_models

def display_running_models(running_model_regions):
    if False:
        while True:
            i = 10
    '\n    Displays running model information.\n    :param running_model_region: A list of AWS Regions\n    and models that are running within each AWS Region.\n    '
    count = 0
    if running_model_regions:
        print('Running models.\n')
        for region in running_model_regions:
            print(region['Region'])
            for model in region['Models']:
                print(f"  Project: {model['Project']}")
                print(f"  Version: {model['Version']}")
                print(f"  ARN: {model['ARN']}\n")
                count += 1
    print(f'There is {count} running model(s).')

def find_running_models(boto3_session):
    if False:
        i = 10
        return i + 15
    '\n    Finds the running Lookout for Vision models across all accessible\n    AWS Regions.\n\n    :param boto3_session A Boto3 session initialized with a credentials profile.\n    :return: A list of running models.\n    '
    running_models = []
    regions = boto3_session.get_available_regions(service_name='lookoutvision')
    for region in regions:
        logger.info('Checking %s', region)
        region_info = {}
        region_info['Region'] = region
        region_info['Models'] = []
        running_models_in_region = []
        lfv_client = boto3_session.client('lookoutvision', region_name=region)
        paginator = lfv_client.get_paginator('list_projects')
        page_iterator = paginator.paginate()
        for page in page_iterator:
            for project in page['Projects']:
                running_models_in_project = find_running_models_in_project(lfv_client, project['ProjectName'])
                for running_model in running_models_in_project:
                    running_models_in_region.append(running_model)
                region_info['Models'] = running_models_in_region
        if region_info['Models']:
            running_models.append(region_info)
    return running_models

def main():
    if False:
        for i in range(10):
            print('nop')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        session = boto3.Session(profile_name='lookoutvision-access')
        running_models = find_running_models(session)
        display_running_models(running_models)
    except TypeError as err:
        print("Couldn't get available AWS Regions: " + format(err))
    except ClientError as err:
        print('Service error occurred: ' + format(err))
    except EndpointConnectionError as err:
        logger.info('Problem calling endpoint: %s', format(err))
        raise
if __name__ == '__main__':
    main()