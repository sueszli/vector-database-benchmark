"""
Purpose

Amazon Lookout for Vision dataset code examples used in the service documentation:
https://docs.aws.amazon.com/lookout-for-vision/latest/developer-guide/edit-dataset.html
Shows how to update a Lookout for Vision dataset with a manifest file.
"""
import logging
import argparse
import boto3
from botocore.exceptions import ClientError
from datasets import Datasets
logger = logging.getLogger(__name__)

def add_arguments(parser):
    if False:
        for i in range(10):
            print('nop')
    '\n    Adds command line arguments to the parser.\n    :param parser: The command line parser.\n    '
    parser.add_argument('project_name', help='The Amazon Resource Name (ARN) of the dataset that you want to update.')
    parser.add_argument('dataset_type', help='The type of the dataset that you want to update (train or test).')
    parser.add_argument('updates_file', help='The manifest file of JSON lines that contains the updates.')

def main():
    if False:
        for i in range(10):
            print('nop')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
        add_arguments(parser)
        args = parser.parse_args()
        print(f'Updating {args.dataset_type} dataset for project {args.project_name} with entries from {{args.updates_file}}.')
        session = boto3.Session(profile_name='lookoutvision-access')
        lookoutvision_client = session.client('lookoutvision')
        (status, status_message) = Datasets.update_dataset_entries(lookoutvision_client, args.project_name, args.dataset_type, args.updates_file)
        print(f'Finished updates dataset: {status} : {status_message}')
    except ClientError as err:
        logger.exception('Problem updating dataset: %s', err)
        print(f'Problem updating dataset: {err}')
    except Exception as err:
        logger.exception('Problem updating dataset: %s', err)
        print(f'Problem updating dataset: {err}')
if __name__ == '__main__':
    main()