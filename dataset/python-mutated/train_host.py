"""
Purpose

Shows how to create and optionally start an Amazon Lookout for Vision model.
"""
import argparse
import logging
import boto3
from projects import Projects
from datasets import Datasets
from models import Models
from hosting import Hosting
logger = logging.getLogger(__name__)

def start_model(lookoutvision_client, project_name, version):
    if False:
        i = 10
        return i + 15
    '\n    Starts a model, if requested.\n\n    :param lookoutvision_client: A Boto3 Lookout for Vision client.\n    :param project_name: The name of the project that contains the model version\n                         you want to start.\n    :param: version: The version of the model that you want to start.\n    '
    start = input('Do you want to start your model (y/n)?')
    if start == 'y':
        print('Starting model...')
        Hosting.start_model(lookoutvision_client, project_name, version, 1)
        print('Your model is ready to use with the following command.\n')
        print(f'python inference.py {project_name} {version} <your_image>')
        print("\nStop your model when you're done. You're charged while it's running. See hosting.py")
    else:
        print('Not starting model.')

def create_dataset(lookoutvision_client, s3_resource, bucket, project_name, dataset_images, dataset_type):
    if False:
        while True:
            i = 10
    '\n    Creates a manifest from images in the supplied bucket and then creates\n    a dataset.\n\n    :param lookoutvision_client: A Boto3 Lookout for Vision client.\n    :param s3_resource: A Boto3 Amazon S3 client.\n    :param bucket: The bucket that stores the manifest file.\n    :param project_name: The project in which to create the dataset.\n    :param dataset_images: The location of the images referenced by the dataset.\n    :param dataset_type: The type of dataset to create (train or test).\n    '
    print(f'Creating {dataset_type} dataset...')
    manifest_file = f's3://{bucket}/{project_name}/manifests/{dataset_type}.manifest'
    logger.info('Creating %s manifest file in %s.', dataset_type, manifest_file)
    Datasets.create_manifest_file_s3(s3_resource, dataset_images, manifest_file)
    logger.info('Create %s dataset for project %s', dataset_type, project_name)
    Datasets.create_dataset(lookoutvision_client, project_name, manifest_file, dataset_type)

def train_model(lookoutvision_client, bucket, project_name):
    if False:
        while True:
            i = 10
    '\n    Trains a model.\n\n    :param lookoutvision_client: A Boto3 Lookout for Vision client.\n    :param bucket: The bucket where the training output is stored.\n    :param project_name: The project that you want to train.\n    '
    print('Training model...')
    training_results = f'{bucket}/{project_name}/output/'
    (status, version) = Models.create_model(lookoutvision_client, project_name, training_results)
    Models.describe_model(lookoutvision_client, project_name, version)
    if status == 'TRAINED':
        print('\nCheck the performance metrics and decide if you need to improve the model performance.')
        print('\nMore information: https://docs.aws.amazon.com/lookout-for-vision/latest/developer-guide/improve.html')
        print('If you are satisfied with your model, you can start it.')
        start_model(lookoutvision_client, project_name, version)
    else:
        print('Model training failed.')

def main():
    if False:
        print('Hello World!')
    '\n    Creates and optionally starts an Amazon Lookout for Vision model using\n    command line arguments.\n\n    A new project, training dataset, optional test dataset, and model are created.\n    After model training is completed, you can use the code in inference.py to try your\n    model with an image.\n    For the training and test folders, place normal images in a folder named normal and\n    anomalous images in a folder named anomaly.\n    Make sure that bucket and the training/test Amazon S3 paths are in the same AWS Region.\n    '
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
    parser.add_argument('project', help='A unique name for your project')
    parser.add_argument('bucket', help='The bucket used to upload your manifest files and store training output')
    parser.add_argument('training', help='The Amazon S3 path where the service gets the training images. ')
    parser.add_argument('test', nargs='?', default=None, help='(Optional) The Amazon S3 path where the service gets the test images.')
    args = parser.parse_args()
    project_name = args.project
    bucket = args.bucket
    training_images = args.training
    test_images = args.test
    session = boto3.Session(profile_name='lookoutvision-access')
    lookoutvision_client = session.client('lookoutvision')
    s3_resource = session.resource('s3')
    print(f'Storing information in s3://{bucket}/{project_name}/')
    print('Creating project...')
    Projects.create_project(lookoutvision_client, project_name)
    create_dataset(lookoutvision_client, s3_resource, bucket, project_name, training_images, 'train')
    if test_images is not None:
        create_dataset(lookoutvision_client, s3_resource, bucket, project_name, test_images, 'test')
    train_model(lookoutvision_client, bucket, project_name)
if __name__ == '__main__':
    main()