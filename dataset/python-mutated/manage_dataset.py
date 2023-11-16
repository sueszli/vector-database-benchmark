import argparse
import os
from google.api_core.client_options import ClientOptions

def create_dataset(project_id):
    if False:
        while True:
            i = 10
    'Creates a dataset for the given Google Cloud project.'
    from google.cloud import datalabeling_v1beta1 as datalabeling
    client = datalabeling.DataLabelingServiceClient()
    if 'DATALABELING_ENDPOINT' in os.environ:
        opts = ClientOptions(api_endpoint=os.getenv('DATALABELING_ENDPOINT'))
        client = datalabeling.DataLabelingServiceClient(client_options=opts)
    formatted_project_name = f'projects/{project_id}'
    dataset = datalabeling.Dataset(display_name='YOUR_DATASET_SET_DISPLAY_NAME', description='YOUR_DESCRIPTION')
    response = client.create_dataset(request={'parent': formatted_project_name, 'dataset': dataset})
    print(f'The dataset resource name: {response.name}')
    print(f'Display name: {response.display_name}')
    print(f'Description: {response.description}')
    print('Create time:')
    print(f'\tseconds: {response.create_time.timestamp_pb().seconds}')
    print(f'\tnanos: {response.create_time.timestamp_pb().nanos}\n')
    return response

def list_datasets(project_id):
    if False:
        print('Hello World!')
    'Lists datasets for the given Google Cloud project.'
    from google.cloud import datalabeling_v1beta1 as datalabeling
    client = datalabeling.DataLabelingServiceClient()
    if 'DATALABELING_ENDPOINT' in os.environ:
        opts = ClientOptions(api_endpoint=os.getenv('DATALABELING_ENDPOINT'))
        client = datalabeling.DataLabelingServiceClient(client_options=opts)
    formatted_project_name = f'projects/{project_id}'
    response = client.list_datasets(request={'parent': formatted_project_name})
    for element in response:
        print(f'The dataset resource name: {element.name}\n')
        print(f'Display name: {element.display_name}')
        print(f'Description: {element.description}')
        print('Create time:')
        print(f'\tseconds: {element.create_time.timestamp_pb().seconds}')
        print(f'\tnanos: {element.create_time.timestamp_pb().nanos}')

def get_dataset(dataset_resource_name):
    if False:
        i = 10
        return i + 15
    'Gets a dataset for the given Google Cloud project.'
    from google.cloud import datalabeling_v1beta1 as datalabeling
    client = datalabeling.DataLabelingServiceClient()
    if 'DATALABELING_ENDPOINT' in os.environ:
        opts = ClientOptions(api_endpoint=os.getenv('DATALABELING_ENDPOINT'))
        client = datalabeling.DataLabelingServiceClient(client_options=opts)
    response = client.get_dataset(request={'name': dataset_resource_name})
    print(f'The dataset resource name: {response.name}\n')
    print(f'Display name: {response.display_name}')
    print(f'Description: {response.description}')
    print('Create time:')
    print(f'\tseconds: {response.create_time.timestamp_pb().seconds}')
    print(f'\tnanos: {response.create_time.timestamp_pb().nanos}')

def delete_dataset(dataset_resource_name):
    if False:
        while True:
            i = 10
    'Deletes a dataset for the given Google Cloud project.'
    from google.cloud import datalabeling_v1beta1 as datalabeling
    client = datalabeling.DataLabelingServiceClient()
    if 'DATALABELING_ENDPOINT' in os.environ:
        opts = ClientOptions(api_endpoint=os.getenv('DATALABELING_ENDPOINT'))
        client = datalabeling.DataLabelingServiceClient(client_options=opts)
    response = client.delete_dataset(request={'name': dataset_resource_name})
    print(f'Dataset deleted. {response}\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    create_parser = subparsers.add_parser('create', help='Create a new dataset.')
    create_parser.add_argument('--project-id', help='Project ID. Required.', required=True)
    list_parser = subparsers.add_parser('list', help='List all datasets.')
    list_parser.add_argument('--project-id', help='Project ID. Required.', required=True)
    get_parser = subparsers.add_parser('get', help='Get a dataset by the dataset resource name.')
    get_parser.add_argument('--dataset-resource-name', help='The dataset resource name. Used in the get or delete operation.', required=True)
    delete_parser = subparsers.add_parser('delete', help='Delete a dataset by the dataset resource name.')
    delete_parser.add_argument('--dataset-resource-name', help='The dataset resource name. Used in the get or delete operation.', required=True)
    args = parser.parse_args()
    if args.command == 'create':
        create_dataset(args.project_id)
    elif args.command == 'list':
        list_datasets(args.project_id)
    elif args.command == 'get':
        get_dataset(args.dataset_resource_name)
    elif args.command == 'delete':
        delete_dataset(args.dataset_resource_name)