"""This application demonstrates how to perform basic operations on dataset
with the Google AutoML Tables API.

For more information, the documentation at
https://cloud.google.com/automl-tables/docs.
"""
import argparse
import os

def create_dataset(project_id, compute_region, dataset_display_name):
    if False:
        while True:
            i = 10
    'Create a dataset.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    dataset = client.create_dataset(dataset_display_name)
    print(f'Dataset name: {dataset.name}')
    print('Dataset id: {}'.format(dataset.name.split('/')[-1]))
    print(f'Dataset display name: {dataset.display_name}')
    print('Dataset metadata:')
    print(f'\t{dataset.tables_dataset_metadata}')
    print(f'Dataset example count: {dataset.example_count}')
    print(f'Dataset create time: {dataset.create_time}')
    return dataset

def list_datasets(project_id, compute_region, filter=None):
    if False:
        for i in range(10):
            print('nop')
    'List all datasets.'
    result = []
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    response = client.list_datasets(filter=filter)
    print('List of datasets:')
    for dataset in response:
        print(f'Dataset name: {dataset.name}')
        print('Dataset id: {}'.format(dataset.name.split('/')[-1]))
        print(f'Dataset display name: {dataset.display_name}')
        metadata = dataset.tables_dataset_metadata
        print('Dataset primary table spec id: {}'.format(metadata.primary_table_spec_id))
        print('Dataset target column spec id: {}'.format(metadata.target_column_spec_id))
        print('Dataset target column spec id: {}'.format(metadata.target_column_spec_id))
        print('Dataset weight column spec id: {}'.format(metadata.weight_column_spec_id))
        print('Dataset ml use column spec id: {}'.format(metadata.ml_use_column_spec_id))
        print(f'Dataset example count: {dataset.example_count}')
        print(f'Dataset create time: {dataset.create_time}')
        print('\n')
        result.append(dataset)
    return result

def get_dataset(project_id, compute_region, dataset_display_name):
    if False:
        while True:
            i = 10
    'Get the dataset.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    dataset = client.get_dataset(dataset_display_name=dataset_display_name)
    print(f'Dataset name: {dataset.name}')
    print('Dataset id: {}'.format(dataset.name.split('/')[-1]))
    print(f'Dataset display name: {dataset.display_name}')
    print('Dataset metadata:')
    print(f'\t{dataset.tables_dataset_metadata}')
    print(f'Dataset example count: {dataset.example_count}')
    print(f'Dataset create time: {dataset.create_time}')
    return dataset

def import_data(project_id, compute_region, dataset_display_name, path, dataset_name=None):
    if False:
        print('Hello World!')
    'Import structured data.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    response = None
    if path.startswith('bq'):
        response = client.import_data(dataset_display_name=dataset_display_name, bigquery_input_uri=path, dataset_name=dataset_name)
    else:
        input_uris = path.split(',')
        response = client.import_data(dataset_display_name=dataset_display_name, gcs_input_uris=input_uris, dataset_name=dataset_name)
    print('Processing import...')
    print(f'Data imported. {response.result()}')

def update_dataset(project_id, compute_region, dataset_display_name, target_column_spec_name=None, weight_column_spec_name=None, test_train_column_spec_name=None):
    if False:
        print('Hello World!')
    'Update dataset.'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    if target_column_spec_name is not None:
        response = client.set_target_column(dataset_display_name=dataset_display_name, column_spec_display_name=target_column_spec_name)
        print(f'Target column updated. {response}')
    if weight_column_spec_name is not None:
        response = client.set_weight_column(dataset_display_name=dataset_display_name, column_spec_display_name=weight_column_spec_name)
        print(f'Weight column updated. {response}')
    if test_train_column_spec_name is not None:
        response = client.set_test_train_column(dataset_display_name=dataset_display_name, column_spec_display_name=test_train_column_spec_name)
        print(f'Test/train column updated. {response}')

def delete_dataset(project_id, compute_region, dataset_display_name):
    if False:
        i = 10
        return i + 15
    'Delete a dataset'
    from google.cloud import automl_v1beta1 as automl
    client = automl.TablesClient(project=project_id, region=compute_region)
    response = client.delete_dataset(dataset_display_name=dataset_display_name)
    print(f'Dataset deleted. {response.result()}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    create_dataset_parser = subparsers.add_parser('create_dataset', help=create_dataset.__doc__)
    create_dataset_parser.add_argument('--dataset_name')
    list_datasets_parser = subparsers.add_parser('list_datasets', help=list_datasets.__doc__)
    list_datasets_parser.add_argument('--filter_')
    get_dataset_parser = subparsers.add_parser('get_dataset', help=get_dataset.__doc__)
    get_dataset_parser.add_argument('--dataset_display_name')
    import_data_parser = subparsers.add_parser('import_data', help=import_data.__doc__)
    import_data_parser.add_argument('--dataset_display_name')
    import_data_parser.add_argument('--path')
    update_dataset_parser = subparsers.add_parser('update_dataset', help=update_dataset.__doc__)
    update_dataset_parser.add_argument('--dataset_display_name')
    update_dataset_parser.add_argument('--target_column_spec_name')
    update_dataset_parser.add_argument('--weight_column_spec_name')
    update_dataset_parser.add_argument('--ml_use_column_spec_name')
    delete_dataset_parser = subparsers.add_parser('delete_dataset', help=delete_dataset.__doc__)
    delete_dataset_parser.add_argument('--dataset_display_name')
    project_id = os.environ['PROJECT_ID']
    compute_region = os.environ['REGION_NAME']
    args = parser.parse_args()
    if args.command == 'create_dataset':
        create_dataset(project_id, compute_region, args.dataset_name)
    if args.command == 'list_datasets':
        list_datasets(project_id, compute_region, args.filter_)
    if args.command == 'get_dataset':
        get_dataset(project_id, compute_region, args.dataset_display_name)
    if args.command == 'import_data':
        import_data(project_id, compute_region, args.dataset_display_name, args.path)
    if args.command == 'update_dataset':
        update_dataset(project_id, compute_region, args.dataset_display_name, args.target_column_spec_name, args.weight_column_spec_name, args.ml_use_column_spec_name)
    if args.command == 'delete_dataset':
        delete_dataset(project_id, compute_region, args.dataset_display_name)