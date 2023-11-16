def list_operation_status(project_id):
    if False:
        return 10
    'List operation status.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    project_location = f'projects/{project_id}/locations/us-central1'
    response = client._transport.operations_client.list_operations(name=project_location, filter_='', timeout=5)
    print('List of operations:')
    for operation in response:
        print(f'Name: {operation.name}')
        print('Operation details:')
        print(operation)