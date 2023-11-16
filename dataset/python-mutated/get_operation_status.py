def get_operation_status(operation_full_id):
    if False:
        while True:
            i = 10
    'Get operation status.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    response = client._transport.operations_client.get_operation(operation_full_id)
    print(f'Name: {response.name}')
    print('Operation details:')
    print(response)