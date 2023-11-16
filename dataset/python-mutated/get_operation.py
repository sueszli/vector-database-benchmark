from google.cloud import optimization_v1

def get_operation(operation_full_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Get operation details and status.'
    client = optimization_v1.FleetRoutingClient()
    response = client.transport.operations_client.get_operation(operation_full_id)
    print(f'Name: {response.name}')
    print('Operation details:')
    print(response)