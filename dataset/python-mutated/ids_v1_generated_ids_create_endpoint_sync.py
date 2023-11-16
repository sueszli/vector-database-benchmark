from google.cloud import ids_v1

def sample_create_endpoint():
    if False:
        while True:
            i = 10
    client = ids_v1.IDSClient()
    endpoint = ids_v1.Endpoint()
    endpoint.network = 'network_value'
    endpoint.severity = 'CRITICAL'
    request = ids_v1.CreateEndpointRequest(parent='parent_value', endpoint_id='endpoint_id_value', endpoint=endpoint)
    operation = client.create_endpoint(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)