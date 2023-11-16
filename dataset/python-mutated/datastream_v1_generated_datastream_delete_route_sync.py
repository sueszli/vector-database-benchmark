from google.cloud import datastream_v1

def sample_delete_route():
    if False:
        for i in range(10):
            print('nop')
    client = datastream_v1.DatastreamClient()
    request = datastream_v1.DeleteRouteRequest(name='name_value')
    operation = client.delete_route(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)