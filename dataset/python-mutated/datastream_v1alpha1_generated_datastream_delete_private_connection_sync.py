from google.cloud import datastream_v1alpha1

def sample_delete_private_connection():
    if False:
        while True:
            i = 10
    client = datastream_v1alpha1.DatastreamClient()
    request = datastream_v1alpha1.DeletePrivateConnectionRequest(name='name_value')
    operation = client.delete_private_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)