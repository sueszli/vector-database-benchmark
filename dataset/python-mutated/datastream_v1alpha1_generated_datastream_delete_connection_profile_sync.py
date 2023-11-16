from google.cloud import datastream_v1alpha1

def sample_delete_connection_profile():
    if False:
        print('Hello World!')
    client = datastream_v1alpha1.DatastreamClient()
    request = datastream_v1alpha1.DeleteConnectionProfileRequest(name='name_value')
    operation = client.delete_connection_profile(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)