from google.cloud import datastream_v1alpha1

def sample_fetch_errors():
    if False:
        return 10
    client = datastream_v1alpha1.DatastreamClient()
    request = datastream_v1alpha1.FetchErrorsRequest()
    operation = client.fetch_errors(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)