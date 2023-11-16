from google.cloud import datastream_v1alpha1

def sample_get_route():
    if False:
        for i in range(10):
            print('nop')
    client = datastream_v1alpha1.DatastreamClient()
    request = datastream_v1alpha1.GetRouteRequest(name='name_value')
    response = client.get_route(request=request)
    print(response)