from google.cloud import datastream_v1

def sample_get_route():
    if False:
        i = 10
        return i + 15
    client = datastream_v1.DatastreamClient()
    request = datastream_v1.GetRouteRequest(name='name_value')
    response = client.get_route(request=request)
    print(response)