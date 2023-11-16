from google.cloud import datastream_v1

def sample_create_route():
    if False:
        while True:
            i = 10
    client = datastream_v1.DatastreamClient()
    route = datastream_v1.Route()
    route.display_name = 'display_name_value'
    route.destination_address = 'destination_address_value'
    request = datastream_v1.CreateRouteRequest(parent='parent_value', route_id='route_id_value', route=route)
    operation = client.create_route(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)