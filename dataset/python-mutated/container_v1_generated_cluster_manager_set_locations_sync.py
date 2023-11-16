from google.cloud import container_v1

def sample_set_locations():
    if False:
        while True:
            i = 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.SetLocationsRequest(locations=['locations_value1', 'locations_value2'])
    response = client.set_locations(request=request)
    print(response)