from google.cloud import compute_v1

def sample_get():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.InterconnectRemoteLocationsClient()
    request = compute_v1.GetInterconnectRemoteLocationRequest(interconnect_remote_location='interconnect_remote_location_value', project='project_value')
    response = client.get(request=request)
    print(response)