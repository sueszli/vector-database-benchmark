from google.cloud import compute_v1

def sample_insert():
    if False:
        print('Hello World!')
    client = compute_v1.RegionBackendServicesClient()
    request = compute_v1.InsertRegionBackendServiceRequest(project='project_value', region='region_value')
    response = client.insert(request=request)
    print(response)