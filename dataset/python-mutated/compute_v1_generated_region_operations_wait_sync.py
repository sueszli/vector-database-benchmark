from google.cloud import compute_v1

def sample_wait():
    if False:
        return 10
    client = compute_v1.RegionOperationsClient()
    request = compute_v1.WaitRegionOperationRequest(operation='operation_value', project='project_value', region='region_value')
    response = client.wait(request=request)
    print(response)