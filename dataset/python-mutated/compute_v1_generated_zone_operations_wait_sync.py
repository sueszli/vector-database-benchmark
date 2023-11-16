from google.cloud import compute_v1

def sample_wait():
    if False:
        i = 10
        return i + 15
    client = compute_v1.ZoneOperationsClient()
    request = compute_v1.WaitZoneOperationRequest(operation='operation_value', project='project_value', zone='zone_value')
    response = client.wait(request=request)
    print(response)