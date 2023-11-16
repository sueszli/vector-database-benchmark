from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.ZoneOperationsClient()
    request = compute_v1.DeleteZoneOperationRequest(operation='operation_value', project='project_value', zone='zone_value')
    response = client.delete(request=request)
    print(response)