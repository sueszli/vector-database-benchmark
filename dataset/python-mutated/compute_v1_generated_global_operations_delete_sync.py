from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.GlobalOperationsClient()
    request = compute_v1.DeleteGlobalOperationRequest(operation='operation_value', project='project_value')
    response = client.delete(request=request)
    print(response)