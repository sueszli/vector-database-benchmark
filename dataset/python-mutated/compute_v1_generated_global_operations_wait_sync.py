from google.cloud import compute_v1

def sample_wait():
    if False:
        i = 10
        return i + 15
    client = compute_v1.GlobalOperationsClient()
    request = compute_v1.WaitGlobalOperationRequest(operation='operation_value', project='project_value')
    response = client.wait(request=request)
    print(response)