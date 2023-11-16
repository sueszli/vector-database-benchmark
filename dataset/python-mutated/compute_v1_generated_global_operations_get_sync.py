from google.cloud import compute_v1

def sample_get():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.GlobalOperationsClient()
    request = compute_v1.GetGlobalOperationRequest(operation='operation_value', project='project_value')
    response = client.get(request=request)
    print(response)