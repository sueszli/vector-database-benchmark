from google.cloud import compute_v1

def sample_get():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.GlobalOrganizationOperationsClient()
    request = compute_v1.GetGlobalOrganizationOperationRequest(operation='operation_value')
    response = client.get(request=request)
    print(response)