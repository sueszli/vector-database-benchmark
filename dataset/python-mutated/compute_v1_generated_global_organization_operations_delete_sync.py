from google.cloud import compute_v1

def sample_delete():
    if False:
        i = 10
        return i + 15
    client = compute_v1.GlobalOrganizationOperationsClient()
    request = compute_v1.DeleteGlobalOrganizationOperationRequest(operation='operation_value')
    response = client.delete(request=request)
    print(response)