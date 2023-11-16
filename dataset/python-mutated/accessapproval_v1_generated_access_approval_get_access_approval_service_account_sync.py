from google.cloud import accessapproval_v1

def sample_get_access_approval_service_account():
    if False:
        i = 10
        return i + 15
    client = accessapproval_v1.AccessApprovalClient()
    request = accessapproval_v1.GetAccessApprovalServiceAccountMessage()
    response = client.get_access_approval_service_account(request=request)
    print(response)