from google.cloud import accessapproval_v1

def sample_dismiss_approval_request():
    if False:
        i = 10
        return i + 15
    client = accessapproval_v1.AccessApprovalClient()
    request = accessapproval_v1.DismissApprovalRequestMessage()
    response = client.dismiss_approval_request(request=request)
    print(response)