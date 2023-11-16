from google.cloud import accessapproval_v1

def sample_invalidate_approval_request():
    if False:
        while True:
            i = 10
    client = accessapproval_v1.AccessApprovalClient()
    request = accessapproval_v1.InvalidateApprovalRequestMessage()
    response = client.invalidate_approval_request(request=request)
    print(response)