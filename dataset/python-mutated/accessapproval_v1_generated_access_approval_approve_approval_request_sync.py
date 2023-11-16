from google.cloud import accessapproval_v1

def sample_approve_approval_request():
    if False:
        print('Hello World!')
    client = accessapproval_v1.AccessApprovalClient()
    request = accessapproval_v1.ApproveApprovalRequestMessage()
    response = client.approve_approval_request(request=request)
    print(response)