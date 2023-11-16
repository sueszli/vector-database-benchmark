from google.cloud import accessapproval_v1

def sample_get_approval_request():
    if False:
        for i in range(10):
            print('nop')
    client = accessapproval_v1.AccessApprovalClient()
    request = accessapproval_v1.GetApprovalRequestMessage()
    response = client.get_approval_request(request=request)
    print(response)