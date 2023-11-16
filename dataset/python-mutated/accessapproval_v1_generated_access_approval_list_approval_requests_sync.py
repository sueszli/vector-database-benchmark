from google.cloud import accessapproval_v1

def sample_list_approval_requests():
    if False:
        for i in range(10):
            print('nop')
    client = accessapproval_v1.AccessApprovalClient()
    request = accessapproval_v1.ListApprovalRequestsMessage()
    page_result = client.list_approval_requests(request=request)
    for response in page_result:
        print(response)