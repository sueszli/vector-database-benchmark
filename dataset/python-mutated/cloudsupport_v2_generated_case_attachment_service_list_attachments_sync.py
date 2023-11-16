from google.cloud import support_v2

def sample_list_attachments():
    if False:
        while True:
            i = 10
    client = support_v2.CaseAttachmentServiceClient()
    request = support_v2.ListAttachmentsRequest(parent='parent_value')
    page_result = client.list_attachments(request=request)
    for response in page_result:
        print(response)