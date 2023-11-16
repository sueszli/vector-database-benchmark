from google.cloud import support_v2

def sample_list_comments():
    if False:
        for i in range(10):
            print('nop')
    client = support_v2.CommentServiceClient()
    request = support_v2.ListCommentsRequest(parent='parent_value')
    page_result = client.list_comments(request=request)
    for response in page_result:
        print(response)