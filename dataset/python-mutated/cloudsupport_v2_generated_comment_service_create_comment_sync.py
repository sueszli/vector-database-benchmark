from google.cloud import support_v2

def sample_create_comment():
    if False:
        while True:
            i = 10
    client = support_v2.CommentServiceClient()
    request = support_v2.CreateCommentRequest(parent='parent_value')
    response = client.create_comment(request=request)
    print(response)