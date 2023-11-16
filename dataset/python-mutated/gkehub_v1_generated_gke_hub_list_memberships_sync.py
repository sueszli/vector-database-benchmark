from google.cloud import gkehub_v1

def sample_list_memberships():
    if False:
        i = 10
        return i + 15
    client = gkehub_v1.GkeHubClient()
    request = gkehub_v1.ListMembershipsRequest(parent='parent_value')
    page_result = client.list_memberships(request=request)
    for response in page_result:
        print(response)