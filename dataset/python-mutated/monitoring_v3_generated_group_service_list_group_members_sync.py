from google.cloud import monitoring_v3

def sample_list_group_members():
    if False:
        print('Hello World!')
    client = monitoring_v3.GroupServiceClient()
    request = monitoring_v3.ListGroupMembersRequest(name='name_value')
    page_result = client.list_group_members(request=request)
    for response in page_result:
        print(response)