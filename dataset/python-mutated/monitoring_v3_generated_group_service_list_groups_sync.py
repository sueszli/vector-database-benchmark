from google.cloud import monitoring_v3

def sample_list_groups():
    if False:
        while True:
            i = 10
    client = monitoring_v3.GroupServiceClient()
    request = monitoring_v3.ListGroupsRequest(children_of_group='children_of_group_value', name='name_value')
    page_result = client.list_groups(request=request)
    for response in page_result:
        print(response)