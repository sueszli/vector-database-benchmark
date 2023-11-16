from google.cloud import dialogflowcx_v3

def sample_list_transition_route_groups():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.TransitionRouteGroupsClient()
    request = dialogflowcx_v3.ListTransitionRouteGroupsRequest(parent='parent_value')
    page_result = client.list_transition_route_groups(request=request)
    for response in page_result:
        print(response)