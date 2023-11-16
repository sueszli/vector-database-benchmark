from google.cloud import dialogflowcx_v3beta1

def sample_list_transition_route_groups():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.TransitionRouteGroupsClient()
    request = dialogflowcx_v3beta1.ListTransitionRouteGroupsRequest(parent='parent_value')
    page_result = client.list_transition_route_groups(request=request)
    for response in page_result:
        print(response)