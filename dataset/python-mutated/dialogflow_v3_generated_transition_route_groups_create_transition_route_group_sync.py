from google.cloud import dialogflowcx_v3

def sample_create_transition_route_group():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.TransitionRouteGroupsClient()
    transition_route_group = dialogflowcx_v3.TransitionRouteGroup()
    transition_route_group.display_name = 'display_name_value'
    request = dialogflowcx_v3.CreateTransitionRouteGroupRequest(parent='parent_value', transition_route_group=transition_route_group)
    response = client.create_transition_route_group(request=request)
    print(response)