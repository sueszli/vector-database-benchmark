from google.cloud import dialogflowcx_v3

def sample_update_transition_route_group():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.TransitionRouteGroupsClient()
    transition_route_group = dialogflowcx_v3.TransitionRouteGroup()
    transition_route_group.display_name = 'display_name_value'
    request = dialogflowcx_v3.UpdateTransitionRouteGroupRequest(transition_route_group=transition_route_group)
    response = client.update_transition_route_group(request=request)
    print(response)