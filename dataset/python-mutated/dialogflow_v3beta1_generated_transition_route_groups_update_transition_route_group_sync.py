from google.cloud import dialogflowcx_v3beta1

def sample_update_transition_route_group():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3beta1.TransitionRouteGroupsClient()
    transition_route_group = dialogflowcx_v3beta1.TransitionRouteGroup()
    transition_route_group.display_name = 'display_name_value'
    request = dialogflowcx_v3beta1.UpdateTransitionRouteGroupRequest(transition_route_group=transition_route_group)
    response = client.update_transition_route_group(request=request)
    print(response)