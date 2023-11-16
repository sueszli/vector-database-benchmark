from google.cloud import dialogflowcx_v3beta1

def sample_create_transition_route_group():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3beta1.TransitionRouteGroupsClient()
    transition_route_group = dialogflowcx_v3beta1.TransitionRouteGroup()
    transition_route_group.display_name = 'display_name_value'
    request = dialogflowcx_v3beta1.CreateTransitionRouteGroupRequest(parent='parent_value', transition_route_group=transition_route_group)
    response = client.create_transition_route_group(request=request)
    print(response)