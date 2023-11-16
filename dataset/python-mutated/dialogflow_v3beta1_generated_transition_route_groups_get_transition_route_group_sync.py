from google.cloud import dialogflowcx_v3beta1

def sample_get_transition_route_group():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.TransitionRouteGroupsClient()
    request = dialogflowcx_v3beta1.GetTransitionRouteGroupRequest(name='name_value')
    response = client.get_transition_route_group(request=request)
    print(response)