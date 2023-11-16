from google.cloud import dialogflowcx_v3

def sample_get_transition_route_group():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.TransitionRouteGroupsClient()
    request = dialogflowcx_v3.GetTransitionRouteGroupRequest(name='name_value')
    response = client.get_transition_route_group(request=request)
    print(response)