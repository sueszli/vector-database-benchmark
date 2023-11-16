from google.cloud import dialogflowcx_v3

def sample_delete_transition_route_group():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.TransitionRouteGroupsClient()
    request = dialogflowcx_v3.DeleteTransitionRouteGroupRequest(name='name_value')
    client.delete_transition_route_group(request=request)