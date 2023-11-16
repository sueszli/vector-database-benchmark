from google.cloud import vmwareengine_v1

def sample_list_private_connection_peering_routes():
    if False:
        print('Hello World!')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.ListPrivateConnectionPeeringRoutesRequest(parent='parent_value')
    page_result = client.list_private_connection_peering_routes(request=request)
    for response in page_result:
        print(response)