from google.cloud import vmwareengine_v1

def sample_list_network_policies():
    if False:
        i = 10
        return i + 15
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.ListNetworkPoliciesRequest(parent='parent_value')
    page_result = client.list_network_policies(request=request)
    for response in page_result:
        print(response)