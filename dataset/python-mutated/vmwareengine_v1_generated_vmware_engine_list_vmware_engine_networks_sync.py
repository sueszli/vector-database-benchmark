from google.cloud import vmwareengine_v1

def sample_list_vmware_engine_networks():
    if False:
        i = 10
        return i + 15
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.ListVmwareEngineNetworksRequest(parent='parent_value')
    page_result = client.list_vmware_engine_networks(request=request)
    for response in page_result:
        print(response)