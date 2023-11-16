from google.cloud import vmwareengine_v1

def sample_list_subnets():
    if False:
        for i in range(10):
            print('nop')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.ListSubnetsRequest(parent='parent_value')
    page_result = client.list_subnets(request=request)
    for response in page_result:
        print(response)