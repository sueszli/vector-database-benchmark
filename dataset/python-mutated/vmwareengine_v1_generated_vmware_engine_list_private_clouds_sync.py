from google.cloud import vmwareengine_v1

def sample_list_private_clouds():
    if False:
        while True:
            i = 10
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.ListPrivateCloudsRequest(parent='parent_value')
    page_result = client.list_private_clouds(request=request)
    for response in page_result:
        print(response)