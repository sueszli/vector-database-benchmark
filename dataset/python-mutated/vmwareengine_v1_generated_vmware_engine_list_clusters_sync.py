from google.cloud import vmwareengine_v1

def sample_list_clusters():
    if False:
        print('Hello World!')
    client = vmwareengine_v1.VmwareEngineClient()
    request = vmwareengine_v1.ListClustersRequest(parent='parent_value')
    page_result = client.list_clusters(request=request)
    for response in page_result:
        print(response)