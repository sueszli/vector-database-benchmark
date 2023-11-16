from google.cloud import tpu_v2alpha1

def sample_list_nodes():
    if False:
        print('Hello World!')
    client = tpu_v2alpha1.TpuClient()
    request = tpu_v2alpha1.ListNodesRequest(parent='parent_value')
    page_result = client.list_nodes(request=request)
    for response in page_result:
        print(response)