from google.cloud import tpu_v1

def sample_list_nodes():
    if False:
        i = 10
        return i + 15
    client = tpu_v1.TpuClient()
    request = tpu_v1.ListNodesRequest(parent='parent_value')
    page_result = client.list_nodes(request=request)
    for response in page_result:
        print(response)