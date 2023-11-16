from google.cloud import tpu_v2

def sample_list_nodes():
    if False:
        for i in range(10):
            print('nop')
    client = tpu_v2.TpuClient()
    request = tpu_v2.ListNodesRequest(parent='parent_value')
    page_result = client.list_nodes(request=request)
    for response in page_result:
        print(response)