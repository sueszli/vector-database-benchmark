from google.cloud import tpu_v1

def sample_list_tensor_flow_versions():
    if False:
        i = 10
        return i + 15
    client = tpu_v1.TpuClient()
    request = tpu_v1.ListTensorFlowVersionsRequest(parent='parent_value')
    page_result = client.list_tensor_flow_versions(request=request)
    for response in page_result:
        print(response)