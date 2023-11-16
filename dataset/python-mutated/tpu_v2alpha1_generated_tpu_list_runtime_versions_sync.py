from google.cloud import tpu_v2alpha1

def sample_list_runtime_versions():
    if False:
        return 10
    client = tpu_v2alpha1.TpuClient()
    request = tpu_v2alpha1.ListRuntimeVersionsRequest(parent='parent_value')
    page_result = client.list_runtime_versions(request=request)
    for response in page_result:
        print(response)