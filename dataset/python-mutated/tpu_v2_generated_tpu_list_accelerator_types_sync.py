from google.cloud import tpu_v2

def sample_list_accelerator_types():
    if False:
        print('Hello World!')
    client = tpu_v2.TpuClient()
    request = tpu_v2.ListAcceleratorTypesRequest(parent='parent_value')
    page_result = client.list_accelerator_types(request=request)
    for response in page_result:
        print(response)